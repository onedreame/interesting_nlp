# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import sys
from glob import glob
import argparse
import logging
from pprint import pformat

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import CONFIG_NAME, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import (OpenAIGPTLMHeadModel, GPT2LMHeadModel,
                          OpenAIGPTConfig, GPT2Config, BertTokenizer)
from pytorch_lightning import Trainer

sys.path.append(os.getcwd())
from datautils import LSCCDataSet
from module import LabelSmoothing, huggingface_initilize
from trainer import BaseTrainer

logger = logging.getLogger('lightning')
logger.setLevel(logging.INFO)
logger.propagate = False

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'),
                  filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now
    top_k = min(top_k, logits.size(-1))
    # print('initial: ', logits[:10])
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        # print(f"top k: {logits[:10]} sum:{(logits!=filter_value).sum()}")

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # print(f"sorted logits:{sorted_logits[:args.top_k+4]}")
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # print(cumulative_probabilities[:10])

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    # print(f"final: {logits[:10]} elements: {(logits!=filter_value).sum()}")
    # if temperature is not None:
    #     logits /= temperature
    return logits

def generate(args, history, tokenizer, model, speaker1, speaker2, special_token_ids):
    current_output = []
    with torch.no_grad():
        for i in range(args.max_len):
            input_dict = LSCCDataSet.encode(history, [current_output], tokenizer, speaker1, speaker2, do_encode=False)
            new_user_input_ids, token_type_ids = input_dict['input_ids'], input_dict['token_type_ids']
            new_user_input_ids_tensor = torch.tensor(new_user_input_ids, dtype=torch.long).unsqueeze(0).cuda()
            token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).cuda()
            outputs = model(new_user_input_ids_tensor, token_type_ids=token_type_ids_tensor)
            print(f"input:{tokenizer.convert_ids_to_tokens(new_user_input_ids)}")
            logits = outputs[0,-1,:] / args.temperature
            # logits = outputs[0][0, -1, :]
            logits = top_filtering(logits, args.top_k,top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            # print(f"step:{i} token:{tokenizer.convert_ids_to_tokens(prev.item())} output[0]:{outputs[0].size()}"
            #       f"current output:{tokenizer.convert_ids_to_tokens(current_output)}")
            if i < args.min_length and prev.item() in special_token_ids:
                while prev.item() in special_token_ids:
                    print("resample: ", tokenizer.convert_ids_to_tokens(prev.item()))
                    # exit(0)
                    prev = torch.multinomial(probs, 1)
            if prev.item() in special_token_ids:
                break
            current_output.append(prev.item())
    return current_output

# def interact(args, model, tokenizer):
#     user_inputs = input(">> user:")
#     speaker1, speaker2 = LSCCDataSet.get_identity_id(tokenizer)
#
#     special_token_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, speaker2, speaker1]
#     tokenizer.add_special_tokens({"additional_special_tokens": [tokenizer.convert_ids_to_tokens(_id)
#                                                                 for _id in [speaker1, speaker2]]})
#     print(tokenizer.all_special_tokens)
#     history = []
#     while user_inputs != "bye":
#         while not user_inputs:
#             print("输入不能为空")
#             user_inputs = input(">> user:")
#         user_inputs = " ".join(list(user_inputs.replace(" ", "")))
#         history.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(user_inputs)))
#         output_id = generate(args, history, tokenizer, model, speaker1, speaker2, special_token_ids)
#         print(output_id)
#         history.append(output_id)
#         history = history[-(2*args.max_history+1):]
#         print("Bot: ", tokenizer.decode(output_id,skip_special_tokens=True))
#         user_inputs = input(">> user:")

def huggingface_initialize_model():
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
    tokenizer_class = BertTokenizer
    if args.pretrained:
        # 不收敛的情况下检查学习率和初始化的模型是否有问题， dataset的shuffle也可能导致收敛效果不好
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True,
                                                    never_split=["[speaker1]", "[speaker2]"])
        model = model_class.from_pretrained(args.model_checkpoint)
    else:
        tokenizer = tokenizer_class(os.path.join(args.model_checkpoint, "vocab.txt"), do_lower_case=True,
                                    never_split=["[speaker1]", "[speaker2]"])
        config = config_class.from_json_file(os.path.join(args.model_checkpoint, CONFIG_NAME))
        model = model_class(config)
    return model, tokenizer

def interact():
    checkpoint = glob(os.path.join('.', "*ckpt"))[0]
    # model.load_state_dict(torch.load(checkpoint)['state_dict'])
    gpt2_model = GPT2Proto.load_from_checkpoint(checkpoint).eval().cuda()

    user_inputs = input(">> user:")
    speaker1, speaker2 = LSCCDataSet.get_identity_id(tokenizer)

    special_token_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, speaker2, speaker1]
    tokenizer.add_special_tokens({"additional_special_tokens": [tokenizer.convert_ids_to_tokens(_id)
                                                                for _id in [speaker1, speaker2]]})
    print(tokenizer.all_special_tokens)
    history = []
    while user_inputs != "bye":
        while not user_inputs:
            print("输入不能为空")
            user_inputs = input(">> user:")
        user_inputs = " ".join(list(user_inputs.replace(" ", "")))
        history.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(user_inputs)))
        output_id = generate(args, history, tokenizer, gpt2_model, speaker1, speaker2, special_token_ids)
        print(output_id)
        history.append(output_id)
        history = history[-(2 * args.max_history + 1):]
        print("Bot: ", tokenizer.decode(output_id, skip_special_tokens=True))
        user_inputs = input(">> user:")


class GPT2Proto(BaseTrainer):
    def __init__(self):
        super(GPT2Proto, self).__init__(args)
        self.model, self.tokenizer = model, tokenizer
        self.criterion = LabelSmoothing(self.tokenizer.vocab_size, padding_idx=self.tokenizer.pad_token_id,
                                        smoothing=args.smooth, ignore_idx=-100)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--gpt2', action='store_true', help="use gpt2")
        parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
        parser.add_argument("--model_checkpoint", type=str, default="config/cgpt/",
                            help="Path or URL of the module, if in interact mode, must set")
        parser.add_argument("--train_path", type=str, default="data/toy_train.txt",
                            help="Path of the train dataset for dist dataset. ")
        parser.add_argument("--valid_path", type=str, default="data/toy_valid.txt",
                            help="Path of the valid dataset for dist dataset. ")
        parser.add_argument("--dataset_cache", type=str, default="dataset_cache",
                            help="Path or url of the dataset cache")
        parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
        parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
        parser.add_argument("--max_history", type=int, default=15,
                            help="Number of previous exchanges to keep in history")
        parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear', 'cos'],
                            help="method of optim")
        parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
        parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                            help="Accumulate gradients on several steps")
        parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")

        parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
        parser.add_argument("--max_len", type=int, default=30, help="Sequence length to generate")
        parser.add_argument('--smooth', default=0.1, type=float)
        parser.add_argument("--top_k", type=int, default=30, help="top k sampling to generate")
        parser.add_argument("--top_p", type=float, default=0.7, help="top p sampling to generate")
        parser.add_argument("--temperature", type=float, default=0.8, help="temperature for softmax to generate")
        return parser

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, lm_labels = tuple(input_tensor for input_tensor in batch)
        if input_ids.size(1) > 512:
            logger.warning(f"Train out of size: input ids:{input_ids.size()} token:{token_type_ids.size()}")
            input_ids = input_ids[..., :512]
            token_type_ids = token_type_ids[..., :512]
            lm_labels = lm_labels[..., :512]
            # return {'loss': 0, 'log': {'loss': 0, 'lr': self.optimizer.param_groups[0]['lr'], 'ppl': 0}}
        lm_logits, *_ =self.model(input_ids, token_type_ids=token_type_ids)
        if batch_idx % args.print_freq == 0:
            logger.warning(
                f'Train: 预测句子：{self.tokenizer.decode(lm_logits.argmax(-1)[0])}\n\t原始：'
                f'{self.tokenizer.decode(lm_labels[0])}'
            )
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        loss = self.criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        ppl = torch.exp(F.cross_entropy(lm_logits_flat_shifted, lm_labels_flat_shifted, ignore_index=-100))
        tensorboard_logs = {
            'train_loss': loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'train_ppl': ppl
        }
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=True)
        return loss
        # return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx:int):
        input_ids, token_type_ids, lm_labels = tuple(input_tensor for input_tensor in batch)
        if input_ids.size(1) > 512:
            logger.warning(f"valid out of size: input ids:{input_ids.size()} token:{token_type_ids.size()}")
            input_ids = input_ids[..., :512]
            token_type_ids = token_type_ids[..., :512]
            lm_labels = lm_labels[..., :512]
        lm_logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        if batch_idx % args.print_freq == 0:
            logger.warning(f'Valid:预测句子：{self.tokenizer.decode(lm_logits.argmax(-1)[0])}\n\t原始：'
                           f'{self.tokenizer.decode(lm_labels[0])}')
        loss = self.criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        ppl = torch.exp(F.cross_entropy(lm_logits_flat_shifted, lm_labels_flat_shifted, ignore_index=-100))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ppl', ppl, prog_bar=True)
        # return {"val_loss": loss, 'val_ppl': ppl}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_ppl = torch.stack([x['val_ppl'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss.item(), 'val_ppl': avg_ppl.item()}
    #     return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def forward(self, input_ids, token_type_ids):
        if input_ids.size(1) > 512:
            logger.warning(f"out of size: input ids:{input_ids.size()} token:{token_type_ids.size()}")
            input_ids = input_ids[..., :512]
            token_type_ids = token_type_ids[..., :512]
        lm_logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
        return lm_logits

    def setup(self, stage: str):
        logger.warning(f'stage--->{stage}')
        self.train_data = LSCCDataSet(args.train_path, self.tokenizer, args.max_history, cache_dir=args.dataset_cache)
        self.val_data = LSCCDataSet(args.valid_path, self.tokenizer, args.max_history, cache_dir=args.dataset_cache)
        logger.warning(f'args---->\n{pformat(args)}')

    def configure_optimizers(self):
        optimizer = AdamW([{'params': self.model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)
        model_size = args.n_emd
        noam_lambda = lambda step: (
                model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
        scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)

        if args.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=len(
                                                            self.train_data)//args.batch_size*args.n_epochs)
        if args.scheduler == "cos":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=len(
                                                            self.train_data)//args.batch_size*args.n_epochs)
        self.optimizer = optimizer
        return [self.optimizer], [{'scheduler': scheduler,'interval':'step', 'frequency':1, 'name': 'lr_w' }]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run generation')
    parser.add_argument('--monitor', default='auto', type=str, help='callback monitor')
    parser.add_argument('--dirpath', default='.', type=str, help='dir path of checkpoint')
    parser.add_argument('--filename', default='checkpoint', type=str, help='filename of checkpoint')
    parser.add_argument('--period', default=1, type=int, help='number of epoches to save')
    parser.add_argument('--mode', default='min', type=str, help='mode of monitor, (min,max)')
    parser.add_argument('--print_freq', default=500, type=int, help='print frequency')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument("--patience", default=3, type=int, help="internal of val epochs to save")
    parser.add_argument('--interact', action="store_true", help='activate interact mode')
    parser = GPT2Proto.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
    tokenizer_class = BertTokenizer
    model, tokenizer = huggingface_initilize(model_class, config_class, tokenizer_class, do_lower_case=True,
                                             pretrained=args.pretrained,
                                             model_checkpoint=args.model_checkpoint,
                                             never_split=["[speaker1]", "[speaker2]"])
    if args.interact:
        interact()
    else:
        gpt2 = GPT2Proto(model, tokenizer)
        trainer = Trainer.from_argparse_args(args, checkpoint_callback=gpt2.checkpoint_callback,
                                             default_root_dir=args.logdir,
                                             callbacks=[gpt2.early_stop_callback, gpt2.lr_monitor],
                                             check_val_every_n_epoch = args.valid_steps,
                                             auto_select_gpus=True, #auto_lr_find=True,
                                             accumulate_grad_batches=args.gradient_accumulation_steps,
                                             gradient_clip_val=args.max_norm,
                                             max_epochs=args.n_epochs,
                                             accelerator='dp')
        # trainer.tune(gpt2)
        trainer.fit(gpt2)

