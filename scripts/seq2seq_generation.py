# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import sys
import argparse
import logging
from pprint import pformat

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from pytorch_lightning import Trainer

sys.path.append(os.getcwd())
from datautils import TextFileDataset
from module import LabelSmoothing, BasicTokenizer
from model import Seq2Seq
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
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

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


class Seq2SeqProto(BaseTrainer):
    def __init__(self, conf):
        super(Seq2SeqProto, self).__init__(conf)
        self.tokenizer = BasicTokenizer(self.conf.vocab_file, need_tokenize=False)
        self.train_data = TextFileDataset(self.conf.train_path, self.tokenizer, self.conf.max_len,
                                          cache_dir=self.conf.dataset_cache)
        self.val_data = TextFileDataset(self.conf.valid_path, self.tokenizer, self.conf.max_len,
                                        cache_dir=self.conf.dataset_cache)
        self.model = Seq2Seq(self.tokenizer.vocab_size, self.conf.hidden_size, self.conf.n_layers,
                             self.conf.dropout, self.conf.max_len, self.conf.attn, share_emb=True,
                             sos=self.tokenizer.sos_token_id, eos=self.tokenizer.eos_token_id)
        # self.criterion = LabelSmoothing(self.tokenizer.vocab_size, padding_idx=self.tokenizer.pad_token_id,
        #                                 smoothing=self.conf.smooth, ignore_idx=self.tokenizer.unk_token_id)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_checkpoint", type=str, default="config/cgpt/",
                            help="Path or URL of the module, if in interact mode, must set")
        parser.add_argument('--vocab_file', type=str,
                            default='/root/PycharmProjects/interesting_nlp/datasets/doupo/label.tsv')
        parser.add_argument("--train_path", type=str, default="data/toy_train.txt",
                            help="Path of the train dataset for dist dataset. ")
        parser.add_argument("--valid_path", type=str, default="data/toy_valid.txt",
                            help="Path of the valid dataset for dist dataset. ")
        parser.add_argument("--dataset_cache", type=str, default="dataset_cache",
                            help="Path or url of the dataset cache")
        parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
        parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
        parser.add_argument("--scheduler", type=str, default="cos", choices=['noam', 'linear', 'cos'],
                            help="method of optim")
        parser.add_argument("--hidden_size", type=int, default=64, help="hidden size")
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--attn", type=str, default="concat")
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
        parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                            help="Accumulate gradients on several steps")
        parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
        parser.add_argument("--n_layers", type=int, default=2, help="layers of RNN")
        parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
        parser.add_argument("--max_len", type=int, default=30, help="Sequence length to generate")
        parser.add_argument('--smooth', default=0.1, type=float)
        parser.add_argument("--top_k", type=int, default=30, help="top k sampling to generate")
        parser.add_argument("--top_p", type=float, default=0.7, help="top p sampling to generate")
        parser.add_argument("--temperature", type=float, default=0.8, help="temperature for softmax to generate")
        return parser

    def training_step(self, batch, batch_idx):
        input_ids, input_lens = tuple(input_tensor for input_tensor in batch)
        lm_logits = self.model(input_ids, input_ids, input_lens)
        if batch_idx % args.print_freq == 0:
            logger.warning(
                f'Train: 预测句子：{self.tokenizer.decode(lm_logits.argmax(-1)[0].tolist())}\n\t原始：'
                f'{self.tokenizer.decode(input_ids[0].tolist())}'
            )
        lm_logits_flat_shifted = lm_logits.view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = input_ids.view(-1)
        loss = self.criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        ppl = torch.exp(F.cross_entropy(lm_logits_flat_shifted, lm_labels_flat_shifted, ignore_index=-100))
        tensorboard_logs = {
            'train_loss': loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'train_ppl': ppl
        }
        # self.log_dict(tensorboard_logs, prog_bar=True, on_step=True)
        return {'loss': loss, "lr": self.optimizer.param_groups[0]['lr']}
        # return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx: int):
        input_ids, input_lens = tuple(input_tensor for input_tensor in batch)
        lm_logits = self.model(input_ids, input_ids, src_lengths=input_lens)
        lm_logits_flat_shifted = lm_logits.view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = input_ids.view(-1)
        if batch_idx % args.print_freq == 0:
            logger.warning(f'Valid:预测句子：{self.tokenizer.decode(lm_logits.argmax(-1)[0].tolist())}\n\t原始：'
                           f'{self.tokenizer.decode(input_ids[0].tolist())}')
        loss = self.criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        ppl = torch.exp(F.cross_entropy(lm_logits_flat_shifted, lm_labels_flat_shifted, ignore_index=-100))
        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_ppl', ppl, prog_bar=True)
        # return {"val_loss": loss, 'val_ppl': ppl}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_ppl = torch.stack([x['val_ppl'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss.item(), 'val_ppl': avg_ppl.item()}
    #     return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def forward(self, input_ids, token_type_ids):
        lm_logits = self.model(input_ids)
        return lm_logits

    def setup(self, stage: str):
        logger.warning(f'stage--->{stage}')
        logger.warning(f'args---->\n{pformat(args)}')

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(),lr=args.lr, momentum=0.98, weight_decay=1e-4)
        # AdamWeightDecayOptimizer：https://blog.csdn.net/yinyu19950811/article/details/90476956#16_AdamAdaptiVe_Moment_Estimation_82
        optimizer = AdamW([{'params': self.model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)
        model_size = args.hidden_size
        noam_lambda = lambda step: (
                model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
        scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)

        if args.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=len(
                                                            self.train_data) // args.batch_size * args.n_epochs)
        if args.scheduler == "cos":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=len(
                                                            self.train_data) // args.batch_size * args.n_epochs)
        self.optimizer = optimizer
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1, 'name': 'lr_w'}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run generation')
    parser.add_argument('--monitor', default='auto', type=str, help='callback monitor')
    parser.add_argument("--logdir", default="log", type=str)
    parser.add_argument('--dirpath', default='.', type=str, help='dir path of checkpoint')
    parser.add_argument('--filename', default='checkpoint', type=str, help='filename of checkpoint')
    parser.add_argument('--period', default=1, type=int, help='number of epoches to save')
    parser.add_argument('--mode', default='min', type=str, help='mode of monitor, (min,max)')
    parser.add_argument('--print_freq', default=500, type=int, help='print frequency')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument("--patience", default=3, type=int, help="internal of val epochs to save")
    parser.add_argument('--interact', action="store_true", help='activate interact mode')
    parser = Seq2SeqProto.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.interact:
        # interact()
        pass
    else:
        seq2seq = Seq2SeqProto(args)
        trainer = Trainer.from_argparse_args(args, checkpoint_callback=seq2seq.checkpoint_callback,
                                             default_root_dir=args.logdir,
                                             callbacks=[seq2seq.early_stop_callback, seq2seq.lr_monitor],
                                             check_val_every_n_epoch=args.valid_steps,
                                             auto_select_gpus=True,  # auto_lr_find=True,
                                             accumulate_grad_batches=args.gradient_accumulation_steps,
                                             gradient_clip_val=args.max_norm,
                                             max_epochs=args.n_epochs,
                                             accelerator='dp',
                                             gpus=1)
        # trainer.tune(gpt2)
        trainer.fit(seq2seq)
