# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import sys
import math
from functools import partial
import argparse
import logging
from pprint import pformat

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import CONFIG_NAME, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from pytorch_lightning import Trainer
from pytorch_lightning.metrics import Accuracy

sys.path.append(os.getcwd())
from datautils import THUCNewsDataset
from module import LabelSmoothing, FullTokenizer
from trainer import BaseTrainer, BertForSequenceClassification


logger = logging.getLogger('lightning')
logger.setLevel(logging.INFO)
logger.propagate = False

class GPT2Proto(BaseTrainer):
    def __init__(self, args: argparse.Namespace):
        super(GPT2Proto, self).__init__(args)
        # 不收敛的情况下检查学习率和初始化的模型是否有问题， dataset的shuffle也可能导致收敛效果不好
        self.model = BertForSequenceClassification(args.num_classes, args.model_checkpoint, args.pretrained, logger)
        self.args = args
        self.lr = 1e-5  # for auto lr find
        # self.classifier = torch.nn.Linear(args.n_emd, args.num_classes)
        # self._init_weights(self.classifier)
        # self.accuracy = Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = LabelSmoothing(args.num_classes, padding_idx=-1,
        #                                 smoothing=args.smooth, ignore_idx=-1)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--arch', type=str, default='BertModel', help="module architecture")
        parser.add_argument('--model_config', type=str, default='', help='module config filename')
        parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument("--model_checkpoint", type=str, default="RoBERTa-wwm-ext/", help="Path or URL of the module")
        parser.add_argument("--train_path", type=str, default="data/toy_train.txt",
                            help="Path of the train dataset for dist dataset. ")
        parser.add_argument("--valid_path", type=str, default="data/toy_valid.txt",
                            help="Path of the valid dataset for dist dataset. ")
        parser.add_argument("--dataset_cache", type=str, default="dataset_cache",
                            help="Path or url of the dataset cache")
        parser.add_argument('--label_path', default='', type=str)
        parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
        parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
        parser.add_argument("--max_history", type=int, default=512,
                            help="Number of previous exchanges to keep in history")
        parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear', 'cos'],
                            help="method of optim")
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--eval_before_start", action='store_true',
                            help="If true start with a first evaluation before training")
        parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
        parser.add_argument("--valid_steps", type=int, default=10, help="Perfom validation every X steps")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                            help="Accumulate gradients on several steps")
        parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
        parser.add_argument('--smooth', default=0.1, type=float)
        return parser

    def accuracy(self, probs, labels):
        logits = probs.argmax(dim=-1)
        assert logits.size()==labels.size(), f'logits shape != labels shape[{logits.size()}!={labels.size()}]'
        return logits.eq(labels).sum()/ labels.size(0)

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attn_mask, lm_labels = tuple(input_tensor for input_tensor in batch)
        input_ids = input_ids.view(-1, input_ids.size(-1))  # size(-1)表示最后一维的size
        attn_mask = attn_mask.view(-1, attn_mask.size(-1))  # size(-1)表示最后一维的size
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # size(-1)表示最后一维的size
        if input_ids.size(1) > 512:
            logger.warning(f"Train out of size: input ids:{input_ids.size()} token:{token_type_ids.size()}")
            input_ids = input_ids[..., :512]
            token_type_ids = token_type_ids[..., :512]
            attn_mask = attn_mask[..., :512]
        lm_logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attn_mask)
        loss = self.criterion(lm_logits, lm_labels)

        tensorboard_logs = {
            'lr': self.optimizer.param_groups[0]['lr'],
            'trn_prec': self.accuracy(lm_logits, lm_labels)
        }
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=True)
        return loss
        # return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx:int):
        input_ids, token_type_ids, attn_mask, lm_labels = tuple(input_tensor for input_tensor in batch)
        input_ids = input_ids.view(-1, input_ids.size(-1))  # size(-1)表示最后一维的size
        attn_mask = attn_mask.view(-1, attn_mask.size(-1))  # size(-1)表示最后一维的size
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # size(-1)表示最后一维的size
        if input_ids.size(1) > 512:
            logger.warning(f"valid out of size: input ids:{input_ids.size()} token:{token_type_ids.size()}")
            input_ids = input_ids[..., :512]
            token_type_ids = token_type_ids[..., :512]
            attn_mask = attn_mask[..., :512]
        lm_logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attn_mask)
        loss = self.criterion(lm_logits, lm_labels)
        # self.log('val_loss', loss, prog_bar=True)
        # self.log('val_prec', self.accuracy(lm_logits, lm_labels), prog_bar=True)
        correct_sum = lm_logits.argmax(dim=-1).eq(lm_labels).sum()
        example_sum = lm_labels.size(0)
        return {"val_loss": loss, 'correct_sum': correct_sum, 'example_sum': example_sum}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct_sum = torch.stack([x['correct_sum'] for x in outputs]).sum().item()
        example_sum = sum([sum(x['example_sum']) for x in outputs])
        tensorboard_logs = {'val_loss': avg_loss.item(), 'val_prec': correct_sum/example_sum}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def forward(self, input_ids, token_type_ids, attn_mask):
        input_ids = input_ids.view(-1, input_ids.size(-1))  # size(-1)表示最后一维的size
        attn_mask = attn_mask.view(-1, attn_mask.size(-1))  # size(-1)表示最后一维的size
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # size(-1)表示最后一维的size
        if input_ids.size(1) > 512:
            logger.warning(f"out of size: input ids:{input_ids.size()} token:{token_type_ids.size()}")
            input_ids = input_ids[..., :512]
            token_type_ids = token_type_ids[..., :512]
            attn_mask = attn_mask[..., :512]
        lm_logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attn_mask)
        return lm_logits

    def setup(self, stage: str):
        logger.warning(f'stage--->{stage}')
        self.train_data = THUCNewsDataset(self.args.train_path, self.args.label_path, self.model.tokenizer,
                                          self.args.max_history)
        self.val_data = THUCNewsDataset(self.args.valid_path, self.args.label_path, self.model.tokenizer,
                                        self.args.max_history)
        logger.warning(f'args---->\n{pformat(self.args)}')

    def configure_optimizers(self):
        param_optimizer = self.model.named_parameters() # 模型参数名字列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'initial_lr': args.lr
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'initial_lr': args.lr
            }
        ]
        # [{
        #     'params': optimizer_grouped_parameters, 'initial_lr': args.lr
        # }]
        optimizer = AdamW([{'params': self.model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=True)
        if args.scheduler == "noam":
            model_size = args.n_emd
            noam_lambda = lambda step: (
                    model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
            scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)

        if args.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                        num_training_steps=len(
                                                            self.train_data)//self.args.batch_size*self.args.n_epochs)
        if args.scheduler == "cos":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                        num_training_steps=len(
                                                            self.train_data)//self.args.batch_size*self.args.n_epochs)
        self.optimizer = optimizer
        return [self.optimizer], [{'scheduler': scheduler,'interval':'step', 'frequency':1, 'name': 'lr_w' }]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run generation')
    parser.add_argument('--logdir', default='log', type=str, help='dir to log')
    parser.add_argument('--monitor', default='auto', type=str, help='callback monitor')
    parser.add_argument('--dirpath', default='.', type=str, help='dir path of checkpoint')
    parser.add_argument('--filename', default='checkpoint', type=str, help='filename of checkpoint')
    parser.add_argument('--period', default=1, type=int, help='number of epoches to save')
    parser.add_argument('--mode', default='min', type=str, help='mode of monitor, (min,max)')
    parser.add_argument('--print_freq', default=500, type=int, help='print frequency')
    parser.add_argument('--logger_name', default='logger', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser = GPT2Proto.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    gpt2 = GPT2Proto(args)
    trainer = Trainer.from_argparse_args(args, checkpoint_callback=gpt2.checkpoint_callback,
                                         callbacks=[gpt2.early_stop_callback, gpt2.lr_monitor],
                                         check_val_every_n_epoch = args.valid_steps,
                                         auto_select_gpus=True, #auto_lr_find=True,
                                         accumulate_grad_batches=args.gradient_accumulation_steps,
                                         gradient_clip_val=args.max_norm,
                                         max_epochs=args.n_epochs,
                                         # overfit_batches = 20,
                                         # fast_dev_run=True,
                                         accelerator='dp')
    # trainer.tune(gpt2)
    trainer.fit(gpt2)

