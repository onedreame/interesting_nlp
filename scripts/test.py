from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import math
import os
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


def warmup_cosine(training_percentage, warmup=0.002):
    if training_percentage < warmup:
        return training_percentage / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * training_percentage))


def warmup_constant(training_percentage, warmup=0.002):
    if training_percentage < warmup:
        return training_percentage / warmup
    return 1.0


def warmup_linear(training_percentage, warmup=0.002):
    if training_percentage < warmup:
        return training_percentage / warmup
    return 1.0 - training_percentage
SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}
class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,      # pylint: disable=invalid-name
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        learning_rates = []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                learning_rates.append(lr_scheduled)
        return learning_rates

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the module
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(param.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(param.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(param, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * param.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                param.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

import torch.nn as nn
from transformers import CONFIG_NAME, BertModel
from trainer.base import _init_weights
from functools import partial
class BertForSequenceClassification(nn.Module):
    def __init__(self, num_classes, model_checkpoint, pretrained, **kwargs):
        super(BertForSequenceClassification, self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_checkpoint, CONFIG_NAME))
        if pretrained:
            # 不收敛的情况下检查学习率和初始化的模型是否有问题， dataset的shuffle也可能导致收敛效果不好
            print(f'加载预训练模型...')
            tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)
            bert_model = BertModel.from_pretrained(model_checkpoint)
        else:
            # 对于中文，使用修改后的分词器
            tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)
            bert_model = BertModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_classes)
        )
        initilizer = partial(_init_weights, config=config)
        self.classifier.apply(initilizer)
        self.model = bert_model
        self.tokenizer = tokenizer

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        lm_logits = self.classifier(outputs[1])
        return lm_logits

from datautils import THUCNewsDataset
from torch.utils.data import DataLoader
import  argparse

parser = argparse.ArgumentParser('Test')
parser.add_argument('--check',default=str)
parser.add_argument('--train_path', type=str)
parser.add_argument('--valid_path', type=str)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--label_path', type=str)
parser.add_argument('--scheduler', default='cos', type=str)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('-c', type=str)
args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained(args.check, cache_dir='./cache_down')
config = BertConfig.from_pretrained(args.check, cache_dir='./cache_down', num_labels=14)
model = BertForSequenceClassification(14, args.check, True)
criterion = nn.CrossEntropyLoss()
# module = BertForSequenceClassification.from_pretrained(args.check, cache_dir='./cache_down', config=config)
max_len=32
batch_size=32
epoch=40
trn_dataset = THUCNewsDataset(args.train_path, args.c, tokenizer, max_len)
val_dataset = THUCNewsDataset(args.valid_path, args.c, tokenizer, max_len)
tl = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, num_workers=8,collate_fn=trn_dataset.collate)
vl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8,collate_fn=trn_dataset.collate)

no_decay = ["bias", "gamma", "beta"]
optimizer_parameters = [
    {"params" : [p for name, p in model.named_parameters() \
        if name not in no_decay], "weight_decay_rate" : 0.01},
    {"params" : [p for name, p in model.named_parameters() \
        if name in no_decay], "weight_decay_rate" : 0.0}
]
num_train_steps = int(len(trn_dataset) / batch_size / \
    args.gradient_accumulation_steps * epoch)
# optimizer = BERTAdam(optimizer_parameters,
#                      lr=5e-5,
#                      warmup=0.1,
#                      t_total=num_train_steps)
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
optimizer = AdamW([{'params': model.parameters(), 'initial_lr': 5e-5}], lr=5e-5, correct_bias=True)
if args.scheduler == "noam":
    model_size = 768
    noam_lambda = lambda step: (
            model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
    scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)

if args.scheduler == "linear":
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300,
                                                num_training_steps=len(
                                                    trn_dataset)//batch_size*epoch)
if args.scheduler == "cos":
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=300,
                                                num_training_steps=len(
                                                    trn_dataset)//batch_size*epoch)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = model.cuda()
for epoch in range(epoch):
    '''获取批数据'''
    model.train()
    total_loss = 0.0
    correct_sum = 0

    for step, batch in enumerate(tl):
        input_ids, segment_ids, input_mask, lm_labels = tuple(input_tensor.cuda() for input_tensor in batch)

        '''输入BERT前一定要注意格式'''
        input_ids = input_ids.view(-1, input_ids.size(-1))  # size(-1)表示最后一维的size
        input_mask = input_mask.view(-1, input_mask.size(-1))  # size(-1)表示最后一维的size
        segment_ids = segment_ids.view(-1, segment_ids.size(-1))  # size(-1)表示最后一维的size

        '''BERT分类模型的输入可以参考下huggingface'''
        # loss, logit = module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
        #                 labels=lm_labels)
        logit = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        loss = criterion(logit, lm_labels)
        '''计算loss'''
        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            model.zero_grad()
        loss_val = loss.item()
        _, top_index = logit.topk(1)
        correct_sum = (top_index.view(-1) == lm_labels).sum().item()
        total_loss += loss_val
        if step%1000==0:
            print(f"{step}->loss:{loss_val} accuray:{correct_sum/lm_labels.size(0)}")
    model.eval()
    ac = 0
    acc = 0
    for step, batch in enumerate(vl):
        input_ids, segment_ids, input_mask, lm_labels = tuple(input_tensor.cuda() for input_tensor in batch)

        '''输入BERT前一定要注意格式'''
        input_ids = input_ids.view(-1, input_ids.size(-1))  # size(-1)表示最后一维的size
        input_mask = input_mask.view(-1, input_mask.size(-1))  # size(-1)表示最后一维的size
        segment_ids = segment_ids.view(-1, segment_ids.size(-1))  # size(-1)表示最后一维的size

        '''BERT分类模型的输入可以参考下huggingface'''
        # loss, logit = module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
        #                 labels=lm_labels)
        logit = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        loss = criterion(logit, lm_labels)
        _, top_index = logit.topk(1)
        ac += (top_index.view(-1) == lm_labels).sum().item()
        acc+= lm_labels.size(0)
    print(f'eval acc:{ac/acc}')