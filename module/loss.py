import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['masked_cross_entropy', 'MultiGPULossCompute', 'LabelSmoothing']


def sequence_mask(sequence_length, max_len=None, device=torch.device("cuda:0")):
    """
    Args:
        sequence_length (Variable, LongTensor) [batch_size]
            - list of sequence length of each batch
        max_len (int)
    Return:
        masks (bool): [batch_size, max_len]
            - True if current sequence is valid (not padded), False otherwise

    Ex.
    sequence length: [3, 2, 1]

    seq_length_expand
    [[3, 3, 3],
     [2, 2, 2]
     [1, 1, 1]]

    seq_range_expand
    [[0, 1, 2]
     [0, 1, 2],
     [0, 1, 2]]

    masks
    [[True, True, True],
     [True, True, False],
     [True, False, False]]
    """
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)

    # [max_len]
    seq_range = torch.arange(0, max_len).long().to(device)  # [0, 1, ... max_len-1]

    # [batch_size, max_len]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)

    # [batch_size, max_len]
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)

    # [batch_size, max_len]
    masks = seq_range_expand < seq_length_expand

    return masks


# https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def masked_cross_entropy(logits, target, length, per_example=False, device=torch.device("cuda:0")):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood:
    #   -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len, device=device)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    # loss = losses.sum() / length.float().sum()

    if per_example:
        # loss: [batch_size]
        return losses.sum(1)

    loss = losses.sum()
    return loss, length.float().sum()


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0,
                 ignore_idx=None, **kwargs):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.ignore_idx = ignore_idx
        self.size = size

    def forward(self, x, target):
        '''
        计算smooth loss
        :param x: 预测概率分布，(batch_size*max_len, tgt_vocab)
        :param target: groud truth (batch_size*max_len, )
        :return: loss
        '''
        assert x.size(1) == self.size, f"x.size(-1)[{x.size()}] != {self.size}"
        x = F.log_softmax(x, dim=-1)
        # 过滤掉padding idx的logits
        if self.ignore_idx is not None:
            x = x[target.ne(self.ignore_idx)]
            target = target[target.ne(self.ignore_idx)]
        x = x[target.ne(self.padding_idx)]
        target = target[target.ne(self.padding_idx)]
        true_dist = x.clone().type_as(x)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return self.criterion(x, true_dist)


class MultiGPULossCompute(object):
    '''多个gpu计算loss'''

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        self.generator = generator
        self.devices = devices
        self.criterion = nn.parallel.replicate(criterion, devices)
        self.r = criterion
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, target, normalize, is_training=True, cur_epoch=None):
        '''
        划分多GPU，计算loss
        :param out: (batch_size, max_len, embedding_size)
        :param target: (batch_size, max_len)
        :param normalize:
        :return: 汇总loss
        '''
        total_loss = 0
        generator = nn.parallel.replicate(self.generator, self.devices)
        out_scatter = nn.parallel.scatter(out, self.devices)

        out_grad = [[] for _ in out_scatter]
        target = nn.parallel.scatter(target, self.devices)

        # divide generating into chunks
        for i in range(0, out_scatter[0].size(1), self.chunk_size):
            out_column = [[torch.tensor(o[:, i:i + self.chunk_size], requires_grad=is_training)]
                          for o in out_scatter]

            gen = nn.parallel.parallel_apply(generator, out_column, devices=self.devices)

            y = [[g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + self.chunk_size].contiguous().view(-1)]
                 for g, t in zip(gen, target)]
            loss = nn.parallel.parallel_apply(self.criterion, y, devices=self.devices)

            # sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum() / normalize
            total_loss += l.item()

            # backprop loss to output of transformer
            if self.opt is not None and is_training:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())
                    if torch.isnan(out_grad[j][-1]).any():
                        exit(0)

        # backprop all loss through transformer
        if self.opt is not None and is_training:
            out_grad = [torch.cat(og, dim=1) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step(cur_epoch)
            self.opt.optimizer.zero_grad()
        return total_loss * normalize
