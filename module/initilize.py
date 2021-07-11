import os
import torch.nn as nn
from transformers import CONFIG_NAME

__all__ = ['init_weights', 'huggingface_initilize']


def init_weights(module, config=None):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def huggingface_initilize(model_class, config_class, tokenizer_class, do_lower_case=True, pretrained=False,
                          model_checkpoint=None, never_split=None):
    if pretrained:
        # 不收敛的情况下检查学习率和初始化的模型是否有问题， dataset的shuffle也可能导致收敛效果不好
        tokenizer = tokenizer_class.from_pretrained(model_checkpoint, do_lower_case=do_lower_case,
                                                    never_split=never_split)
        model = model_class.from_pretrained(model_checkpoint)
    else:
        tokenizer = tokenizer_class(os.path.join(model_checkpoint, "vocab.txt"), do_lower_case=do_lower_case,
                                    never_split=never_split)
        config = config_class.from_json_file(os.path.join(model_checkpoint, CONFIG_NAME))
        model = model_class(config)
    return model, tokenizer
