import os
from functools import partial
import torch.nn as nn
from typing import Union, List, Any
from torch.utils.data import DataLoader

from transformers import BertModel, BertConfig, BertTokenizer, CONFIG_NAME
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor

from utils import setup_seed

__all__ = [ 'BaseTrainer', "BertForSequenceClassification"]


class BaseTrainer(LightningModule):
    def __init__(self, dirpath=None, filename=None, period=3, seed=1234,
                 monitor=None, mode="min", patience=3, num_workers=8):
        super(BaseTrainer, self).__init__()
        self.checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename=filename, save_top_k=1,
                                                   period=period,  monitor=monitor, mode=mode)
        # early stop 的patience统计的是val的epochs数目，不是train的epochs数目，这里注意
        self.early_stop_callback = EarlyStopping(monitor=monitor, mode=mode,patience=patience)
        self.lr_monitor = LearningRateMonitor('step')
        self.seed = seed
        self.num_workers = num_workers

    def on_fit_start(self):
        setup_seed(self.seed)
        super().on_fit_start()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.args.batch_size, num_workers=self.num_workers,
                          collate_fn=self.train_data.collate)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data,  batch_size=self.args.batch_size, num_workers=self.num_workers,
                          collate_fn=self.val_data.collate)


class BertForSequenceClassification(nn.Module):
    def __init__(self, num_classes, model_checkpoint, pretrained, logger, **kwargs):
        super(BertForSequenceClassification, self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_checkpoint, CONFIG_NAME))
        if pretrained:
            # 不收敛的情况下检查学习率和初始化的模型是否有问题， dataset的shuffle也可能导致收敛效果不好
            logger.warning(f'加载预训练模型...')
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