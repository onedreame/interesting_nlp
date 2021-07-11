# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
from pprint import pformat

import torch
from torch.nn.parallel import DistributedDataParallel

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (WEIGHTS_NAME, CONFIG_NAME)

__all__ = ['train']


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    print('scalar_t'.center(30,'-'))
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    print('all reduce'.center(30, '='))
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    print('result'.center(30,'#'))
    return scalar_t.item()


def train(args, logger, model, tokenizer, train_loader, val_loader, train_sampler, val_sampler,
          optimizer, scheduler=None, criterion=torch.nn.CrossEntropyLoss(ignore_index=-1)):
    distributed = args.local_rank != -1
    model.to(args.device)
    # Prepare module for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Training function and trainer
    def update(engine, batch):
        input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
        model.train()
        if input_ids.size(1) > 512:
            logger.warning(f'input ids:{input_ids.size()} token:{token_type_ids.size()}')
            input_ids = input_ids[..., :512]
            token_type_ids = token_type_ids[..., :512]
            lm_labels = lm_labels[..., :512]
        lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        lm_loss = criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            logger.info(f'train:预测句子：{tokenizer.decode(lm_logits.argmax(-1)[0])}\n\t原始：'
                  f'{tokenizer.decode(lm_labels[0])}')
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(), optimizer.param_groups[0]['lr']

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        logger.info('infering'.center(60,'-'))
        model.eval()
        with torch.no_grad():
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            if input_ids.size(1) > 512:
                logger.warning(f'inference input ids:{input_ids.size()} token:{token_type_ids.size()}')
                input_ids = input_ids[..., :512]
                token_type_ids = token_type_ids[..., :512]
                lm_labels = lm_labels[..., :512]
            lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            logger.info(f'inference:预测句子：{tokenizer.decode(lm_logits.argmax(-1)[0])}\n\t原始：'
                        f'{tokenizer.decode(lm_labels[0])}')
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Evaluation during training
    @trainer.on(Events.ITERATION_STARTED)
    def log_iterations(engine):
        # if engine.state.iteration % max(int(0.1 * len(train_loader)), 1) == 0:
        if engine.state.iteration % args.valid_steps == 0:
            evaluator.run(val_loader)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: val_sampler.set_epoch(engine.state.epoch))

    if scheduler is not None:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints
    # And save module, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True, mininterval=2)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.EPOCH_COMPLETED)

        def score_function(engine):
            return engine.state.metrics['average_ppl']

        to_save = {'module': model, 'optimizer': optimizer, 'scheduler': scheduler}
        checkpoint_handler = Checkpoint(
            to_save, n_saved=3, filename_prefix='best_ppl', score_function=score_function,
            score_name='average_ppl', global_step_transform=global_step_from_engine(trainer),
            save_handler=DiskSaver(os.path.join(tb_logger.writer.logdir, 'checkpoint'), create_dir=True)
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
        # checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
        # # save module after evaluation
        # evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
        #     'mymodel': getattr(module, 'module', module)})
        # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
        #     'mymodel': getattr(module, 'module', module)})  # "getattr" take care of distributed encapsulation
        # print(f'tb logger: {tb_logger} \n args:{args}')
        # torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.logdir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint
    # (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1],
                  os.path.join(tb_logger.writer.logdir,
                               WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    import pandas as pd
    fp = 'videoplay_predict.out'
    data = pd.read_csv(fp, sep='\t', header=None)
    columns = ['imp_id', 'predict', 'litectr_label', 'pctr']
    data.columns = columns
    print(data.head())

