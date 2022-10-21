import collections
import os
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer, is_torch_tpu_available, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl, EarlyStoppingCallback
from transformers.deepspeed import deepspeed_init
from transformers.integrations import WandbCallback, INTEGRATION_TO_CALLBACK
from transformers.trainer import logger
from transformers.trainer_pt_utils import find_batch_size, nested_numpify, nested_concat, IterableDatasetShard, \
    nested_truncate
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, denumpify_detensorize, SchedulerType

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl


class WandbResumeCallback(WandbCallback):
    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also override the following environment
        variables:

        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training. Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to disable gradient logging or `"all"` to
                log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely. Set *WANDB_DISABLED=true* to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            if self._wandb.run is None:
                assert args.wandb_run_id, 'please supply wandb_run_id to resume'
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    resume="allow",
                    id=args.wandb_run_id,
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )


INTEGRATION_TO_CALLBACK['wandb_resume'] = WandbResumeCallback


@dataclass
class GECTrainingArguments(TrainingArguments):
    cold_steps: int = field(default=-1, metadata={"help": "If > 0: set number of training steps to perform with base model weights frozen."})
    cold_epochs: int = field(default=-1, metadata={"help": "If > 0: set number of training epochs to perform with base model weights frozen."})
    cold_lr: float = field(default=None, metadata={"help": "The learning rate for AdamW while base model weights are frozen."})

    early_stopping_patience: int = field(default=-1, metadata={"help": "If > 0: set patience for early stopping based on metric_for_best_model."})

    wandb_run_id: str = field(default=None, metadata={"help": "Wandb run id to resume (when using report_to = wandb_resume)."})

    def __post_init__(self):
        super().__post_init__()
        if self.cold_steps > 0 or self.cold_epochs > 0:
            assert self.cold_lr is not None, 'cold_lr argument required for use with cold_steps'
            assert self.lr_scheduler_type == SchedulerType.CONSTANT, 'only constant lr scheduler implemented'

        if self.early_stopping_patience > 0:
            assert self.metric_for_best_model is not None, 'metric_for_best_model argument required for use with early_stopping_patience'

        if 'wandb_resume' in self.report_to:
            assert self.wandb_run_id is not None, 'report_to=wandb_resume, but wandb_run_id is not set'

        if self.wandb_run_id is not None:
            assert 'wandb_resume' in self.report_to, 'wandb_run_id is set, but report_to=wandb_resume is not set'


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def set_optimizer_lr(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def set_scheduler_lr(scheduler, learning_rate):
    scheduler.base_lrs = [learning_rate] * len(scheduler.base_lrs)


class FreezeUnfreezeCallback(TrainerCallback):
    def on_step_begin(self, args: GECTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if args.cold_steps > 0:

            if state.global_step == 0:
                set_requires_grad(kwargs['model'].base_model, False)
                set_optimizer_lr(kwargs['optimizer'], args.cold_lr)
                set_scheduler_lr(kwargs['lr_scheduler'], args.cold_lr)
            elif state.global_step == args.cold_steps:
                set_requires_grad(kwargs['model'].base_model, True)
                set_optimizer_lr(kwargs['optimizer'], args.learning_rate)
                set_scheduler_lr(kwargs['lr_scheduler'], args.learning_rate)

    def on_epoch_begin(self, args: GECTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        if args.cold_epochs > 0:

            if state.epoch == 0:
                set_requires_grad(kwargs['model'].base_model, False)
                set_optimizer_lr(kwargs['optimizer'], args.cold_lr)
                set_scheduler_lr(kwargs['lr_scheduler'], args.cold_lr)
            elif state.epoch == args.cold_epochs:
                set_requires_grad(kwargs['model'].base_model, True)
                set_optimizer_lr(kwargs['optimizer'], args.learning_rate)
                set_scheduler_lr(kwargs['lr_scheduler'], args.learning_rate)


class GECTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if kwargs['args'].cold_steps > 0 or kwargs['args'].cold_epochs > 0:
            callbacks = kwargs.get('callbacks', [])
            callbacks.append(FreezeUnfreezeCallback())
            kwargs['callbacks'] = callbacks

        if kwargs['args'].early_stopping_patience > 0:
            callbacks = kwargs.get('callbacks', [])
            callbacks.append(EarlyStoppingCallback(kwargs['args'].early_stopping_patience))
            kwargs['callbacks'] = callbacks
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            labels_d = torch.ones_like(labels)
            if model.unk_tag_idx is not None:
                labels_d[labels == model.unk_tag_idx] = 2
            labels_d[labels == 0] = 0
            labels_d[labels == -100] = -100

            preds = torch.concat([l.argmax(2).detach() for l in logits], dim=1)
            labels = torch.concat([labels, labels_d], dim=1)  # should I detach here?

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if preds is not None:
                preds = self._pad_across_processes(preds)
                preds = self._nested_gather(preds)
                preds_host = preds if preds_host is None else nested_concat(preds_host, preds, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    preds = nested_numpify(preds_host)
                    all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            preds = nested_numpify(preds_host)
            all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
