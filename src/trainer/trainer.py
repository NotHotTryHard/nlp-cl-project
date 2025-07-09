import random
from collections import defaultdict
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import librosa
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from src.datasets import MixedSequentialDataset
from src.model import T5SVDLoRA, T5LoRASequential
from src.utils.util import check_cuda_memory, clear_cuda_cache

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            inference_on_evaluation=False,
            inference_indices=None,
            first_epoch_eval_only=True,
            eval_adapter_order=None
    ):
        super().__init__(
            model, criterion, metrics,
            optimizer, lr_scheduler, config, device,
            first_epoch_eval_only
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]

        self.train_dataset = self.train_dataloader.dataset
        
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "extra_loss", 'total_loss', "grad norm",
            *[m.name for m in self.metrics if self._compute_on_train(m)],
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "extra_loss", "total_loss", *[m.name for m in self.metrics], writer=self.writer
        )
        self.perform_generative_eval = len(self.metrics) > 0 

        self.eval_adapter_order = eval_adapter_order

        self.log_singular_values = self.config["trainer"].get("log_singular_values", False) # only works for SVDLoRA
        self.log_sv_module_names = self.config["trainer"].get("log_sv_module_names", None)

        # now in base_trainer.py
        # self.first_epoch_eval_only = first_epoch_eval_only

        self.inference_on_evaluation = inference_on_evaluation
        self.inference_indices = inference_indices

        if self.inference_on_evaluation:
            self.inference_batch = defaultdict(dict)
            self.inference_texts = defaultdict(list)

            for split, val_dataloader in self.evaluation_dataloaders.items():
                if split in self.inference_indices:
                    inference_samples = [val_dataloader.dataset[ind] for ind in self.inference_indices[split]]
                    self.inference_batch[split] = val_dataloader.collate_fn(inference_samples)
                    self.inference_texts[split] = [sample[1] for sample in inference_samples]

    @staticmethod
    def _compute_on_train(metric):
        if hasattr(metric, "compute_on_train"):
            return metric.compute_on_train
        return True

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        if "decoder_input_ids" in batch:
            batch["decoder_input_ids"] = batch["decoder_input_ids"].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        clear_cuda_cache()
        check_cuda_memory()

        changed_dataset = False
        if isinstance(self.train_dataset, MixedSequentialDataset):
            changed_dataset = self.train_dataset.update_epoch(
                epoch=epoch,
                epochs=self.epochs,
                model=self.model,
                dataloader=self.train_dataloader,
                max_samples=250
            )

        if hasattr(self.model, "update_adapters"):
            self.model.update_adapters(self.train_dataset.current_dataset)

            if changed_dataset:
                params = filter(lambda p: p.requires_grad, self.model.parameters())
                self.optimizer = self.config.init_obj(self.config["optimizer"], torch.optim, params)
                self.lr_scheduler = self.config.init_obj(self.config["lr_scheduler"], torch.optim.lr_scheduler, self.optimizer)
                print("Reinitialized optimizer and lr scheduler for new dataset!")

        # move into update_adapters ?
        if self.config["model"].get("reinit_adapters", False) and hasattr(self.model, "reinit_adapters"):
            self.model.reinit_adapters()
            
        if self.first_epoch_eval_only and epoch == 0:
            log = self.train_metrics.result()
            for part, dataloader in self.evaluation_dataloaders.items():
                val_log = self._evaluation_epoch(epoch, part, dataloader)
                log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
            return log

        if hasattr(self.model, "enable_extra_loss"):
            self.model.enable_extra_loss()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch=batch,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    is_train=True,
                    metrics_tracker=self.train_metrics
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
            
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                # self.writer.set_step(((epoch - 1) * self.len_epoch + batch_idx) * self.train_dataloader.batch_size)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx),
                        batch["loss"].item()
                    )
                )

                last_lr = {}
                if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    last_lr = self.optimizer.param_groups[0]['lr']
                else:
                    last_lr = self.lr_scheduler.get_last_lr()[0]

                self.writer.add_scalar("learning rate", last_lr)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx >= self.len_epoch:
                break

        if self.lr_scheduler is not None:
            if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step()
    
        log = last_train_metrics
        if epoch % self.config["trainer"].get("eval_frequency", 1) == 0 or changed_dataset:
            for part, dataloader in self.evaluation_dataloaders.items():
                val_log = self._evaluation_epoch(epoch, part, dataloader)
                log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
            
            if self.log_singular_values:
                sv_dict = self.model.collect_singular_values(self.log_sv_module_names)
                for module_name, values in sv_dict.items():
                    pass
                    # log.update(**{module_name: values for name})

        return log

    
    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        if hasattr(self.model, "disable_extra_loss"):
            self.model.disable_extra_loss()

        self.evaluation_metrics.reset()

        if isinstance(self.model, T5LoRASequential):
            adapter_idx = self.eval_adapter_order[part]
            self.model.update_adapters(adapter_idx)

        clear_cuda_cache()
        check_cuda_memory()

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(
                    batch=batch,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    is_train=False,
                    metrics_tracker=self.evaluation_metrics
                )
            
            self.writer.set_step(epoch * self.len_epoch, part)
            # self.writer.set_step((epoch * self.len_epoch) * self.train_dataloader.batch_size, part)
            self._log_scalars(self.evaluation_metrics)
        
            if self.inference_on_evaluation and (part in self.inference_indices):
                inference_predicts = self.model(self.inference_batch[part])
                for ind, text, predict in zip (self.inference_indices[part], self.inference_texts, inference_predicts):
                    self._log_inference_as_table(
                        epoch, text, predict,
                        name=f"sample_{ind}"
                    )

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p.float(), bins="auto")
        
        return self.evaluation_metrics.result()


    def process_batch(self, batch, epoch=0, batch_idx=0, is_train=False, metrics_tracker: MetricTracker = None):
        # zero grad if made a step on previous batch
        if self.grad_accum_steps == 1 or (batch_idx) % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        batch = self.move_batch_to_device(batch, self.device)
        batch["loss"] = self.model(batch)
        metrics_tracker.update("loss", batch["loss"].item())

        if hasattr(self.model, 'collect_extra_loss') and is_train:
            extra_loss = self.model.collect_extra_loss()
            batch["loss"] = batch["loss"] + extra_loss

            metrics_tracker.update("extra_loss", extra_loss.item())
            metrics_tracker.update("total_loss", batch["loss"].item())

        if is_train:
            batch["loss"].backward()
            if self.grad_accum_steps == 1 or (batch_idx + 1) % self.grad_accum_steps == 0:
                self._clip_grad_norm()
                self.optimizer.step()
        
        elif self.perform_generative_eval:
            inputs, targets, preds = self.model._generative_step(batch)
            batch["inputs"], batch["target"], batch["preds"] = inputs, targets, preds
            if batch_idx == 0:
                full_text_inputs = (batch["input_ids"] == batch["labels"]).all()

                for i, (input, target, pred) in enumerate(zip(inputs, targets, preds)):
                    text = input
                    if not full_text_inputs:
                        text = input + target

                    self._log_inference_as_table(
                        epoch, text, pred,
                        name=f"sample_{i}"
                    )
        
        for met in self.metrics:
            if (not is_train) or self._compute_on_train(met):
                res = met(self.model, batch)
                if isinstance(res, dict):
                    for key, value in res.items():
                        metrics_tracker.update(f"{met.name}_{key}", value)
                else:
                    metrics_tracker.update(met.name, res)
        return batch


    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch, name="spectrogram"):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(name, ToTensor()(image))

    def _log_audio(self, audio, name="audio"):
        sample_rate = self.config["preprocessing"]["mel_spec_config"]["sr"]
        self.writer.add_audio(name, audio, sample_rate=sample_rate)
    
    def _log_text(self, text, name="text"):
        self.writer.add_text(name, text)
    
    def _log_inference_as_table(self, epoch, orig_text, predict, name):
        self.writer.add_table(
            table_name=name,
            data=[epoch, orig_text, predict],
            columns=["epoch", "temp", "target", "predict"]
        )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
