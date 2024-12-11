import os
import sys
import logging
import time
from datetime import datetime
import json
import psutil
try:
    import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy

import transformers
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.optimization import AdamW, get_scheduler

from dataclasses import dataclass, field, fields
from typing import Optional
import math

from model import Encoder, ContrastiveLoss
from data import TableDataModule

# LoRA implementation
class LoRALayer(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = self.lora_alpha / self.r
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result

class LoRALinear(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.base_layer = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return base_out + lora_out

def add_lora_layers(model: nn.Module, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
    """Replace linear layers with LoRA versions"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
        else:
            add_lora_layers(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

def get_lora_params(model: nn.Module):
    """Get only LoRA parameters for optimization"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            params.extend([module.lora_A, module.lora_B])
    return params

@dataclass
class DataArguments:
    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={"help": "bert-base-cased, bert-base-uncased etc"}
    )
    data_path: str = field(
        default='./data/pretrain/',
        metadata={"help": "data path"}
    )
    max_token_length: int = field(
        default=128,
        metadata={"help": "Maximum token length for tokenization"}
    )
    max_row_length: int = field(
        default=100,
        metadata={"help": "Maximum rows per table"}
    )
    max_column_length: int = field(
        default=100,
        metadata={"help": "Maximum columns per table"}
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"}
    )
    valid_ratio: float = field(
        default=0.01,
        metadata={"help": "Validation split ratio"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    max_epoch: int = field(
        default=5,
        metadata={"help": "Maximum training epochs"}
    )
    electra: bool = field(
        default=False,
        metadata={"help": "Use ELECTRA objective"}
    )
    mask_ratio: float = field(
        default=0.15,
        metadata={"help": "Masking ratio"}
    )
    contrast_bipartite_edge: bool = field(
        default=False,
        metadata={"help": "Use contrastive edge objective"}
    )
    bipartite_edge_corrupt_ratio: float = field(
        default=0.3,
        metadata={"help": "Edge corruption ratio"}
    )
    checkpoint_dir: str = field(
        default='checkpoints',
        metadata={"help": "Directory for checkpoints"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    from_scratch: bool = field(
        default=False,
        metadata={"help": "Training from scratch"}
    )

@dataclass
class OptimizerConfig:
    batch_size: int = field(
        default=128,
        metadata={"help": "Training batch size"}
    )
    base_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Base learning rate"}
    )
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: transformers.SchedulerType = "linear"
    warmup_step_ratio: float = 0.05
    optimizer: str = "Adam"
    adam_w_mode: bool = True
    save_every_n_epochs: int = field(
        default=1,
        metadata={"help": "Save frequency in epochs"}
    )
    save_top_k: int = field(
        default=3,
        metadata={"help": "Number of best checkpoints to keep"}
    )
    checkpoint_path: str = field(
        default="",
        metadata={"help": "Path to checkpoint for finetuning"}
    )

    @classmethod
    def dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def get_optimizer(self, optim_groups, learning_rate):
        optimizer = self.optimizer.lower()
        optim_cls = {
            "adam": AdamW if self.adam_w_mode else Adam,
        }[optimizer]

        kwargs = {
            "lr": learning_rate,
            "eps": self.adam_epsilon,
            "betas": (self.adam_beta1, self.adam_beta2),
        }
        optimizer = optim_cls(optim_groups, **kwargs)
        return optimizer

class PlModel(pl.LightningModule):
    def __init__(self, model_config, optimizer_cfg, data_args):
        super().__init__()
        self.model = Encoder(model_config)
        self.model_config = model_config
        self.optimizer_cfg = optimizer_cfg
        self.data_args = data_args
        self.save_hyperparameters()

        if self.model_config.electra:
            self.dense = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)
            self.act = nn.GELU()
            self.dense_prediction = nn.Linear(self.model_config.hidden_size, 1)
            self.criterion = nn.BCEWithLogitsLoss()
            self.pre = BinaryPrecision(threshold=0.5)
            self.rec = BinaryRecall(threshold=0.5)
            self.f1 = BinaryF1Score(threshold=0.5)
            self.acc = BinaryAccuracy(threshold=0.5)
        elif self.model_config.contrast_bipartite_edge:
            self.con_loss = ContrastiveLoss(temperature=0.07)

        self.epoch_start_time = None
        self.epoch_times = []
        self.peak_gpu_memory = 0

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)
            self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu_memory)
            
            if NVIDIA_SMI_AVAILABLE:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                
                self.log_dict({
                    'gpu_memory_used_mb': info.used / (1024**2),
                    'gpu_utilization': gpu_util.gpu
                })
        
        self.log('epoch_time_seconds', epoch_time)

    def on_train_end(self):
        stats = {
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'average_epoch_time_seconds': sum(self.epoch_times) / len(self.epoch_times),
            'total_training_time_seconds': sum(self.epoch_times),
            'peak_gpu_memory_mb': float(self.peak_gpu_memory),
            'num_epochs': len(self.epoch_times),
            'epoch_times': self.epoch_times,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_ram_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            }
        }
        
        stats_path = os.path.join(self.trainer.logger.log_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def training_step(self, batch, batch_idx):
        if self.model_config.electra:
            outputs = self.model(batch)
            cell_embeds = outputs[0]
            hyperedge_outputs = outputs[1]
            col_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(batch.col_mask).squeeze())
            all_embeds = torch.cat([cell_embeds, col_embeds], axis=0)
            hidden_states = self.dense(all_embeds)
            hidden_states = self.act(hidden_states)
            logits = self.dense_prediction(hidden_states).view(-1)
            c_lbls = batch.electra_c
            h_lbls = batch.electra_h
            lbls = torch.cat([c_lbls, h_lbls])
            loss_pos = self.criterion(logits[lbls==1.], lbls[lbls==1.])
            loss_neg = self.criterion(logits[lbls==0.], lbls[lbls==0.])
            loss = loss_pos + loss_neg
        elif self.model_config.contrast_bipartite_edge:
            self.model_config.update({'edge_neg_view': 1})
            outputs1 = self.model(batch)
            hyperedge_outputs1 = outputs1[1]
            hyper_embeds1 = torch.index_select(hyperedge_outputs1, 0, torch.nonzero(batch.hyper_mask).squeeze())
            
            self.model_config.update({'edge_neg_view': 2})
            outputs2 = self.model(batch)
            hyperedge_outputs2 = outputs2[1]
            hyper_embeds2 = torch.index_select(hyperedge_outputs2, 0, torch.nonzero(batch.hyper_mask).squeeze())
            loss = self.con_loss(hyper_embeds1, hyper_embeds2)

        self.log("training_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_config.electra:
            outputs = self.model(batch)
            cell_embeds = outputs[0]
            hyperedge_outputs = outputs[1]
            col_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(batch.col_mask).squeeze())
            all_embeds = torch.cat([cell_embeds, col_embeds], axis=0)
            hidden_states = self.dense(all_embeds)
            hidden_states = self.act(hidden_states)
            logits = self.dense_prediction(hidden_states).view(-1)
            c_lbls = batch.electra_c
            h_lbls = batch.electra_h
            lbls = torch.cat([c_lbls, h_lbls])
            loss_pos = self.criterion(logits[lbls==1.], lbls[lbls==1.])
            loss_neg = self.criterion(logits[lbls==0.], lbls[lbls==0.])
            loss = loss_pos + loss_neg
            self.log("validation_loss", loss, prog_bar=True)
            return {"logits": logits, "labels": lbls}
        
        elif self.model_config.contrast_bipartite_edge:
            self.model_config.update({'edge_neg_view': 1})
            outputs1 = self.model(batch)
            hyperedge_outputs1 = outputs1[1]
            hyper_embeds1 = torch.index_select(hyperedge_outputs1, 0, torch.nonzero(batch.hyper_mask).squeeze())
            
            self.model_config.update({'edge_neg_view': 2})
            outputs2 = self.model(batch)
            hyperedge_outputs2 = outputs2[1]
            hyper_embeds2 = torch.index_select(hyperedge_outputs2, 0, torch.nonzero(batch.hyper_mask).squeeze())
            loss = self.con_loss(hyper_embeds1, hyper_embeds2)
            self.log("validation_loss", loss, prog_bar=True)
            return loss

    def validation_epoch_end(self, outputs):
        if self.model_config.electra:
            logits = torch.cat([out["logits"] for out in outputs], dim=0)
            labels = torch.cat([out["labels"] for out in outputs], dim=0).long()
            probs = torch.sigmoid(logits)
            precision = self.pre(probs, labels)
            recall = self.rec(probs, labels)
            f1_score = self.f1(probs, labels)
            acc = self.acc(probs, labels)
            
            self.log_dict({
                'val_f1': f1_score,
                'acc': acc,
                'val_precision': precision,
                'val_recall': recall
            }, prog_bar=True)

    def configure_optimizers(self):
        from dataclasses import asdict
        self.logger.log_hyperparams(asdict(self.optimizer_cfg))
        
        learning_rate = self.optimizer_cfg.base_learning_rate

        if self.data_args.use_lora and not self.data_args.from_scratch:
            # Only optimize LoRA parameters when using LoRA
            trainable_params = get_lora_params(self.model)
            optim_groups = [
                {"params": trainable_params, "weight_decay": self.optimizer_cfg.weight_decay}
            ]
        else:
            # Regular optimization of all parameters
            no_decay = ["bias", "LayerNorm.weight"]
            params_decay = [
                p for n, p in self.named_parameters() 
                if not any(nd in n for nd in no_decay)
            ]
            params_nodecay = [
                p for n, p in self.named_parameters()
                if any(nd in n for nd in no_decay)
            ]
            optim_groups = [
                {"params": params_decay, "weight_decay": self.optimizer_cfg.weight_decay},
                {"params": params_nodecay, "weight_decay": 0.0},
            ]
        
        optimizer = self.optimizer_cfg.get_optimizer(optim_groups, learning_rate)
        
        num_training_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        num_warmup_steps = int(self.optimizer_cfg.warmup_step_ratio * num_training_steps)
        
        scheduler = get_scheduler(
            self.optimizer_cfg.lr_scheduler_type,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "reduce_on_plateau": False,
            "monitor": "validation_loss",
        }]

class CustomCheckpoint(pl.callbacks.Callback):
    def __init__(self, dirpath, monitor="validation_loss", mode="min"):
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(dirpath, 'checkpoint')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        
        is_better = (self.mode == "min" and current < self.best_metric) or \
                   (self.mode == "max" and current > self.best_metric)
        
        if is_better:
            self.best_metric = current
            
            # Save model state
            model_path = os.path.join(self.checkpoint_dir, "mp_rank_00_model_states.pt")
            torch.save(pl_module.model.state_dict(), model_path)
            
            # Save optimizer state
            optim_path = os.path.join(self.checkpoint_dir, "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
            torch.save(trainer.optimizers[0].state_dict(), optim_path)

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((DataArguments, OptimizerConfig))
    parser = pl.Trainer.add_argparse_args(parser)
    
    data_args, optimizer_cfg, trainer_args = parser.parse_args_into_dataclasses()
    
    os.makedirs(data_args.checkpoint_dir, exist_ok=True)
    
    tb_logger = TensorBoardLogger(
        save_dir=data_args.checkpoint_dir,
        name="pretrain",
        default_hp_metric=True
    )

    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Optimizer config: {optimizer_cfg}")
    logger.info(f"Trainer arguments: {trainer_args}")

    if data_args.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    pl.utilities.seed.seed_everything(data_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    tokenizer.add_tokens(new_tokens)
    logger.info(f"Added new tokens: {new_tokens}")

    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    model_config.update({
        'vocab_size': len(tokenizer),
        "pre_norm": False,
        "activation_dropout": 0.1,
        "gated_proj": False,
        "electra": data_args.electra,
        "contrast_bipartite_edge": data_args.contrast_bipartite_edge
    })
    logger.info(f"Model config: {model_config}")

    # Create model
    model_module = PlModel(model_config, optimizer_cfg, data_args)

    # Load checkpoint if specified and apply LoRA if needed
    if optimizer_cfg.checkpoint_path:
        logger.info(f"Loading checkpoint from {optimizer_cfg.checkpoint_path}")
        state_dict = torch.load(optimizer_cfg.checkpoint_path, 
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict['module'].items():
            if 'model' in k:
                name = k[13:]  # remove `module.model.`
                new_state_dict[name] = v
        
        # Load with strict=False to allow missing LoRA parameters
        model_module.model.load_state_dict(new_state_dict, strict=False)
        
        # After loading base model weights, apply LoRA if specified
        if data_args.use_lora and not data_args.from_scratch:
            logger.info("Applying LoRA adaptation...")
            add_lora_layers(
                model_module.model,
                r=data_args.lora_r,
                lora_alpha=data_args.lora_alpha,
                lora_dropout=data_args.lora_dropout
            )
            logger.info("LoRA layers added successfully")

        # Run initial validation
        logger.info("Running initial validation with loaded checkpoint...")
        validation_trainer = pl.Trainer.from_argparse_args(
            trainer_args,
            strategy="deepspeed_stage_1",
            callbacks=[],  # No callbacks needed for validation
            logger=tb_logger,
            max_epochs=0,  # No training, just validation
            enable_checkpointing=False,  # Disable checkpointing for validation
            precision='bf16'
        )
        
        # Set up data module for validation
        data_module = TableDataModule(
            tokenizer=tokenizer,
            data_args=data_args,
            seed=data_args.seed,
            batch_size=optimizer_cfg.batch_size,
            py_logger=logger,
            objective='electra' if model_config.electra else 'contrast'
        )
        data_module.setup('validate')
        
        validation_results = validation_trainer.validate(model_module, datamodule=data_module)
        logger.info(f"Initial validation results: {validation_results}")

    # Set up data module (for training)
    data_module = TableDataModule(
        tokenizer=tokenizer,
        data_args=data_args,
        seed=data_args.seed,
        batch_size=optimizer_cfg.batch_size,
        py_logger=logger,
        objective='electra' if model_config.electra else 'contrast'
    )

    # Configure callbacks with simplified checkpoint system
    checkpoint_dir = os.path.join(data_args.checkpoint_dir)
    callbacks = [
        CustomCheckpoint(
            dirpath=checkpoint_dir,
            monitor="validation_loss" if data_args.contrast_bipartite_edge else "val_f1",
            mode="min" if data_args.contrast_bipartite_edge else "max"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
    ]
    
    # Configure logger to save directly to checkpoint_dir
    tb_logger = TensorBoardLogger(
        save_dir=checkpoint_dir,
        name=None,  # This prevents creation of "pretrain" subdirectory
        version='',  # This prevents creation of version subdirectories
        default_hp_metric=False  # This prevents creation of hparams.yaml
    )

    # Configure training
    if trainer_args.gpus == -1:
        trainer_args.gpus = torch.cuda.device_count()
    
    assert trainer_args.replace_sampler_ddp == False, "replace_sampler_ddp must be False for correct data sampling"

    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        trainer_args,
        strategy="deepspeed_stage_1",
        callbacks=callbacks,
        logger=tb_logger,
        max_epochs=data_args.max_epoch,
        precision='bf16',
        accumulate_grad_batches=getattr(trainer_args, 'accumulate_grad_batches', 1),
        gradient_clip_val=getattr(trainer_args, 'gradient_clip_val', None),
    )

    # # Load from DeepSpeed checkpoint if specified
    # if optimizer_cfg.checkpoint_path:
    #     logger.info(f"Loading model from DeepSpeed checkpoint: {optimizer_cfg.checkpoint_path}")
    #     # Load DeepSpeed checkpoint
    #     state_dict = torch.load(optimizer_cfg.checkpoint_path)
    #     # DeepSpeed saves the model state directly, not under "state_dict" key
    #     missing_keys, unexpected_keys = model_module.model.load_state_dict(state_dict, strict=False)
    #     logger.info(f"Loaded checkpoint. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")


    # Start training
    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()