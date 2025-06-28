import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from config import TrainingConfig
from model import SmolLM2Vision


class InstructionTrainer(Trainer):
    """Trainer for instruction fine-tuning with LoRA"""
    
    def __init__(
        self, 
        model: SmolLM2Vision, 
        training_config: TrainingConfig,
        **kwargs
    ):
        # Load projection weights
        print(f"Loading projection weights from: {training_config.projection_checkpoint}")
        model.load_checkpoint(training_config.projection_checkpoint, load_lora=False)
        
        model.set_training_mode('lora_finetune')
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            warmup_ratio=training_config.warmup_ratio,
            weight_decay=training_config.weight_decay,
            max_grad_norm=training_config.max_grad_norm,
            fp16=training_config.fp16,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=training_config.dataloader_num_workers,
            remove_unused_columns=False,
        )
        
        # Create custom optimizer with different learning rates
        optimizers = (self._create_optimizer(model, training_config), None)
        
        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=training_args,
            optimizers=optimizers,
            **kwargs
        )
        
        self.training_config = training_config
        
    def _create_optimizer(self, model: SmolLM2Vision, config: TrainingConfig):
        """Create optimizer with different learning rates for projection and LoRA"""
        projection_params = []
        lora_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "mm_projection" in name:
                    projection_params.append(param)
                else:
                    lora_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": projection_params,
                "lr": config.learning_rate * 0.1,  # Lower LR for pre-trained projection
            },
            {
                "params": lora_params,
                "lr": config.learning_rate,
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=config.weight_decay
        )
        
        return optimizer
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override _save to use our custom checkpoint saving"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use our model's save method (saves both projection and LoRA)
        self.model.save_checkpoint(output_dir)
        
        # Save trainer state
        torch.save({
            'global_step': self.state.global_step,
            'epoch': self.state.epoch,
            'best_metric': self.state.best_metric,
            'best_model_checkpoint': self.state.best_model_checkpoint,
        }, os.path.join(output_dir, "trainer_state.pt"))


def train_instruction(
    model: SmolLM2Vision,
    train_dataset,
    eval_dataset,
    training_config: TrainingConfig,
):
    """Train instruction following with LoRA"""
    
    trainer = InstructionTrainer(
        model=model,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    last_checkpoint = get_last_checkpoint(training_config.output_dir)
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    
    return trainer
