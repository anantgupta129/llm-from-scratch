import os
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import TrainingConfig
from model import SmolLM2Vision

class ProjectionTrainer:
    def __init__(
        self,
        *,
        model: SmolLM2Vision,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.model.set_training_mode('projection')
        
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.train_losses = []
        self.eval_losses = []
        
    def _create_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def evaluate(self, eval_dataloader: DataLoader) -> dict:
        """Evaluate the model on the validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    labels=batch.get("labels")
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.model.train()
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity
        }
        
    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader):
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        
        scheduler = self._create_scheduler(num_training_steps)
        
        if eval_dataloader is not None:
            print("\nRunning initial evaluation...")
            eval_metrics = self.evaluate(eval_dataloader)
            print(f"Initial eval loss: {eval_metrics['eval_loss']:.4f}")
            print(f"Initial eval perplexity: {eval_metrics['eval_perplexity']:.4f}")
            self.eval_losses.append((0, eval_metrics['eval_loss']))
        
            
        self.model.train()
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            epoch_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}: Training")
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    labels=batch.get("labels")
                )
                
                loss = outputs.loss
                
                # Backward pass with gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    current_loss = loss.item() * self.config.gradient_accumulation_steps
                    self.train_losses.append((self.global_step, current_loss))
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = loss.item() * self.config.gradient_accumulation_steps
                        current_lr = scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'step': self.global_step
                        })
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                
                epoch_loss += loss.item()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Average epoch loss: {avg_epoch_loss:.4f}")

            # End of epoch evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                print(f"  Evaluation loss: {eval_metrics['eval_loss']:.4f}")
                print(f"  Evaluation perplexity: {eval_metrics['eval_perplexity']:.4f}")
                self.eval_losses.append((self.global_step, eval_metrics['eval_loss']))
        
        final_eval_metrics = None
        if eval_dataloader is not None:
            print("\nRunning final evaluation...")
            final_eval_metrics = self.evaluate(eval_dataloader)
            print(f"Final eval loss: {final_eval_metrics['eval_loss']:.4f}")
            print(f"Final eval perplexity: {final_eval_metrics['eval_perplexity']:.4f}")
            
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        self._print_training_summary()
        print("Training completed!")
    
    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint"""
        checkpoint_name = "final" if is_final else f"checkpoint-{self.global_step}"
        save_dir = os.path.join(self.config.output_dir, checkpoint_name)
        
        # Save model
        self.model.save_projection(save_dir)
        
        # Save training state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'config': self.config.model_dump(),
            'training_mode': 'projection'
        }, os.path.join(save_dir, "trainer_state.pt"))
        
        print(f"Checkpoint saved to {save_dir}")
    
    def resume_from_checkpoint(self, checkpoint_dir: str):
        """Resume training from checkpoint"""
        
        # Load model weights
        self.model.load_projection_weights(
            os.path.join(checkpoint_dir, "mm_projector.bin")
        )
        
        # Load training state
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            state = torch.load(trainer_state_path)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.global_step = state['global_step']
            self.current_epoch = state['epoch']
            print(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
    
    def _print_training_summary(self):
        """Print training summary statistics"""
        print("\n\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        if self.train_losses:
            train_losses_values = [loss for _, loss in self.train_losses]
            print(f"Training Loss:")
            print(f"  Initial: {train_losses_values[0]:.4f}")
            print(f"  Final: {train_losses_values[-1]:.4f}")
            print(f"  Min: {min(train_losses_values):.4f}")
            print(f"  Max: {max(train_losses_values):.4f}")
        
        if self.eval_losses:
            eval_losses_values = [loss for _, loss in self.eval_losses]
            print(f"\nValidation Loss:")
            print(f"  Initial: {eval_losses_values[0]:.4f}")
            print(f"  Final: {eval_losses_values[-1]:.4f}")
            print(f"  Best: {min(eval_losses_values):.4f}")
            
            # Find best checkpoint
            best_step = self.eval_losses[eval_losses_values.index(min(eval_losses_values))][0]
            print(f"  Best checkpoint: step {best_step}")
        
        print("="*50)
        