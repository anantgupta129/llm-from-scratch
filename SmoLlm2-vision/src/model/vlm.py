import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, SiglipModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from config import ModelConfig
from .projection import ProjectionLayer


class SmolLM2Vision(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()        
        
        self.config = config
        
        print(f"Loading vision encoder: {config.vision_encoder}")
        self.vision_tower = AutoModel.from_pretrained(config.vision_encoder)
        
        print(f"Loading language model: {config.language_model}")
        self.language_model = AutoModelForCausalLM.from_pretrained(config.language_model)
        
        # Get hidden sizes - handle SigLIP specifically
        if isinstance(self.vision_tower, SiglipModel):
            # SigLIP model structure
            self.vision_hidden_size = self.vision_tower.config.vision_config.hidden_size
        else:
            raise ValueError(f"Unknown vision model type: {type(self.vision_tower)}")
        
        # Language model hidden size
        self.text_hidden_size = self.language_model.config.hidden_size
        
        # Create projection layer
        self.mm_projection = ProjectionLayer(
            input_dim=self.vision_hidden_size,
            output_dim=self.text_hidden_size,
            projection_type=self.config.projection_type
        )
        
        # Initialize weights
        self._initialize_projection_weights()
        
        self.lora_config = None
        self.lora_applied = False
        
    def _initialize_projection_weights(self):
        """Initialize projection layer weights"""
        if hasattr(self.mm_projection, 'projection'):
            if isinstance(self.mm_projection.projection, nn.Linear):
                nn.init.xavier_uniform_(self.mm_projection.projection.weight)
                nn.init.zeros_(self.mm_projection.projection.bias)
            elif isinstance(self.mm_projection.projection, nn.Sequential):
                for layer in self.mm_projection.projection:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
    
    @property
    def get_vision_model(self):
        return self.vision_tower
    
    @property
    def get_language_model(self):
        return self.language_model
    
    @property
    def get_projection_layer(self):
        return self.mm_projection
    
    def encode_image(self, x: torch.Tensor):
        """Encode images through vision encoder and projection layer"""
        # Get vision features - handle different model types
        if isinstance(self.vision_tower, SiglipModel):
            # Get all patch features
            vision_outputs = self.vision_tower.vision_model(x)
            # Use last_hidden_state to get all 196 patches
            # Shape: [batch_size, 196, 768]
            image_features = vision_outputs.last_hidden_state
        else:
            raise ValueError(f"Unknown vision model type: {type(self.vision_tower)}")
        
        # Project to language model dimension
        projection = self.mm_projection(image_features)
        
        return projection
    
    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        image_token_positions: torch.Tensor | None = None,
    ):
        """Prepare inputs for multimodal forward pass."""
        
        if pixel_values is None:
            # Text-only input
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        # Encode images
        image_features = self.encode_image(pixel_values)
        bs, num_patches, hidden_dim = image_features.shape
        
        # Get text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Handle image insertion based on training stage
        if image_token_positions is not None and len(image_token_positions) > 0:
            # Insert image features at specific positions, #for instruct training
            new_embeddings = []
            new_attention_mask = []
            new_labels = [] if labels is not None else None
            
            for i in range(bs):
                text_emb = inputs_embeds[i]
                img_positions = image_token_positions[i] if image_token_positions.dim() > 1 else image_token_positions
                
                if len(img_positions) == 0 or img_positions[0] == -1:
                    # No image tokens in this sample
                    new_embeddings.append(text_emb)
                    if attention_mask is not None:
                        new_attention_mask.append(attention_mask[i])
                    if labels is not None:
                        new_labels.append(labels[i])
                else:
                    # Insert image features at first image token position
                    pos = img_positions[0].item()
                    
                    # Split embeddings and insert image features
                    before = text_emb[:pos]
                    after = text_emb[pos+1:]  # Skip the image token itself
                    combined = torch.cat([before, image_features[i], after], dim=0)
                    new_embeddings.append(combined)
                    
                    # Update attention mask
                    if attention_mask is not None:
                        mask_before = attention_mask[i][:pos]
                        mask_after = attention_mask[i][pos+1:]
                        mask_image = torch.ones(num_patches, device=attention_mask.device)
                        new_mask = torch.cat([mask_before, mask_image, mask_after])
                        new_attention_mask.append(new_mask)
                    
                    # Update labels
                    if labels is not None:
                        label_before = labels[i][:pos]
                        label_after = labels[i][pos+1:]
                        label_image = torch.full((num_patches,), -100, device=labels.device)
                        new_label = torch.cat([label_before, label_image, label_after])
                        new_labels.append(new_label)
            
            # Pad sequences to same length
            max_len = max(emb.shape[0] for emb in new_embeddings)
            
            padded_embeddings = []
            padded_masks = []
            padded_labels = []
            
            for i, emb in enumerate(new_embeddings):
                pad_len = max_len - emb.shape[0]
                if pad_len > 0:
                    padding = torch.zeros(pad_len, emb.shape[1], device=emb.device, dtype=emb.dtype)
                    padded_emb = torch.cat([emb, padding], dim=0)
                    padded_embeddings.append(padded_emb)
                    
                    if attention_mask is not None:
                        mask_padding = torch.zeros(pad_len, device=attention_mask.device)
                        padded_mask = torch.cat([new_attention_mask[i], mask_padding], dim=0)
                        padded_masks.append(padded_mask)
                    
                    if labels is not None:
                        label_padding = torch.full((pad_len,), -100, device=labels.device)
                        padded_label = torch.cat([new_labels[i], label_padding], dim=0)
                        padded_labels.append(padded_label)
                else:
                    padded_embeddings.append(emb)
                    if attention_mask is not None:
                        padded_masks.append(new_attention_mask[i])
                    if labels is not None:
                        padded_labels.append(new_labels[i])
            
            inputs_embeds = torch.stack(padded_embeddings, dim=0)
            if attention_mask is not None:
                attention_mask = torch.stack(padded_masks, dim=0)
            if labels is not None:
                labels = torch.stack(padded_labels, dim=0)
                
        else:
            # this is used for pre-training projection layer
            inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)
            
            # Update attention mask
            image_attention = torch.ones(
                (bs, num_patches),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
            
            # Update labels if provided
            if labels is not None:
                image_labels = torch.full(
                    (bs, num_patches),
                    -100,  # Ignore index for loss calculation
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([image_labels, labels], dim=1)
        
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,  # Allow both names
        labels: torch.Tensor | None = None,
        image_token_positions: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs
    ):
        """Forward pass through the model"""
        
        # Handle both 'images' and 'pixel_values' argument names
        if pixel_values is None and images is not None:
            pixel_values = images
        
        if pixel_values is not None:
            # Multimodal forward pass
            inputs = self.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                image_token_positions=image_token_positions
            )
            
            # Forward through language model with prepared inputs
            outputs = self.language_model(
                inputs_embeds=inputs['inputs_embeds'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels'],
                return_dict=return_dict,
                **kwargs
            )
        else:
            # Text-only forward pass
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=return_dict,
                **kwargs
            )
        
        return outputs

    def _apply_lora(self):
        """Apply LoRA to the language model"""
        
        if self.lora_applied:
            print("LoRA already applied")
            return
            
        # Define LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            modules_to_save=["lm_head"],  # Also train the output layer
        )
        
        # Apply LoRA to language model
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.lora_config = lora_config
        self.lora_applied = True
        
        print("LoRA applied to language model")
        self.language_model.print_trainable_parameters()
        
    def set_training_mode(self, mode: Literal["projection", "lora_finetune"]):
        """Set which parameters should be trainable"""
        # Freeze base models
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        for param in self.mm_projection.parameters():
            param.requires_grad = True            
            
        # LoRA weights will be added separately in stage 2
        if mode == 'lora_finetune':
            for param in self.mm_projection.parameters():
                param.requires_grad = True
            
            # Apply LoRA if not already applied
            if not self.lora_applied:
                self._apply_lora()
        
        # Print training configuration
        self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_param
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_param:,} || "
              f"Trainable%: {trainable_percent:.2f}")
    
    def save_checkpoint(self, save_dir: str | Path):
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.mm_projection.state_dict(), save_dir / "mm_projector.bin")
        config_dict = self.config.model_dump()
                
        if self.lora_applied:
            # Save LoRA weights
            lora_dir = save_dir / "lora_weights"
            self.language_model.save_pretrained(lora_dir)
            
            if self.lora_config:
                config_dict["lora_config"] = self.lora_config.to_dict()
            
            print(f"LoRA weights saved to {save_dir}")
        
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
    
    def load_checkpoint(self, checkpoint_path: str | Path, load_lora: bool = False):
        """Load checkpoint (projection + optional LoRA).
        
        To load only projection weights, use full path to weights file. In that case `load_lora` is ignored.
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load projection weights
        if checkpoint_path.is_file():
            # Direct path to projection weights
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.mm_projection.load_state_dict(state_dict)
            print(f"Loaded projection weights from {checkpoint_path}")
        else:
            # Directory containing checkpoint
            projection_path = checkpoint_path / "mm_projector.bin"
            if projection_path.exists():
                state_dict = torch.load(projection_path, map_location='cpu')
                self.mm_projection.load_state_dict(state_dict)
                print(f"Loaded projection weights from {projection_path}")
            
            # Load LoRA weights if requested and available
            if load_lora:
                lora_path = checkpoint_path / "lora_weights"
                if lora_path.exists():
                    # Apply LoRA first if needed
                    if not self.lora_applied:
                        self._apply_lora()
                    
                    # Load LoRA weights
                    self.language_model = PeftModel.from_pretrained(
                        self.language_model.base_model,
                        lora_path
                    )
                    print(f"Loaded LoRA weights from {lora_path}")
    
    def merge_and_save(self, save_dir: str | Path):
        """Merge LoRA weights and save the full model"""
        
        if not self.lora_applied:
            print("No LoRA weights to merge")
            return
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge LoRA weights
        merged_model = self.language_model.merge_and_unload()
        
        merged_model.save_pretrained(save_dir / "language_model")
        
        torch.save(
            self.mm_projection.state_dict(), 
            save_dir / "mm_projector.bin"
        )
        
        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)
        
        print(f"Merged model saved to {save_dir}")
