import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, SiglipModel

from config import ModelConfig
from .projection import ProjectionLayer


class SmolLM2Vision(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()        
        
        self.config = config
        
        # Load vision model
        print(f"Loading vision encoder: {config.vision_encoder}")
        self.vision_tower = AutoModel.from_pretrained(config.vision_encoder)
        
        # Load language model
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
        
        print(f"Vision hidden size: {self.vision_hidden_size}")
        print(f"Text hidden size: {self.text_hidden_size}")
        
        # Create projection layer
        self.mm_projection = ProjectionLayer(
            input_dim=self.vision_hidden_size,
            output_dim=self.text_hidden_size,
            projection_type=self.config.projection_type
        )
        
        # Initialize weights
        self._initialize_projection_weights()
        
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
    ):
        """Prepare inputs for multimodal forward pass"""
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
        
        # Concatenate (prepend images)
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
                labels=labels
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

    def set_training_mode(self, mode: Literal["projection", "lora_finetune"]):
        """Set which parameters should be trainable"""
        # Freeze base models
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        # Handle projection layer
        for param in self.mm_projection.parameters():
            if mode == "projection":
                # Only projection layer is trainable
                param.requires_grad = True            
            elif mode == "lora_finetune":
                param.requires_grad = False
            
        # LoRA weights will be added separately in stage 2
        
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
    
    def save_projection(self, save_dir: str | Path):
        """Save model components"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save projection layer
        torch.save(
            self.mm_projection.state_dict(), 
            save_dir / "mm_projector.bin"
        )
        
        # Save config as JSON
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)
        
        print(f"Model saved to {save_dir}")
    
    def load_projection_weights(self, checkpoint_path: str | Path):
        """Load projection layer weights from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.is_file():
            # Direct path to weights file
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            # Directory containing checkpoint
            weights_path = checkpoint_path / "mm_projector.bin"
            state_dict = torch.load(weights_path, map_location='cpu')
        
        self.mm_projection.load_state_dict(state_dict)
        print(f"Loaded projection weights from {checkpoint_path}")