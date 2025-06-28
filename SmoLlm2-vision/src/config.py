from typing import Literal
from pydantic import BaseModel

class BaseConfig(BaseModel):
    def to_json_string(self):
        return self.model_dump_json(indent=4, ensure_ascii=False)

class ModelConfig(BaseConfig):
    
    vision_encoder: str = 'google/siglip-base-patch16-224'
    language_model:  str = 'HuggingFaceTB/SmolLM2-1.7B'
    projection_type: Literal['mlp', 'linear'] = 'mlp'
    init_projection: bool = True
    
    # LoRA configuration for Stage 2
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = [
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # MLP
    ]


class DataConfig(BaseConfig):
    
    dataset_name: str = 'liuhaotian/LLaVA-CC3M-Pretrain-595K'
    max_len: int = 2048 #change to 512 for projection layer training
    split: str = "train"
    
    num_samples: int | None = None
    """Number of samples to load. If None, all samples are loaded, Used for testing."""
    
    test_size: float = 0.25
    random_state: int = 42


class TrainingConfig(BaseConfig):
    
    batch_size: int = 16
    num_epochs: int = 3
    output_dir: str = "./checkpoints"
    projection_checkpoint: str | None = None
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    multi_gpu: bool = False
    fp16: bool = True
    dataloader_num_workers: int = 2
    
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
