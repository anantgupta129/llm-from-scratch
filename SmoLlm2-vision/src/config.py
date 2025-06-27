from typing import Literal
from pydantic import BaseModel


class ModelConfig(BaseModel):
    
    vision_encoder: str = 'google/siglip-base-patch16-224'
    language_model:  str = 'HuggingFaceTB/SmolLM2-1.7B'
    projection_type: Literal['mlp', 'linear'] = 'mlp'
    init_projection: bool = True


class DataConfig(BaseModel):
    
    dataset_name: str = 'liuhaotian/LLaVA-CC3M-Pretrain-595K'
    max_len: int = 512
    split: str = "train"
    
    num_samples: int | None = None
    """Number of samples to load. If None, all samples are loaded, Used for testing."""
    
    test_size: float = 0.25
    random_state: int = 42


class TrainingConfig(BaseModel):
    
    batch_size: int = 16
    num_epochs: int = 3
    output_dir: str = "./checkpoints"
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-3
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    multi_gpu: bool = False
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    
    fp16: bool = False
    dataloader_num_workers: int = 4
