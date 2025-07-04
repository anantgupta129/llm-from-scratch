import argparse
import os

import torch

from config import ModelConfig, DataConfig, TrainingConfig
from model import SmolLM2Vision
from data import DataModule
from trainer import ProjectionTrainer, train_instruction


def setup_multi_gpu():
    """Setup for multi-GPU training on Kaggle"""
    # Set environment variables for better multi-GPU performance
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n🖥️  GPUs Available: {num_gpus}")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)")
    
    return num_gpus

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="projection",
                       choices=["projection", "lora"],
                       help="Training stage: projection or lora")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume from checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to use (for debugging)")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs to train for")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multi-GPU training")
    parser.add_argument("--language_model", type=str, default="HuggingFaceTB/SmolLM2-1.7B",
                       help="Language model to use")
    parser.add_argument("--projection_checkpoint", type=str, default=None,
                       help="Path to MM projection weight file")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of gradient accumulation steps")
    
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load language model in 8-bit (only for LoRA stage)")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load language model in 4-bit (only for LoRA stage)")
    
    args = parser.parse_args()
    
    if (args.load_in_8bit or args.load_in_4bit) and args.stage == "projection":
        print("⚠️  Warning: Quantization is only supported for LoRA stage. Ignoring quantization flags.")
        args.load_in_8bit = False
        args.load_in_4bit = False
        
    if args.stage == "lora":
        print(f"\nProjection Checkpoint: {args.projection_checkpoint}")
        if not args.projection_checkpoint:
            raise ValueError("--projection_checkpoint is required for LoRA training")
    
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization. Choose one.")
        
    num_gpus = setup_multi_gpu()
    use_multi_gpu = num_gpus > 1 and args.multi_gpu
    
    model_config = ModelConfig(language_model=args.language_model)
    data_config = DataConfig()
    
    training_config = TrainingConfig(
        num_epochs=args.num_epochs, 
        output_dir=args.output_dir, 
        multi_gpu=use_multi_gpu,
        projection_checkpoint=args.projection_checkpoint,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    
    if args.num_samples is not None:
        data_config.num_samples = args.num_samples
    
    # Adjust batch size for multi-GPU
    if use_multi_gpu:
        print("\n✅ Multi-GPU Training Enabled")
        print(f"   Batch size per GPU: {training_config.batch_size}")
        print(f"   Total batch size: {training_config.batch_size * num_gpus}")
        
    print(f"Training Stage: {args.stage}")
    print(f"Model Config: {model_config.model_dump()}")
    print(f"Data Config: {data_config.model_dump()}")
    
    # Create model
    print("\nInitializing model...")
    if args.stage == "lora" and (args.load_in_8bit or args.load_in_4bit):
        print(f"Using {'8-bit' if args.load_in_8bit else '4-bit'} quantization for language model")
        training_config.fp16 = True
        model = SmolLM2Vision(model_config, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    else:
        model = SmolLM2Vision(model_config)
        
    # Create data module
    print("Setting up data...")
    data_module = DataModule(data_config)
    data_module.setup(model_config)
        
    if args.stage == "projection":
        bs = training_config.batch_size * num_gpus if use_multi_gpu else training_config.batch_size

        train_loader = data_module.get_train_dataloader(batch_size=bs, mode="pretrain")
        test_loader = data_module.get_test_dataloader(batch_size=bs, mode="pretrain")
        
        # Create trainer
        trainer = ProjectionTrainer(model=model, config=training_config)
        if args.resume_from:
            trainer.resume_from_checkpoint(args.resume_from)
            
        trainer.train(train_loader, test_loader)
    else:
        train_dataset = data_module.create_dataset(split="train", mode="instruct")
        eval_dataset = data_module.create_dataset(split="test", mode="instruct")
        
        if len(train_dataset.tokenizer) > model.language_model.config.vocab_size:
            print(f"Resizing model embeddings from {model.language_model.config.vocab_size} to {len(train_dataset.tokenizer)}")
            model.language_model.resize_token_embeddings(len(train_dataset.tokenizer))
            
        train_instruction(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
        )
    

if __name__ == "__main__":
    run()
    