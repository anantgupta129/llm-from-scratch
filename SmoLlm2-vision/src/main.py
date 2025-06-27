import argparse
import torch

from config import ModelConfig, DataConfig, TrainingConfig
from model import SmolLM2Vision
from data import DataModule
from trainer import ProjectionTrainer

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
    args = parser.parse_args()
    
    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = TrainingConfig(num_epochs=args.num_epochs)
    
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    
    if args.num_samples is not None:
        data_config.num_samples = args.num_samples
    
    print(f"Training Stage: {args.stage}")
    print(f"Model Config: {model_config.model_dump()}")
    print(f"Data Config: {data_config.model_dump()}")
    
    # Create model
    print("\nInitializing model...")
    model = SmolLM2Vision(model_config)
    
    # Create data module
    print("Setting up data...")
    data_module = DataModule(data_config)
    data_module.setup(model_config)

    # Get dataloaders
    mode = "pretrain" if args.stage == "projection" else "instruct"
    train_loader = data_module.get_train_dataloader(batch_size=training_config.batch_size, mode=mode)
    test_loader = data_module.get_test_dataloader(batch_size=training_config.batch_size, mode=mode)
    
    if mode == "pretrain":
        # Create trainer
        trainer = ProjectionTrainer(model=model, config=training_config)
        if args.resume_from:
            trainer.resume_from_checkpoint(args.resume_from)
            
        trainer.train(train_loader, test_loader)
    else:
        # TODO: implement lora training
        raise NotImplementedError
    

if __name__ == "__main__":
    run()
    