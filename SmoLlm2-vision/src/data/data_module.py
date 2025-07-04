import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal, Any, Optional, Union
import json
import zipfile
import io

from datasets import load_dataset, Dataset
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from config import DataConfig, ModelConfig


class MultiModalDataset(TorchDataset):
    def __init__(
        self,
        data,
        processor,
        tokenizer,
        mode: Literal['pretrain', 'instruct'],
        max_len: int = 512,
    ):
        super().__init__()
        
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        
        # Special tokens for instruct tuning
        self.image_token = "<image>"
        self.ignore_index = -100
        self.image_token_id: int | None = None
        
        if self.mode == "instruct":
            self._setup_image_token()
        
    def __len__(self):
        return len(self.data)
    
    def _setup_image_token(self):
        
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': [self.image_token]
            })
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)            
            
    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or return if already PIL Image"""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, str):
            # If it's a path, load it
            return Image.open(image).convert('RGB')
        else:
            raise ValueError(f"Unknown image type: {type(image)}")
    
    def _process_pretrain(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process sample for pretraining (simple image-caption pairs)"""
        
        # Get image
        image = self._load_image(item['image'])
        
        # Get caption - handle different formats
        caption = item.get('caption', '')
        if not caption and 'conversations' in item:
            # Extract from conversations (LLaVA format)
            for conv in item['conversations']:
                if conv.get('from') == 'gpt':
                    caption = conv.get('value', '')
                    break
        
        # Process image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # Tokenize caption
        text_encoding = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': text_encoding.input_ids.squeeze(0),
            'attention_mask': text_encoding.attention_mask.squeeze(0),
            'labels': text_encoding.input_ids.squeeze(0)
        }
    
    def _process_instruct(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process sample for instruction tuning (conversations)"""
        # Get image
        image = self._load_image(item['image'])
        
        # Process image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # Build conversation
        conversations = item.get('conversations', [])
        
        # Simple format for now - concatenate all turns
        full_text = ""
        assistant_starts = []  # Track where assistant responses start
        
        for i, conv in enumerate(conversations):
            role = conv.get('from', '')
            value = conv.get('value', '').strip()
            
            if role == 'human':
                # For first human turn, ensure image token is present
                if i == 0:
                    if self.image_token not in value:
                        value = f"{self.image_token}\n{value}"
                else:
                    # Remove image token from subsequent turns
                    value = value.replace(self.image_token, '').strip()
                
                full_text += f"Human: {value}\n"
                
            elif role == 'gpt':
                # Mark the start of assistant response
                assistant_starts.append(len(full_text))
                full_text += f"Assistant: {value}\n"
        
        # Tokenize full conversation
        text_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = text_encoding.input_ids.squeeze(0)
        attention_mask = text_encoding.attention_mask.squeeze(0)
        
        labels = input_ids.clone()        
        labels[:] = self.ignore_index
        
        # Unmask assistant responses
        # Simple approach: find "Assistant:" tokens and unmask everything after them
        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Find all occurrences of "Assistant:" in the decoded text
        current_pos = 0
        while True:
            assistant_pos = decoded_text.find("Assistant:", current_pos)
            if assistant_pos == -1:
                break
            
            # Find the next "Human:" or end of text
            next_human_pos = decoded_text.find("Human:", assistant_pos)
            if next_human_pos == -1:
                next_human_pos = len(decoded_text)
            
            # Approximate token positions (this is simplified)
            # In practice, you'd want exact token-level alignment
            start_ratio = assistant_pos / len(decoded_text)
            end_ratio = next_human_pos / len(decoded_text)
            
            start_token = int(start_ratio * len(input_ids))
            end_token = int(end_ratio * len(input_ids))
            
            # Unmask these tokens
            if start_token < len(labels) and end_token <= len(labels):
                labels[start_token:end_token] = input_ids[start_token:end_token]
            
            current_pos = next_human_pos
        
        # Fallback: if no assistant responses found, unmask second half
        if torch.all(labels == self.ignore_index):
            mid_point = len(labels) // 2
            labels[mid_point:] = input_ids[mid_point:]
        
        # Find image token positions
        image_token_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'image_token_positions': image_token_positions,
        }
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample"""
        item = self.data[idx]
        
        if self.mode == "pretrain":
            return self._process_pretrain(item)
        elif self.mode == "instruct":
            return self._process_instruct(item)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def load_llava_pretrain_dataset(
    num_samples: int | None = None,
    load_images: bool = True,
    test_size: float = 0.1,
    random_state: int = 42
) -> tuple[list, list]:
    """Load LLaVA dataset with train/test split"""
    
    # Download chat.json
    chat_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
        filename="chat.json",
        repo_type="dataset"
    )
    
    # Load conversations
    print("Loading chat data...")
    with open(chat_path, 'r') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
    
    if load_images:
        # Download and open images.zip
        print("Downloading images.zip...")
        images_path = hf_hub_download(
            repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
            filename="images.zip",
            repo_type="dataset"
        )
        
        print("Loading images from zip...")
        zip_file = zipfile.ZipFile(images_path, 'r')
        
        # Load images
        processed_data = []
        for idx, item in enumerate(data):
            if idx % 10000 == 0:
                print(f"Loading images: {idx}/{len(data)}")
            
            try:
                with zip_file.open(item['image']) as img_file:
                    pil_image = Image.open(io.BytesIO(img_file.read()))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Create new item with PIL image
                    new_item = {**item, 'image': pil_image}
                    processed_data.append(new_item)
            except Exception as e:
                print(f"Error loading image {item['image']}: {e}")
                continue
        
        zip_file.close()
        data = processed_data
    
    # Split into train and test
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    return train_data, test_data


class DataModule:
    def __init__(self, config: DataConfig):
        self.config = config
        self.processor = None
        self.tokenizer = None
        self.train_data = None
        self.test_data = None
        
    def setup(self, model_config: ModelConfig):
        """Setup processor and tokenizer"""
        self.processor = AutoProcessor.from_pretrained(model_config.vision_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.language_model)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load and split data
        self._load_data()
    
    def _load_data(self):
        """Load and split dataset"""
        if self.config.dataset_name == "liuhaotian/LLaVA-CC3M-Pretrain-595K":
            # Use custom loader for LLaVA dataset
            self.train_data, self.test_data = load_llava_pretrain_dataset(
                num_samples=self.config.num_samples,
                load_images=True,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
        else:
            # Generic dataset loading
            self.train_data = load_dataset(
                self.config.dataset_name,
                split='train'
            )
            self.test_data = load_dataset(
                self.config.dataset_name,
                split='test'
            )

    def create_dataset(
        self, 
        split: Literal["train", "test"] = "train",
        mode: Literal["pretrain", "instruct"] = "pretrain"
    ) -> MultiModalDataset:
        """Create dataset for specified split"""
        
        if split == "train":
            data = self.train_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if data is None:
            raise ValueError("Data not loaded. Call setup() first.")
        
        # Create dataset
        return MultiModalDataset(
            data=data,
            processor=self.processor,
            tokenizer=self.tokenizer,
            mode=mode,
            max_len=self.config.max_len
        )
    
    def create_dataloader(
        self,
        dataset: MultiModalDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        drop_last: bool = False
    ) -> DataLoader:
        """Create dataloader"""
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_train_dataloader(self, batch_size: int, mode: str = "pretrain") -> DataLoader:
        """Get training dataloader"""
        train_dataset = self.create_dataset(split="train", mode=mode)
        return self.create_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
    
    def get_test_dataloader(self, batch_size: int, mode: str = "pretrain") -> DataLoader:
        """Get test dataloader"""
        test_dataset = self.create_dataset(split="test", mode=mode)
        return self.create_dataloader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False
        )


# Example usage
if __name__ == "__main__":
    
    # Initialize
    data_config = DataConfig(num_samples=10, test_size=0.3)
    model_config = ModelConfig()
    
    # Create data module
    data_module = DataModule(data_config)
    data_module.setup(model_config)
    
    # Get dataloaders
    train_loader = data_module.get_train_dataloader(batch_size=4)
    test_loader = data_module.get_test_dataloader(batch_size=4)
    
    # Test iteration
    print("\nTesting train dataloader...")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        break
    