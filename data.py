import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import argparse
from tqdm import tqdm
import numpy as np
import random
import tiktoken
import gc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class WebTextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        
        # Create input sequences and targets (standard next token prediction)
        if len(tokens) > self.seq_len + 1:
            # Randomly select a starting point
            start_idx = random.randint(0, len(tokens) - self.seq_len - 1)
            tokens = tokens[start_idx:start_idx + self.seq_len + 1]
        
        # Ensure we have at least 2 tokens
        if len(tokens) < 2:
            # Pad with zeros if necessary
            tokens = tokens + [0] * (2 - len(tokens))
            
        # Make sure both input and target are the same length
        # Take all but the last token for input, all but the first token for target
        inputs = tokens[:-1]
        targets = tokens[1:]
        
        # Make sure inputs and targets are the same length
        assert len(inputs) == len(targets), f"Input length {len(inputs)} != target length {len(targets)}"
        
        # Make sure inputs don't exceed seq_len
        if len(inputs) > self.seq_len:
            inputs = inputs[:self.seq_len]
            targets = targets[:self.seq_len]
            
        x = torch.tensor(inputs, dtype=torch.long)
        y = torch.tensor(targets, dtype=torch.long)
        
        return x, y

def tokenize_text(text, encoder):
    """Tokenize a single text using tiktoken encoder"""
    return encoder.encode(text)

def tokenize_batch(examples, encoder):
    """Tokenize a batch of texts using tiktoken encoder"""
    return {"tokens": [encoder.encode(text) for text in examples["text"]]}

def load_and_prepare_openwebtext(
    save_path='./data',
    seq_len=1024,
    test_size=0.05,
    max_samples=None,
    batch_size=32,
    num_workers=4,
    seed=42,
    use_memory_mapping=True,
    tokenizer_name='gpt2'
):
    """
    Load and prepare the OpenWebText dataset for training using tiktoken.
    
    Args:
        save_path: Path to save processed data
        seq_len: Maximum sequence length
        test_size: Fraction of data to use for testing
        max_samples: Maximum number of samples to process (None for all)
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        use_memory_mapping: Whether to use memory mapping for large datasets
        tokenizer_name: Name of the tiktoken tokenizer to use
    
    Returns:
        train_dataloader, val_dataloader, tokenizer, vocab_size
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(save_path, exist_ok=True)
    tokenized_cache_path = os.path.join(save_path, 'tokenized_cache')
    train_tokens_path = os.path.join(save_path, 'train_tokens.pt')
    val_tokens_path = os.path.join(save_path, 'val_tokens.pt')
    
    # Initialize tiktoken encoder
    print(f"Initializing tiktoken encoder for {tokenizer_name}...")
    encoder = tiktoken.get_encoding(tokenizer_name)
    vocab_size = encoder.n_vocab
    print(f"Vocabulary size: {vocab_size}")
    
    # Check if processed data exists
    if os.path.exists(train_tokens_path) and os.path.exists(val_tokens_path):
        print("Loading pre-tokenized data...")
        
        # Load data using memory mapping if enabled
        if use_memory_mapping:
            train_data = torch.load(train_tokens_path, map_location='cpu')
            val_data = torch.load(val_tokens_path, map_location='cpu')
        else:
            train_data = torch.load(train_tokens_path)
            val_data = torch.load(val_tokens_path)
            
    else:
        print("Loading OpenWebText dataset...")
        dataset = load_dataset("openwebtext")
        
        if max_samples:
            # Select a subset of the data
            print(f"Using {max_samples} samples out of {len(dataset['train'])}")
            dataset['train'] = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
        
        print("Tokenizing texts with tiktoken (this may take a while)...")
        
        # Process in batches to avoid memory issues
        batch_size_tokenize = 1000
        tokenized_texts = []
        
        # Use multiprocessing for faster tokenization
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for i in tqdm(range(0, len(dataset['train']), batch_size_tokenize)):
                batch = dataset['train'][i:i+batch_size_tokenize]['text']
                batch_tokens = list(executor.map(tokenize_text, batch, [encoder] * len(batch)))
                tokenized_texts.extend([{"tokens": tokens} for tokens in batch_tokens])
                
                # Free memory
                if i % (batch_size_tokenize * 10) == 0:
                    gc.collect()
        
        # Split into train and validation
        print("Splitting into train and validation sets...")
        random.seed(seed)  # Reset seed for reproducibility
        random.shuffle(tokenized_texts)
        
        split_idx = int(len(tokenized_texts) * (1 - test_size))
        train_data = tokenized_texts[:split_idx]
        val_data = tokenized_texts[split_idx:]
        
        # Save tokenized data
        print(f"Saving tokenized data to {save_path}...")
        torch.save(train_data, train_tokens_path)
        torch.save(val_data, val_tokens_path)
    
    # Create datasets
    train_dataset = WebTextDataset(train_data, seq_len)
    val_dataset = WebTextDataset(val_data, seq_len)
    
    print(f"Dataset sizes: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    # Create a simple class that mimics the HuggingFace tokenizer interface
    class TikTokenWrapper:
        def __init__(self, encoder):
            self.encoder = encoder
            
        def encode(self, text, return_tensors=None):
            tokens = self.encoder.encode(text)
            if return_tensors == "pt":
                return torch.tensor([tokens])
            return tokens
            
        def decode(self, tokens):
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            return self.encoder.decode(tokens)
    
    tokenizer = TikTokenWrapper(encoder)
    
    return train_dataloader, val_dataloader, tokenizer, vocab_size

def main():
    parser = argparse.ArgumentParser(description='Prepare OpenWebText dataset')
    parser.add_argument('--save_path', type=str, default='./data', help='Path to save processed data')
    parser.add_argument('--seq_len', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.05, help='Test set size ratio')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to process')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--memory_mapping', action='store_true', help='Use memory mapping for large datasets')
    
    args = parser.parse_args()
    
    # Process the dataset
    train_dl, val_dl, tokenizer, vocab_size = load_and_prepare_openwebtext(
        save_path=args.save_path,
        seq_len=args.seq_len,
        test_size=args.test_size,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_memory_mapping=args.memory_mapping
    )
    
    # Display sample batch
    batch = next(iter(train_dl))
    x, y = batch
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    # Decode a sample
    sample_idx = 0
    sample_text = tokenizer.decode(x[sample_idx].tolist())
    print(f"\nSample text:\n{sample_text[:500]}...\n")
    
    print(f"Vocabulary size: {vocab_size}")
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
