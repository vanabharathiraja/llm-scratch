"""
Simple character-level tokenizer for the GPT model.
For production use, consider using BPE tokenizer like tiktoken.
"""

import pickle
import os


class CharTokenizer:
    """Character-level tokenizer."""
    
    def __init__(self, chars=None):
        if chars is None:
            # Default ASCII printable characters
            chars = [chr(i) for i in range(32, 127)]
        
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, text):
        """Encode text to list of integers."""
        return [self.stoi.get(c, 0) for c in text]
    
    def decode(self, tokens):
        """Decode list of integers to text."""
        return ''.join([self.itos.get(i, '') for i in tokens])
    
    @classmethod
    def from_text(cls, text):
        """Create tokenizer from text corpus."""
        chars = sorted(list(set(text)))
        return cls(chars)
    
    def save(self, path):
        """Save tokenizer to file."""
        with open(path, 'wb') as f:
            pickle.dump({'chars': self.chars}, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['chars'])


class SimpleTokenizer:
    """Simple word-level tokenizer with special tokens."""
    
    def __init__(self, vocab=None):
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        if vocab is None:
            vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.itos = {i: token for i, token in enumerate(vocab)}
        
        self.pad_token_id = self.stoi[self.pad_token]
        self.unk_token_id = self.stoi[self.unk_token]
        self.bos_token_id = self.stoi[self.bos_token]
        self.eos_token_id = self.stoi[self.eos_token]
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to list of integers."""
        tokens = text.split()
        token_ids = [self.stoi.get(token, self.unk_token_id) for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode list of integers to text."""
        tokens = []
        for idx in token_ids:
            token = self.itos.get(idx, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    @classmethod
    def from_text(cls, text, min_freq=1):
        """Create tokenizer from text corpus."""
        from collections import Counter
        
        # Count word frequencies
        words = text.split()
        word_counts = Counter(words)
        
        # Build vocabulary
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        vocab = special_tokens + [word for word, count in word_counts.items() if count >= min_freq]
        
        return cls(vocab)
    
    def save(self, path):
        """Save tokenizer to file."""
        with open(path, 'wb') as f:
            pickle.dump({'vocab': self.vocab}, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['vocab'])
