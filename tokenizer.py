import re
from typing import List, Dict, Set
from collections import defaultdict

class SimpleTokenizer:
    def __init__(self):
        """Initialize the tokenizer with default settings."""
        self.vocab = {}  # token -> id mapping
        self.inverse_vocab = {}  # id -> token mapping
        self.vocab_size = 0
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3   # End of sequence
        }
        
        # Initialize vocab with special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
        
        self.vocab_size = len(self.special_tokens)
        
        # Token patterns
        self.word_pattern = re.compile(r'\b\w+\b')
        self.punct_pattern = re.compile(r'[^\w\s]')
        
    def _basic_tokenize(self, text: str) -> List[str]:
        """
        Basic tokenization: split on whitespace and separate punctuation.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = []
        
        # Find all words and punctuation
        words = self.word_pattern.findall(text.lower())
        punct = self.punct_pattern.findall(text)
        
        # Simple approach: tokenize by splitting on whitespace then handling punctuation
        parts = text.lower().split()
        for part in parts:
            # Check if part contains punctuation
            if any(p in part for p in '.,!?;:"()[]{}'):
                # Split punctuation from words
                current = ""
                for char in part:
                    if char.isalnum():
                        current += char
                    else:
                        if current:
                            tokens.append(current)
                            current = ""
                        if char.strip():  # Don't add whitespace as tokens
                            tokens.append(char)
                if current:
                    tokens.append(current)
            else:
                if part.strip():  # Don't add empty strings
                    tokens.append(part)
        
        return tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of training texts
            min_freq: Minimum frequency for a token to be included in vocab
        """
        # Count token frequencies
        token_freq = defaultdict(int)
        
        for text in texts:
            tokens = self._basic_tokenize(text)
            for token in tokens:
                token_freq[token] += 1
        
        # Add tokens that meet minimum frequency requirement
        for token, freq in token_freq.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.inverse_vocab[self.vocab_size] = token
                self.vocab_size += 1
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self._basic_tokenize(text)
        
        # Convert tokens to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<BOS>'])
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.special_tokens['<UNK>'])
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<EOS>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                    
                tokens.append(token)
            else:
                tokens.append('<UNK>')
        
        # Simple reconstruction - just join with spaces
        # In a more sophisticated tokenizer, you'd want to handle
        # punctuation and spacing more intelligently
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self.vocab.copy()
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for token, idx in self.vocab.items():
                f.write(f"{token}\t{idx}\n")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from a file."""
        self.vocab = {}
        self.inverse_vocab = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token, idx = line.split('\t')
                    idx = int(idx)
                    self.vocab[token] = idx
                    self.inverse_vocab[idx] = token
        
        self.vocab_size = len(self.vocab)


# Example usage and testing
if __name__ == "__main__":
    # Sample training data
    training_texts = [
        "Hello, world! This is a simple tokenizer.",
        "It can handle punctuation and basic text processing.",
        "The tokenizer splits text into tokens and assigns IDs.",
        "Hello there! How are you doing today?",
        "This is another example sentence for training."
    ]
    
    # Initialize and train tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_texts, min_freq=1)
    
    print("=== Simple Custom Tokenizer Demo ===")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print("\nSample vocabulary:")
    vocab_sample = list(tokenizer.get_vocab().items())[:10]
    for token, idx in vocab_sample:
        print(f"  '{token}' -> {idx}")
    
    # Test encoding and decoding
    test_text = "Hello, this is a test sentence!"
    print(f"\nOriginal text: '{test_text}'")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    # Test with unknown tokens
    unknown_text = "This contains some unknown vocabulary words like supercalifragilisticexpialidocious!"
    print(f"\nText with unknown words: '{unknown_text}'")
    encoded_unknown = tokenizer.encode(unknown_text)
    print(f"Encoded: {encoded_unknown}")
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"Decoded: '{decoded_unknown}'")