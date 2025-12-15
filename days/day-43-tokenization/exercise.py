"""
Day 43: Tokenization Strategies - Exercise

Business Scenario:
You're the NLP Engineer at a global AI company building multilingual language models. 
Your team needs to implement and optimize various tokenization strategies for different 
use cases: English-focused models, multilingual systems, and domain-specific applications 
(medical, legal, code).

Your task is to implement different tokenization algorithms, optimize them for production, 
and build evaluation frameworks to compare their effectiveness across different scenarios.

Requirements:
1. Implement BPE, WordPiece, and SentencePiece-style tokenizers from scratch
2. Create multilingual tokenization pipelines with proper normalization
3. Build domain adaptation capabilities for specialized vocabularies
4. Implement efficient production-ready tokenization with caching
5. Develop comprehensive evaluation metrics and benchmarking tools

Success Criteria:
- All tokenization algorithms produce correct and consistent results
- Multilingual pipeline handles CJK, RTL, and Indic scripts properly
- Domain adaptation improves tokenization quality for specialized text
- Production implementation achieves >1000 texts/second throughput
- Evaluation framework provides actionable insights for tokenizer selection
"""

import re
import json
import time
import unicodedata
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Any
import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer training and inference"""
    vocab_size: int = 30000
    min_frequency: int = 2
    special_tokens: List[str] = None
    normalization: str = "nfc"  # "nfc", "nfkc", "lowercase", "none"
    handle_unk: bool = True
    max_token_length: int = 50
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]


class TextNormalizer:
    """
    Comprehensive text normalization for multilingual text
    """
    
    def __init__(self, 
                 normalization_form: str = "nfc",
                 lowercase: bool = False,
                 remove_accents: bool = False,
                 handle_contractions: bool = True):
        self.normalization_form = normalization_form.upper()
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.handle_contractions = handle_contractions
        
        # Define contraction mappings
        self.contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is", "let's": "let us",
            "that's": "that is", "who's": "who is", "what's": "what is",
            "where's": "where is", "when's": "when is", "why's": "why is",
            "how's": "how is", "there's": "there is", "here's": "here is"
        }
        
    def normalize(self, text: str) -> str:
        """
        Apply comprehensive text normalization
        
        Args:
            text: Input text to normalize
            
        Returns:
            normalized_text: Normalized text
        """
        # Unicode normalization
        if hasattr(unicodedata, 'normalize'):
            text = unicodedata.normalize('NFC', text)
        
        # Handle contractions if enabled
        if self.handle_contractions:
            for contraction, expansion in self.contractions.items():
                text = text.replace(contraction, expansion)
        
        # Remove accents if enabled
        if self.remove_accents:
            text = ''.join(c for c in unicodedata.normalize('NFD', text)
                          if unicodedata.category(c) != 'Mn')
        
        # Apply lowercase if enabled
        if self.lowercase:
            text = text.lower()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implementation from scratch
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.normalizer = TextNormalizer()
        self.vocab = {}
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """
        Extract word frequencies from training corpus
        
        Args:
            texts: List of training texts
            
        Returns:
            word_freqs: Dictionary mapping words to frequencies
        """
        word_freqs = defaultdict(int)
        
        # Process each text and count word frequencies
        for text in texts:
            # Normalize text
            normalized = self.normalizer.normalize(text)
            
            # Split into words and count
            words = normalized.split()
            for word in words:
                if len(word) >= 1:  # Basic length filter
                    word_freqs[word] += 1
        
        return dict(word_freqs)
    
    def _initialize_vocabulary(self, word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Initialize vocabulary with characters and special tokens
        
        Args:
            word_freqs: Word frequency dictionary
            
        Returns:
            vocab: Initial vocabulary with frequencies
        """
        vocab = defaultdict(int)
        
        # Add special tokens
        for token in self.config.special_tokens:
            vocab[token] = 0  # Special tokens get frequency 0
        
        # Add all characters from the corpus
        for word, freq in word_freqs.items():
            for char in word:
                vocab[char] += freq
        
        return dict(vocab)
    
    def _get_pairs(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Get all adjacent pairs and their frequencies
        
        Args:
            word_freqs: Current word frequencies with space-separated tokens
            
        Returns:
            pairs: Dictionary mapping pairs to frequencies
        """
        pairs = defaultdict(int)
        
        # Extract all adjacent pairs from words
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        
        return dict(pairs)
    
    def train(self, texts: List[str]) -> None:
        """
        Train BPE tokenizer on corpus
        
        Args:
            texts: Training corpus
        """
        print("🔧 Training BPE Tokenizer...")
        
        # Get word frequencies
        word_freqs = self._get_word_frequencies(texts)
        print(f"   Found {len(word_freqs)} unique words")
        
        # Initialize vocabulary with characters
        vocab = self._initialize_vocabulary(word_freqs)
        
        # Convert words to character sequences
        word_splits = {}
        for word in word_freqs:
            word_splits[word] = ' '.join(word)
        
        # Perform BPE merges until target vocabulary size
        num_merges = self.config.vocab_size - len(vocab)
        merges = []
        
        for i in range(num_merges):
            # Get all pairs and their frequencies
            pairs = self._get_pairs(word_splits)
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the best pair in all words
            new_token = best_pair[0] + best_pair[1]
            
            for word in word_splits:
                word_splits[word] = word_splits[word].replace(
                    f"{best_pair[0]} {best_pair[1]}", new_token
                )
            
            # Add new token to vocabulary
            vocab[new_token] = pairs[best_pair]
            
            # Store merge rule
            merges.append(best_pair)
            
            if (i + 1) % 1000 == 0:
                print(f"   Completed {i + 1} merges")
        
        # Create token mappings
        self.vocab = vocab
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.token_to_id = {token: i for i, token in enumerate(vocab.keys())}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        print(f"✅ Training complete! Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using trained BPE
        
        Args:
            text: Input text
            
        Returns:
            token_ids: List of token IDs
        """
        # Normalize text
        normalized = self.normalizer.normalize(text)
        
        # Split into words
        words = normalized.split()
        
        token_ids = []
        for word in words:
            # Apply BPE encoding to each word
            word_tokens = self._encode_word(word)
            
            # Convert tokens to IDs
            for token in word_tokens:
                token_id = self.token_to_id.get(token, self.token_to_id.get("[UNK]", 0))
                token_ids.append(token_id)
        
        return token_ids
    
    def _encode_word(self, word: str) -> List[str]:
        """
        Encode a single word using BPE merges
        
        Args:
            word: Word to encode
            
        Returns:
            tokens: List of subword tokens
        """
        # Start with character sequence
        tokens = list(word)
        
        # Apply merges in the order they were learned
        while len(tokens) > 1:
            # Find pairs that can be merged
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            # Find the pair that was merged earliest (if any)
            mergeable_pairs = [(pair, self.merges[pair]) for pair in pairs if pair in self.merges]
            
            if not mergeable_pairs:
                break
                
            # Apply the merge with lowest index (earliest learned)
            best_pair, _ = min(mergeable_pairs, key=lambda x: x[1])
            
            # Replace the pair with merged token
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            text: Decoded text
        """
        # Convert IDs to tokens
        tokens = [self.id_to_token.get(id, "[UNK]") for id in token_ids]
        
        # Join tokens and clean up
        text = ' '.join(tokens)
        text = re.sub(r' +', ' ', text).strip()
        
        return text


class WordPieceTokenizer:
    """
    WordPiece tokenizer implementation with likelihood-based merging
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.normalizer = TextNormalizer()
        self.vocab = set()
        self.token_to_id = {}
        self.id_to_token = {}
        
    def _calculate_pair_score(self, pair: Tuple[str, str], 
                            pair_freq: int, 
                            left_freq: int, 
                            right_freq: int) -> float:
        """
        Calculate WordPiece likelihood score for a pair
        
        Args:
            pair: Token pair
            pair_freq: Frequency of the pair
            left_freq: Frequency of left token
            right_freq: Frequency of right token
            
        Returns:
            score: Likelihood score
        """
        # Implement WordPiece scoring formula
        if left_freq == 0 or right_freq == 0:
            return 0.0
        
        score = pair_freq / (left_freq * right_freq)
        return score
    
    def train(self, texts: List[str]) -> None:
        """
        Train WordPiece tokenizer using likelihood-based merging
        
        Args:
            texts: Training corpus
        """
        print("🔧 Training WordPiece Tokenizer...")
        
        # Get word frequencies (similar to BPE)
        word_freqs = defaultdict(int)
        for text in texts:
            normalized = self.normalizer.normalize(text)
            words = normalized.split()
            for word in words:
                if len(word) >= 1:
                    word_freqs[word] += 1
        
        word_freqs = {word: freq for word, freq in word_freqs.items() 
                     if freq >= self.config.min_frequency}
        
        # Initialize with characters and special tokens
        vocab = set()
        token_freqs = defaultdict(int)
        
        # Add special tokens
        for token in self.config.special_tokens:
            vocab.add(token)
            token_freqs[token] = 0
        
        # Add characters
        for word, freq in word_freqs.items():
            for char in word:
                vocab.add(char)
                token_freqs[char] += freq
        
        # Convert words to character sequences
        word_splits = {word: ' '.join(word) for word in word_freqs}
        
        # Perform likelihood-based merges
        num_merges = self.config.vocab_size - len(vocab)
        
        for i in range(num_merges):
            # Get all pairs
            pairs = defaultdict(int)
            for word, split in word_splits.items():
                symbols = split.split()
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j + 1])
                    pairs[pair] += word_freqs[word]
            
            if not pairs:
                break
            
            # Calculate likelihood scores
            pair_scores = {}
            for pair, pair_freq in pairs.items():
                left_freq = token_freqs.get(pair[0], 0)
                right_freq = token_freqs.get(pair[1], 0)
                score = self._calculate_pair_score(pair, pair_freq, left_freq, right_freq)
                pair_scores[pair] = score
            
            # Select best pair
            best_pair = max(pair_scores, key=pair_scores.get)
            
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            
            # Update word splits
            for word in word_splits:
                word_splits[word] = word_splits[word].replace(
                    f"{best_pair[0]} {best_pair[1]}", new_token
                )
            
            # Add to vocabulary
            vocab.add(new_token)
            token_freqs[new_token] = pairs[best_pair]
            
            if (i + 1) % 1000 == 0:
                print(f"   Completed {i + 1}/{num_merges} merges")
        
        # Store results
        self.vocab = vocab
        sorted_tokens = sorted(vocab, key=lambda x: (
            0 if x in self.config.special_tokens else 1,
            -token_freqs[x], x
        ))
        self.token_to_id = {token: i for i, token in enumerate(sorted_tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        print(f"✅ WordPiece training complete! Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text using WordPiece longest-match-first algorithm
        
        Args:
            text: Input text
            
        Returns:
            token_ids: List of token IDs
        """
        # TODO: Implement WordPiece encoding
        # HINT: Use longest-match-first with "##" continuation markers
        
        normalized = self.normalizer.normalize(text)
        words = normalized.split()
        
        token_ids = []
        for word in words:
            word_tokens = self._encode_word_wordpiece(word)
            for token in word_tokens:
                token_id = self.token_to_id.get(token, self.token_to_id.get("[UNK]", 0))
                token_ids.append(token_id)
        
        return token_ids
    
    def _encode_word_wordpiece(self, word: str) -> List[str]:
        """
        Encode word using WordPiece longest-match algorithm
        
        Args:
            word: Word to encode
            
        Returns:
            tokens: List of subword tokens with ## markers
        """
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            # Find longest subword in vocabulary
            while start < end:
                substr = word[start:end]
                
                # Add "##" prefix for continuation tokens
                if start > 0:
                    substr = "##" + substr
                
                # Check if substring is in vocabulary
                if substr in self.vocab:
                    cur_substr = substr
                    break
                    
                end -= 1
            
            # Handle unknown subwords
            if cur_substr is None:
                return ["[UNK]"]
            
            tokens.append(cur_substr)
            start = end
        
        return tokens


class SentencePieceStyleTokenizer:
    """
    SentencePiece-style tokenizer (simplified unigram language model)
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
    def train(self, texts: List[str]) -> None:
        """
        Train SentencePiece-style tokenizer
        
        Args:
            texts: Training corpus
        """
        print("🔧 Training SentencePiece-style Tokenizer...")
        
        # Extract all possible substrings with frequencies
        all_substrings = self._extract_substrings(texts)
        print(f"   Extracted {len(all_substrings)} substrings")
        
        # Initialize with large vocabulary
        self.vocab = all_substrings
        
        # Add special tokens first
        for token in self.config.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = 1
        
        # Keep most frequent tokens up to target size (simplified pruning)
        target_size = self.config.vocab_size
        if len(self.vocab) > target_size:
            sorted_tokens = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
            self.vocab = dict(sorted_tokens[:target_size])
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab.keys())}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        print(f"✅ SentencePiece training complete! Vocabulary size: {len(self.vocab)}")
    
    def _extract_substrings(self, texts: List[str]) -> Dict[str, int]:
        """
        Extract all possible substrings with frequencies
        
        Args:
            texts: Training texts
            
        Returns:
            substrings: Dictionary of substrings and frequencies
        """
        substrings = defaultdict(int)
        
        # Extract all substrings up to max_token_length
        for text in texts:
            # No pre-tokenization - treat as character sequence
            text = unicodedata.normalize('NFC', text)
            
            for i in range(len(text)):
                for j in range(i + 1, min(i + self.config.max_token_length + 1, len(text) + 1)):
                    substr = text[i:j]
                    if len(substr.strip()) > 0:  # Skip whitespace-only
                        substrings[substr] += 1
        
        # Filter by minimum frequency
        filtered = {substr: freq for substr, freq in substrings.items() 
                   if freq >= self.config.min_frequency}
        
        return dict(substrings)


class MultilingualTokenizer:
    """
    Multilingual tokenizer with script-specific handling
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.base_tokenizer = BPETokenizer(config)
        
        # Define script-specific normalizers
        self.script_normalizers = {
            'latin': TextNormalizer(lowercase=True, handle_contractions=True),
            'cjk': TextNormalizer(normalization_form='nfc', lowercase=False, handle_contractions=False),
            'arabic': TextNormalizer(normalization_form='nfkc', lowercase=False, handle_contractions=False),
            'indic': TextNormalizer(normalization_form='nfc', lowercase=False, handle_contractions=False)
        }
    
    def detect_script(self, text: str) -> str:
        """
        Detect the primary script of the text
        
        Args:
            text: Input text
            
        Returns:
            script: Detected script category
        """
        # Implement script detection using Unicode ranges
        
        script_counts = defaultdict(int)
        
        for char in text:
            # TODO: Classify character by script
            # HINT: Use unicodedata.name() or character ranges
            if '\u4e00' <= char <= '\u9fff':  # CJK
                script_counts['cjk'] += 1
            elif '\u0600' <= char <= '\u06ff':  # Arabic
                script_counts['arabic'] += 1
            elif '\u0900' <= char <= '\u097f':  # Devanagari (example Indic)
                script_counts['indic'] += 1
            else:
                script_counts['latin'] += 1
        
        # Return most common script
        return max(script_counts, key=script_counts.get) if script_counts else 'latin'
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text with script-specific preprocessing
        
        Args:
            text: Input text
            
        Returns:
            token_ids: List of token IDs
        """
        # Detect script and apply appropriate normalization
        script = self.detect_script(text)
        normalizer = self.script_normalizers.get(script, self.script_normalizers['latin'])
        
        # Apply script-specific preprocessing
        normalized_text = normalizer.normalize(text)
        
        # Use base tokenizer for encoding
        if self.base_tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.base_tokenizer.encode(normalized_text)


class ProductionTokenizer:
    """
    Production-ready tokenizer with caching and optimization
    """
    
    def __init__(self, base_tokenizer, cache_size: int = 10000):
        self.base_tokenizer = base_tokenizer
        self.cache_size = cache_size
        
        # Initialize caches
        self.encode_cache = {}
        self.decode_cache = {}
        
    @lru_cache(maxsize=10000)
    def encode_cached(self, text: str) -> Tuple[int, ...]:
        """
        Cached encoding with LRU eviction
        
        Args:
            text: Input text
            
        Returns:
            token_ids: Tuple of token IDs (for hashing)
        """
        # Use base tokenizer and convert to tuple for caching
        return tuple(self.base_tokenizer.encode(text))
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Efficient batch encoding
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            batch_encoding: Dictionary with input_ids, attention_mask, etc.
        """
        # Implement efficient batch processing
        batch_token_ids = []
        batch_attention_masks = []
        
        # Get pad token ID
        pad_id = getattr(self.base_tokenizer, 'token_to_id', {}).get("[PAD]", 0)
        
        for text in texts:
            # Encode each text
            token_ids = list(self.encode_cached(text))
            
            # Handle max_length (truncation/padding)
            if max_length:
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                else:
                    # Pad with pad_token_id
                    token_ids.extend([pad_id] * (max_length - len(token_ids)))
            
            # Create attention mask
            attention_mask = [1 if token_id != pad_id else 0 for token_id in token_ids]
            
            batch_token_ids.append(token_ids)
            batch_attention_masks.append(attention_mask)
        
        return {
            'input_ids': batch_token_ids,
            'attention_mask': batch_attention_masks
        }


class TokenizerEvaluator:
    """
    Comprehensive tokenizer evaluation framework
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_compression_ratio(self, texts: List[str], tokenizer) -> float:
        """
        Calculate compression ratio (characters per token)
        
        Args:
            texts: Test texts
            tokenizer: Tokenizer to evaluate
            
        Returns:
            compression_ratio: Average characters per token
        """
        # Calculate total characters and tokens
        total_chars = sum(len(text) for text in texts)
        total_tokens = 0
        
        for text in texts:
            try:
                tokens = tokenizer.encode(text)
                total_tokens += len(tokens)
            except Exception:
                continue
        
        return total_chars / total_tokens if total_tokens > 0 else 0.0
    
    def calculate_unk_rate(self, texts: List[str], tokenizer) -> float:
        """
        Calculate unknown token rate
        
        Args:
            texts: Test texts
            tokenizer: Tokenizer to evaluate
            
        Returns:
            unk_rate: Percentage of unknown tokens
        """
        # Count unknown tokens
        total_tokens = 0
        unk_tokens = 0
        
        unk_id = getattr(tokenizer, 'token_to_id', {}).get("[UNK]", -1)
        
        for text in texts:
            try:
                token_ids = tokenizer.encode(text)
                total_tokens += len(token_ids)
                if unk_id != -1:
                    unk_tokens += token_ids.count(unk_id)
            except Exception:
                continue
        
        return unk_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def evaluate_tokenizer(self, tokenizer, test_texts: List[str]) -> Dict[str, float]:
        """
        Comprehensive tokenizer evaluation
        
        Args:
            tokenizer: Tokenizer to evaluate
            test_texts: Test corpus
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Calculate various metrics
        metrics['compression_ratio'] = self.calculate_compression_ratio(test_texts, tokenizer)
        metrics['unk_rate'] = self.calculate_unk_rate(test_texts, tokenizer)
        
        # Add vocabulary statistics
        vocab_size = len(getattr(tokenizer, 'vocab', {}))
        metrics['vocab_size'] = vocab_size
        
        # Calculate average token length
        total_token_chars = 0
        total_tokens = 0
        for text in test_texts:
            try:
                token_ids = tokenizer.encode(text)
                total_tokens += len(token_ids)
                # Approximate token character count
                total_token_chars += len(text)
            except Exception:
                continue
        
        if total_tokens > 0:
            metrics['avg_token_length'] = total_token_chars / total_tokens
        else:
            metrics['avg_token_length'] = 0.0
        
        return metrics


def benchmark_tokenizers():
    """
    Benchmark different tokenizer implementations
    """
    print("🏁 Benchmarking Tokenizer Performance")
    print("=" * 50)
    
    # Create test corpus
    test_texts = [
        "Hello world! This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are fascinating fields.",
        "Don't you think tokenization is important for NLP?",
        "The price is $29.99 for the new model.",
        "Email me at user@example.com for more information."
    ]
    
    # Test different tokenizers
    config = TokenizerConfig(vocab_size=1000)
    
    tokenizers = {
        'BPE': BPETokenizer(config),
        'WordPiece': WordPieceTokenizer(config),
        'SentencePiece': SentencePieceStyleTokenizer(config)
    }
    
    # Train and evaluate each tokenizer
    evaluator = TokenizerEvaluator()
    
    for name, tokenizer in tokenizers.items():
        print(f"\n🔧 Training {name} tokenizer...")
        
        try:
            # Train tokenizer
            start_time = time.time()
            tokenizer.train(test_texts)
            training_time = time.time() - start_time
            
            # Evaluate performance
            metrics = evaluator.evaluate_tokenizer(tokenizer, test_texts)
            
            print(f"✅ {name} Results:")
            print(f"   Training time: {training_time:.2f}s")
            print(f"   Compression ratio: {metrics.get('compression_ratio', 0):.2f}")
            print(f"   UNK rate: {metrics.get('unk_rate', 0):.2%}")
            
        except Exception as e:
            print(f"❌ {name} training failed: {e}")


def test_multilingual_tokenization():
    """
    Test multilingual tokenization capabilities
    """
    print("🌍 Testing Multilingual Tokenization")
    print("=" * 50)
    
    # Create multilingual test cases
    multilingual_texts = {
        'english': "Hello, how are you today?",
        'french': "Bonjour, comment allez-vous?",
        'spanish': "Hola, ¿cómo estás?",
        'german': "Guten Tag, wie geht es Ihnen?"
    }
    
    # Test script detection and tokenization
    config = TokenizerConfig(vocab_size=5000)
    multilingual_tokenizer = MultilingualTokenizer(config)
    
    for language, text in multilingual_texts.items():
        print(f"\n🔤 Testing {language}: '{text}'")
        
        # Detect script
        script = multilingual_tokenizer.detect_script(text)
        print(f"   Detected script: {script}")
        
        # Note: Tokenization requires training first
        print(f"   Text length: {len(text)} characters")


def main():
    """
    Main function to run all tokenization exercises
    """
    print("🎯 Day 43: Tokenization Strategies - Exercise")
    print("=" * 60)
    
    # Test basic BPE implementation
    print("\n1️⃣ Testing BPE Tokenizer")
    print("-" * 30)
    
    config = TokenizerConfig(vocab_size=100)
    bpe_tokenizer = BPETokenizer(config)
    
    # Simple test corpus
    train_texts = [
        "hello world",
        "hello there", 
        "world peace",
        "peace and love"
    ]
    
    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    bpe_tokenizer.train(train_texts)
    
    # Test encoding
    test_text = "hello world peace"
    print(f"Encoding: '{test_text}'")
    token_ids = bpe_tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Test decoding
    decoded = bpe_tokenizer.decode(token_ids)
    print(f"Decoded: '{decoded}'")
    
    # Test WordPiece tokenizer
    print("\n2️⃣ Testing WordPiece Tokenizer")
    print("-" * 30)
    
    wordpiece_tokenizer = WordPieceTokenizer(config)
    try:
        wordpiece_tokenizer.train(train_texts)
        wp_tokens = wordpiece_tokenizer.encode(test_text)
        print(f"WordPiece tokens: {wp_tokens}")
    except Exception as e:
        print(f"WordPiece training: {e}")
    
    # Test multilingual capabilities
    print("\n3️⃣ Testing Multilingual Tokenization")
    print("-" * 30)
    test_multilingual_tokenization()
    
    # Test production optimizations
    print("\n4️⃣ Testing Production Optimizations")
    print("-" * 30)
    
    production_tokenizer = ProductionTokenizer(bpe_tokenizer)
    
    # Test batch encoding
    batch_texts = ["hello world", "machine learning", "artificial intelligence"]
    batch_result = production_tokenizer.encode_batch(batch_texts, max_length=10)
    print(f"Batch encoding shape: {len(batch_result['input_ids'])} x {len(batch_result['input_ids'][0])}")
    
    # Run comprehensive benchmarks
    print("\n5️⃣ Running Performance Benchmarks")
    print("-" * 30)
    benchmark_tokenizers()
    
    print("\n✅ All tokenization exercises completed!")
    print("\nKey Insights:")
    print("- BPE uses frequency-based merging for subword creation")
    print("- WordPiece uses likelihood-based scoring for better linguistic awareness")
    print("- Multilingual tokenization requires script-specific preprocessing")
    print("- Production systems need caching and batch processing for efficiency")
    print("- Evaluation metrics help compare tokenizer effectiveness")


if __name__ == "__main__":
    main()
