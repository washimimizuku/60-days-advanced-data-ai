# Day 43: Tokenization Strategies - Setup Guide

## 📋 Overview

This setup guide will help you prepare your environment for implementing and experimenting with various tokenization strategies including BPE, WordPiece, SentencePiece-style tokenizers, and multilingual text processing.

## 🔧 Prerequisites

- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM (8GB+ recommended for large vocabularies)
- **Storage**: 2GB+ free space for tokenizer models and test data

## 📦 Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv tokenization_env

# Activate environment
# On macOS/Linux:
source tokenization_env/bin/activate
# On Windows:
tokenization_env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Optional: Install additional tokenization libraries for comparison
# pip install tokenizers sentencepiece transformers
```

### 3. Verify Installation

```bash
# Run verification script
python -c "
import unicodedata
import re
import numpy as np
import matplotlib.pyplot as plt
print('✅ All dependencies installed successfully!')
print(f'Python version: {__import__('sys').version}')
print(f'NumPy version: {np.__version__}')
"
```

## 🧠 Tokenization Fundamentals

### Core Concepts

#### 1. **Tokenization Hierarchy**
```
Raw Text → Normalization → Pre-tokenization → Subword Segmentation → Token IDs
```

#### 2. **Vocabulary Trade-offs**
- **Small Vocabulary** (1K-5K): More subword pieces, longer sequences
- **Medium Vocabulary** (16K-32K): Balanced efficiency and coverage
- **Large Vocabulary** (50K+): Fewer pieces per word, larger model size

#### 3. **Algorithm Comparison**

| Algorithm | Merging Criterion | Strengths | Weaknesses |
|-----------|------------------|-----------|------------|
| **BPE** | Frequency-based | Simple, deterministic | Greedy, no linguistic knowledge |
| **WordPiece** | Likelihood-based | Better linguistic awareness | More complex scoring |
| **SentencePiece** | Language-agnostic | No pre-tokenization assumptions | Requires more training data |

### Mathematical Foundations

#### BPE Algorithm
```
1. Initialize: vocab = characters(corpus)
2. While |vocab| < target_size:
   3. pairs = count_adjacent_pairs(corpus)
   4. best_pair = argmax(frequency(pairs))
   5. vocab.add(merge(best_pair))
   6. corpus = apply_merge(corpus, best_pair)
```

#### WordPiece Scoring
```
score(left, right) = freq(left + right) / (freq(left) × freq(right))
```

#### Compression Ratio
```
compression_ratio = total_characters / total_tokens
```

## 🎯 Learning Objectives

By completing this day, you will:

1. **Master Subword Algorithms**
   - Implement BPE from scratch with proper merge rule learning
   - Build WordPiece tokenizer with likelihood-based scoring
   - Create SentencePiece-style tokenizer with unigram model

2. **Handle Multilingual Text**
   - Detect scripts using Unicode ranges
   - Apply script-specific normalization
   - Build unified multilingual vocabularies

3. **Optimize for Production**
   - Implement efficient caching mechanisms
   - Build batch processing capabilities
   - Create comprehensive evaluation frameworks

4. **Evaluate Tokenizer Quality**
   - Measure compression ratios and vocabulary utilization
   - Calculate unknown token rates and fertility scores
   - Benchmark encoding/decoding performance

## 🚀 Quick Start

### 1. Test Basic BPE Implementation

```python
from exercise import TokenizerConfig, BPETokenizer

# Create configuration
config = TokenizerConfig(vocab_size=1000, min_frequency=2)

# Initialize tokenizer
tokenizer = BPETokenizer(config)

# Train on sample corpus
train_texts = [
    "hello world machine learning",
    "natural language processing", 
    "artificial intelligence systems"
]

tokenizer.train(train_texts)

# Test encoding/decoding
test_text = "hello machine learning"
token_ids = tokenizer.encode(test_text)
decoded = tokenizer.decode(token_ids)

print(f"Original: {test_text}")
print(f"Token IDs: {token_ids}")
print(f"Decoded: {decoded}")
```

### 2. Test Multilingual Capabilities

```python
from exercise import MultilingualTokenizer

# Create multilingual tokenizer
multilingual_config = TokenizerConfig(vocab_size=2000)
multilingual_tokenizer = MultilingualTokenizer(multilingual_config)

# Test script detection
texts = [
    "Hello world",           # Latin
    "你好世界",              # CJK
    "مرحبا بالعالم",         # Arabic
    "Привет мир"            # Cyrillic
]

for text in texts:
    script = multilingual_tokenizer.detect_script(text)
    print(f"'{text}' → Script: {script}")
```

### 3. Test Production Features

```python
from exercise import ProductionTokenizer

# Wrap tokenizer with production features
prod_tokenizer = ProductionTokenizer(tokenizer, cache_size=1000)

# Test batch processing
batch_texts = [
    "hello world",
    "machine learning is amazing",
    "tokenization strategies"
]

batch_result = prod_tokenizer.encode_batch(
    batch_texts,
    max_length=20,
    padding=True
)

print(f"Batch shape: {len(batch_result['input_ids'])} x {len(batch_result['input_ids'][0])}")
print(f"Cache stats: {prod_tokenizer.get_cache_stats()}")
```

## 🔍 Implementation Details

### 1. **Text Normalization Pipeline**

```python
class TextNormalizer:
    def normalize(self, text: str) -> str:
        # 1. Unicode normalization (NFC/NFD/NFKC/NFKD)
        text = unicodedata.normalize(self.form, text)
        
        # 2. Contraction expansion
        if self.handle_contractions:
            text = self.expand_contractions(text)
        
        # 3. Case normalization
        if self.lowercase:
            text = text.lower()
        
        # 4. Whitespace cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
```

### 2. **BPE Training Process**

```python
def train_bpe(corpus, vocab_size):
    # Initialize with characters
    vocab = extract_characters(corpus)
    word_splits = {word: ' '.join(word) for word in corpus}
    
    # Iterative merging
    while len(vocab) < vocab_size:
        # Count all adjacent pairs
        pairs = count_pairs(word_splits)
        if not pairs:
            break
            
        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        
        # Merge in all words
        new_token = best_pair[0] + best_pair[1]
        for word in word_splits:
            word_splits[word] = word_splits[word].replace(
                f"{best_pair[0]} {best_pair[1]}", new_token
            )
        
        vocab.add(new_token)
    
    return vocab, merge_rules
```

### 3. **Script Detection Logic**

```python
def detect_script(text):
    script_counts = defaultdict(int)
    
    for char in text:
        code_point = ord(char)
        
        if 0x4e00 <= code_point <= 0x9fff:      # CJK
            script_counts['cjk'] += 1
        elif 0x0600 <= code_point <= 0x06ff:    # Arabic
            script_counts['arabic'] += 1
        elif 0x0400 <= code_point <= 0x04ff:    # Cyrillic
            script_counts['cyrillic'] += 1
        else:                                    # Default to Latin
            script_counts['latin'] += 1
    
    return max(script_counts, key=script_counts.get)
```

## 📊 Evaluation Metrics

### 1. **Compression Efficiency**

```python
def calculate_compression_ratio(texts, tokenizer):
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
    return total_chars / total_tokens
```

### 2. **Vocabulary Coverage**

```python
def calculate_unk_rate(texts, tokenizer):
    total_tokens = 0
    unk_tokens = 0
    unk_id = tokenizer.token_to_id.get("[UNK]", -1)
    
    for text in texts:
        token_ids = tokenizer.encode(text)
        total_tokens += len(token_ids)
        unk_tokens += token_ids.count(unk_id)
    
    return unk_tokens / total_tokens
```

### 3. **Performance Benchmarking**

```python
def benchmark_encoding_speed(texts, tokenizer, num_runs=3):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        for text in texts:
            tokenizer.encode(text)
        times.append(time.time() - start_time)
    
    avg_time = sum(times) / len(times)
    return len(texts) / avg_time  # texts per second
```

## 🧪 Testing Your Implementation

### Run Unit Tests

```bash
# Run all tests
python -m pytest test_tokenization.py -v

# Run specific test category
python -m pytest test_tokenization.py::TestBPETokenizer -v

# Run with coverage
python -m pytest test_tokenization.py --cov=exercise --cov-report=html
```

### Performance Testing

```bash
# Run performance benchmarks
python exercise.py

# This will test:
# - Training speed for different algorithms
# - Encoding/decoding performance
# - Memory usage analysis
# - Caching effectiveness
```

### Manual Testing

```python
# Test round-trip consistency
def test_round_trip(tokenizer, text):
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    print(f"Original: '{text}'")
    print(f"Decoded:  '{decoded}'")
    print(f"Match: {text.strip() == decoded.strip()}")

# Test vocabulary utilization
def test_vocab_usage(tokenizer, texts):
    used_tokens = set()
    for text in texts:
        token_ids = tokenizer.encode(text)
        used_tokens.update(token_ids)
    
    utilization = len(used_tokens) / len(tokenizer.vocab)
    print(f"Vocabulary utilization: {utilization:.2%}")
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Errors During Training**
   ```python
   # Solution: Reduce vocabulary size or use incremental training
   config = TokenizerConfig(vocab_size=16000)  # Instead of 50000
   ```

2. **Slow Encoding Performance**
   ```python
   # Solution: Use caching and batch processing
   prod_tokenizer = ProductionTokenizer(base_tokenizer, cache_size=10000)
   ```

3. **Poor Multilingual Performance**
   ```python
   # Solution: Ensure balanced training data
   def balance_languages(texts, max_per_language=1000):
       # Implement language balancing logic
       pass
   ```

4. **High Unknown Token Rate**
   ```python
   # Solution: Lower minimum frequency or increase vocabulary size
   config = TokenizerConfig(vocab_size=32000, min_frequency=1)
   ```

### Performance Tips

1. **Optimize Training Data**: Remove duplicates, filter by quality
2. **Use Appropriate Vocabulary Size**: 16K-32K for most applications
3. **Implement Caching**: Cache frequent tokenizations
4. **Batch Processing**: Process multiple texts simultaneously
5. **Profile Memory Usage**: Monitor memory consumption during training

## 📚 Additional Resources

### Research Papers
- "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- "Japanese and Korean Voice Search" (WordPiece)
- "SentencePiece: A simple and language independent subword tokenizer"
- "Subword Regularization: Improving Neural Network Translation Models"

### Implementation References
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
- [Google SentencePiece](https://github.com/google/sentencepiece)
- [OpenAI tiktoken](https://github.com/openai/tiktoken)

### Datasets for Testing
- **Multilingual**: Common Crawl, Wikipedia dumps
- **Domain-specific**: PubMed (medical), arXiv (scientific)
- **Code**: GitHub repositories, Stack Overflow

## ✅ Success Criteria

After completing this setup and implementation, you should be able to:

- ✅ Implement BPE tokenizer from scratch with proper merge learning
- ✅ Build WordPiece tokenizer with likelihood-based scoring
- ✅ Handle multilingual text with script-specific preprocessing
- ✅ Optimize tokenizers for production with caching and batching
- ✅ Evaluate tokenizer quality using multiple metrics
- ✅ Debug tokenization issues and optimize performance
- ✅ Compare different tokenization strategies effectively

## 🎯 Next Steps

After mastering tokenization strategies:

1. **Day 44**: LLM training stages and optimization techniques
2. **Day 45**: Advanced prompt engineering with DSPy
3. **Day 46**: Prompt security and injection defense
4. **Integration**: Apply tokenization optimizations to your own models

---

**Ready to master the art of tokenization?** 🚀

Start with the basic BPE implementation and gradually work through multilingual support, production optimizations, and comprehensive evaluation. The test suite will help validate your understanding at each step!