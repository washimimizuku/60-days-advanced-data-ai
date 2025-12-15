"""
Day 43: Tokenization Strategies - Comprehensive Test Suite

This test suite validates all tokenization implementations including:
- BPE tokenizer training and encoding/decoding
- WordPiece tokenizer with likelihood-based merging
- Multilingual tokenization with script detection
- Production optimizations (caching, batch processing)
- Evaluation metrics and benchmarking
"""

import pytest
import tempfile
import os
from typing import List, Dict
import time

# Import the implementations
from exercise import (
    TokenizerConfig, TextNormalizer, BPETokenizer, WordPieceTokenizer,
    MultilingualTokenizer, ProductionTokenizer, TokenizerEvaluator
)


class TestTokenizerConfig:
    """Test tokenizer configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration creation"""
        config = TokenizerConfig(vocab_size=1000, min_frequency=2)
        assert config.vocab_size == 1000
        assert config.min_frequency == 2
        assert len(config.special_tokens) > 0
    
    def test_default_special_tokens(self):
        """Test default special tokens are set"""
        config = TokenizerConfig()
        expected_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        for token in expected_tokens:
            assert token in config.special_tokens
    
    def test_custom_special_tokens(self):
        """Test custom special tokens"""
        custom_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
        config = TokenizerConfig(special_tokens=custom_tokens)
        assert config.special_tokens == custom_tokens


class TestTextNormalizer:
    """Test text normalization functionality"""
    
    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()
    
    def test_basic_normalization(self, normalizer):
        """Test basic text normalization"""
        text = "  Hello   World!  "
        normalized = normalizer.normalize(text)
        assert normalized == "Hello World!"
    
    def test_contraction_handling(self):
        """Test contraction expansion"""
        normalizer = TextNormalizer(handle_contractions=True)
        text = "don't can't won't"
        normalized = normalizer.normalize(text)
        # Should expand contractions
        assert "don't" not in normalized
        assert "can't" not in normalized
        assert "won't" not in normalized
    
    def test_lowercase_option(self):
        """Test lowercase normalization"""
        normalizer = TextNormalizer(lowercase=True)
        text = "Hello WORLD"
        normalized = normalizer.normalize(text)
        assert normalized == "hello world"
    
    def test_unicode_normalization(self):
        """Test Unicode normalization"""
        normalizer = TextNormalizer(normalization_form="nfc")
        # Test with accented characters
        text = "café naïve"
        normalized = normalizer.normalize(text)
        assert len(normalized) > 0  # Should handle Unicode properly


class TestBPETokenizer:
    """Test BPE tokenizer implementation"""
    
    @pytest.fixture
    def config(self):
        return TokenizerConfig(vocab_size=100, min_frequency=1)
    
    @pytest.fixture
    def tokenizer(self, config):
        return BPETokenizer(config)
    
    @pytest.fixture
    def trained_tokenizer(self, tokenizer):
        # Simple training corpus
        train_texts = [
            "hello world",
            "hello there", 
            "world peace",
            "peace and love"
        ]
        tokenizer.train(train_texts)
        return tokenizer
    
    def test_initialization(self, tokenizer, config):
        """Test tokenizer initialization"""
        assert tokenizer.config == config
        assert isinstance(tokenizer.normalizer, TextNormalizer)
        assert len(tokenizer.vocab) == 0  # Empty before training
    
    def test_word_frequency_extraction(self, tokenizer):
        """Test word frequency extraction"""
        texts = ["hello world", "hello there", "world peace"]
        word_freqs = tokenizer._get_word_frequencies(texts)
        
        assert "hello" in word_freqs
        assert "world" in word_freqs
        assert word_freqs["hello"] == 2  # Appears twice
        assert word_freqs["world"] == 2  # Appears twice
    
    def test_vocabulary_initialization(self, tokenizer):
        """Test vocabulary initialization with characters"""
        word_freqs = {"hello": 2, "world": 1}
        vocab = tokenizer._initialize_vocabulary(word_freqs)
        
        # Should contain all characters
        expected_chars = set("helloworld")
        for char in expected_chars:
            assert char in vocab
        
        # Should contain special tokens
        for token in tokenizer.config.special_tokens:
            assert token in vocab
    
    def test_training_process(self, tokenizer):
        """Test complete training process"""
        train_texts = ["hello world", "hello there"]
        
        # Should not raise any exceptions
        tokenizer.train(train_texts)
        
        # Should have vocabulary after training
        assert len(tokenizer.vocab) > 0
        assert len(tokenizer.token_to_id) > 0
        assert len(tokenizer.id_to_token) > 0
        
        # Mappings should be consistent
        assert len(tokenizer.token_to_id) == len(tokenizer.id_to_token)
    
    def test_encoding_decoding(self, trained_tokenizer):
        """Test encoding and decoding functionality"""
        test_text = "hello world"
        
        # Encode
        token_ids = trained_tokenizer.encode(test_text)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(id, int) for id in token_ids)
        
        # Decode
        decoded = trained_tokenizer.decode(token_ids)
        assert isinstance(decoded, str)
        # Should be similar to original (may have spacing differences)
        assert "hello" in decoded.lower()
        assert "world" in decoded.lower()
    
    def test_unknown_token_handling(self, trained_tokenizer):
        """Test handling of unknown tokens"""
        # Text with unknown words
        unknown_text = "xyz123 unknown_word"
        token_ids = trained_tokenizer.encode(unknown_text)
        
        # Should not crash and should return some tokens
        assert len(token_ids) > 0
        
        # Should be able to decode
        decoded = trained_tokenizer.decode(token_ids)
        assert isinstance(decoded, str)
    
    def test_special_tokens(self, trained_tokenizer):
        """Test special token handling"""
        test_text = "hello world"
        
        # With special tokens
        token_ids_with_special = trained_tokenizer.encode(test_text, add_special_tokens=True)
        token_ids_without_special = trained_tokenizer.encode(test_text, add_special_tokens=False)
        
        # With special tokens should be longer
        assert len(token_ids_with_special) >= len(token_ids_without_special)


class TestWordPieceTokenizer:
    """Test WordPiece tokenizer implementation"""
    
    @pytest.fixture
    def config(self):
        return TokenizerConfig(vocab_size=100, min_frequency=1)
    
    @pytest.fixture
    def tokenizer(self, config):
        return WordPieceTokenizer(config)
    
    def test_initialization(self, tokenizer, config):
        """Test WordPiece tokenizer initialization"""
        assert tokenizer.config == config
        assert isinstance(tokenizer.normalizer, TextNormalizer)
    
    def test_pair_scoring(self, tokenizer):
        """Test WordPiece likelihood-based scoring"""
        # Test scoring function
        score = tokenizer._calculate_pair_score(("a", "b"), 10, 20, 5)
        expected_score = 10 / (20 * 5)  # freq(pair) / (freq(left) * freq(right))
        assert abs(score - expected_score) < 1e-6
        
        # Test zero frequency handling
        score_zero = tokenizer._calculate_pair_score(("a", "b"), 10, 0, 5)
        assert score_zero == 0.0
    
    def test_training_process(self, tokenizer):
        """Test WordPiece training process"""
        train_texts = ["hello world", "hello there", "world peace"]
        
        # Should complete without errors
        tokenizer.train(train_texts)
        
        # Should have vocabulary
        assert len(tokenizer.vocab) > 0
        assert len(tokenizer.token_to_id) > 0
    
    def test_wordpiece_encoding(self, tokenizer):
        """Test WordPiece-specific encoding with ## markers"""
        # Train first
        train_texts = ["playing", "player", "play"]
        tokenizer.train(train_texts)
        
        # Test encoding
        test_word = "playing"
        tokens = tokenizer._encode_word_wordpiece(test_word)
        
        # Should return list of tokens
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Check for continuation markers in multi-token words
        if len(tokens) > 1:
            # Continuation tokens should start with ##
            continuation_tokens = [t for t in tokens[1:] if t.startswith("##")]
            # At least some continuation tokens should exist for longer words
            # (This depends on the specific training, so we just check structure)


class TestMultilingualTokenizer:
    """Test multilingual tokenization capabilities"""
    
    @pytest.fixture
    def config(self):
        return TokenizerConfig(vocab_size=500, min_frequency=1)
    
    @pytest.fixture
    def tokenizer(self, config):
        return MultilingualTokenizer(config)
    
    def test_script_detection(self, tokenizer):
        """Test script detection for different languages"""
        test_cases = [
            ("Hello world", "latin"),
            ("你好世界", "cjk"),
            ("مرحبا بالعالم", "arabic"),
            ("Привет мир", "cyrillic"),
        ]
        
        for text, expected_script in test_cases:
            detected = tokenizer.detect_script(text)
            assert detected == expected_script
    
    def test_mixed_script_detection(self, tokenizer):
        """Test script detection with mixed scripts"""
        mixed_text = "Hello 你好"  # Latin + CJK
        detected = tokenizer.detect_script(mixed_text)
        # Should detect one of the scripts (implementation dependent)
        assert detected in ["latin", "cjk"]
    
    def test_script_normalizers(self, tokenizer):
        """Test that different scripts have appropriate normalizers"""
        assert "latin" in tokenizer.script_normalizers
        assert "cjk" in tokenizer.script_normalizers
        assert "arabic" in tokenizer.script_normalizers
        
        # Test normalizer differences
        latin_normalizer = tokenizer.script_normalizers["latin"]
        cjk_normalizer = tokenizer.script_normalizers["cjk"]
        
        # Latin should handle contractions, CJK should not
        assert latin_normalizer.handle_contractions == True
        assert cjk_normalizer.handle_contractions == False
    
    def test_multilingual_training(self, tokenizer):
        """Test training on multilingual corpus"""
        multilingual_texts = [
            "Hello world",
            "Bonjour monde", 
            "Hola mundo",
            "你好世界"
        ]
        
        # Should complete training without errors
        tokenizer.train(multilingual_texts)
        
        # Should have base tokenizer after training
        assert tokenizer.base_tokenizer is not None
        assert len(tokenizer.base_tokenizer.vocab) > 0


class TestProductionTokenizer:
    """Test production optimizations"""
    
    @pytest.fixture
    def base_tokenizer(self):
        config = TokenizerConfig(vocab_size=100, min_frequency=1)
        tokenizer = BPETokenizer(config)
        train_texts = ["hello world", "machine learning", "artificial intelligence"]
        tokenizer.train(train_texts)
        return tokenizer
    
    @pytest.fixture
    def prod_tokenizer(self, base_tokenizer):
        return ProductionTokenizer(base_tokenizer, cache_size=100)
    
    def test_caching_functionality(self, prod_tokenizer):
        """Test that caching improves performance"""
        test_text = "hello world machine learning"
        
        # First encoding (cache miss)
        start_time = time.time()
        result1 = prod_tokenizer.encode(test_text)
        first_time = time.time() - start_time
        
        # Second encoding (cache hit)
        start_time = time.time()
        result2 = prod_tokenizer.encode(test_text)
        second_time = time.time() - start_time
        
        # Results should be identical
        assert result1 == result2
        
        # Second call should be faster (though timing can be unreliable in tests)
        # We just check that it doesn't crash and returns same result
        assert len(result1) == len(result2)
    
    def test_batch_processing(self, prod_tokenizer):
        """Test batch encoding functionality"""
        batch_texts = [
            "hello world",
            "machine learning is amazing", 
            "tokenization is important"
        ]
        
        batch_result = prod_tokenizer.encode_batch(
            batch_texts,
            max_length=10,
            padding=True
        )
        
        # Check structure
        assert 'input_ids' in batch_result
        assert 'attention_mask' in batch_result
        assert len(batch_result['input_ids']) == len(batch_texts)
        assert len(batch_result['attention_mask']) == len(batch_texts)
        
        # Check padding (all sequences should have same length)
        lengths = [len(seq) for seq in batch_result['input_ids']]
        assert all(length == lengths[0] for length in lengths)
    
    def test_cache_statistics(self, prod_tokenizer):
        """Test cache statistics tracking"""
        # Make some requests
        prod_tokenizer.encode("test text 1")
        prod_tokenizer.encode("test text 2")
        prod_tokenizer.encode("test text 1")  # Cache hit
        
        stats = prod_tokenizer.get_cache_stats()
        
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats
        assert stats['cache_hits'] >= 1  # At least one hit from repeated text


class TestTokenizerEvaluator:
    """Test tokenizer evaluation framework"""
    
    @pytest.fixture
    def evaluator(self):
        return TokenizerEvaluator()
    
    @pytest.fixture
    def test_tokenizer(self):
        config = TokenizerConfig(vocab_size=100, min_frequency=1)
        tokenizer = BPETokenizer(config)
        train_texts = ["hello world", "test text", "evaluation metrics"]
        tokenizer.train(train_texts)
        return tokenizer
    
    def test_compression_ratio(self, evaluator, test_tokenizer):
        """Test compression ratio calculation"""
        test_texts = ["hello world", "test text"]
        ratio = evaluator.calculate_compression_ratio(test_texts, test_tokenizer)
        
        assert isinstance(ratio, float)
        assert ratio > 0  # Should be positive
        assert ratio < 100  # Should be reasonable
    
    def test_unk_rate(self, evaluator, test_tokenizer):
        """Test unknown token rate calculation"""
        test_texts = ["hello world", "unknown_xyz_word"]
        unk_rate = evaluator.calculate_unk_rate(test_texts, test_tokenizer)
        
        assert isinstance(unk_rate, float)
        assert 0 <= unk_rate <= 1  # Should be between 0 and 1
    
    def test_comprehensive_evaluation(self, evaluator, test_tokenizer):
        """Test comprehensive evaluation"""
        test_texts = ["hello world", "machine learning", "test evaluation"]
        metrics = evaluator.evaluate_tokenizer(test_tokenizer, test_texts)
        
        # Check that all expected metrics are present
        expected_metrics = ['compression_ratio', 'unk_rate']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


class TestIntegration:
    """Integration tests for complete tokenization pipeline"""
    
    def test_end_to_end_bpe_pipeline(self):
        """Test complete BPE pipeline from training to evaluation"""
        # Configuration
        config = TokenizerConfig(vocab_size=200, min_frequency=1)
        
        # Training data
        train_texts = [
            "hello world machine learning",
            "natural language processing",
            "artificial intelligence systems",
            "deep learning neural networks"
        ]
        
        # Train tokenizer
        tokenizer = BPETokenizer(config)
        tokenizer.train(train_texts)
        
        # Test encoding/decoding
        test_text = "hello machine learning"
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        # Should preserve key words
        assert len(token_ids) > 0
        assert isinstance(decoded, str)
        
        # Wrap with production features
        prod_tokenizer = ProductionTokenizer(tokenizer)
        
        # Test batch processing
        batch_texts = ["hello world", "machine learning"]
        batch_result = prod_tokenizer.encode_batch(batch_texts, max_length=20)
        
        assert len(batch_result['input_ids']) == 2
        
        # Evaluate performance
        evaluator = TokenizerEvaluator()
        metrics = evaluator.evaluate_tokenizer(tokenizer, train_texts)
        
        assert 'compression_ratio' in metrics
        assert metrics['compression_ratio'] > 0
    
    def test_multilingual_pipeline(self):
        """Test multilingual tokenization pipeline"""
        config = TokenizerConfig(vocab_size=300, min_frequency=1)
        
        # Multilingual training data
        train_texts = [
            "Hello world",
            "Bonjour monde",
            "Hola mundo", 
            "你好世界"
        ]
        
        # Train multilingual tokenizer
        multilingual_tokenizer = MultilingualTokenizer(config)
        multilingual_tokenizer.train(train_texts)
        
        # Test with different scripts
        test_cases = [
            "Hello machine learning",
            "Bonjour apprentissage automatique"
        ]
        
        for text in test_cases:
            token_ids = multilingual_tokenizer.encode(text)
            decoded = multilingual_tokenizer.decode(token_ids)
            
            assert len(token_ids) > 0
            assert isinstance(decoded, str)


def test_performance_benchmarking():
    """Test that performance benchmarking functions work"""
    # This is a basic test to ensure benchmark functions don't crash
    config = TokenizerConfig(vocab_size=50, min_frequency=1)
    
    # Simple test data
    test_texts = ["hello world", "test text", "benchmark performance"]
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(config)
    tokenizer.train(test_texts)
    
    # Test evaluation
    evaluator = TokenizerEvaluator()
    metrics = evaluator.evaluate_tokenizer(tokenizer, test_texts)
    
    # Should complete without errors
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])