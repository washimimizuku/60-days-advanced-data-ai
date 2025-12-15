# Day 43 Quiz: Tokenization Strategies

## Instructions
Answer all 10 questions. Each question has 4 options (a, b, c, d). Choose the best answer.

---

### 1. What is the key advantage of subword tokenization over word-level tokenization?
- a) It always produces shorter sequences than word-level tokenization
- b) It handles out-of-vocabulary words and reduces vocabulary size while maintaining semantic meaning
- c) It is computationally faster than word-level tokenization
- d) It works only with English text and requires no special handling

**Answer: b) It handles out-of-vocabulary words and reduces vocabulary size while maintaining semantic meaning**

**Explanation:** Subword tokenization solves the main problems of word-level tokenization: the out-of-vocabulary (OOV) problem and vocabulary explosion. By breaking words into meaningful subword units, it can represent any word using a fixed vocabulary of 16K-50K tokens, while still capturing morphological patterns and semantic relationships that pure character-level tokenization would lose.

---

### 2. How does WordPiece differ from BPE in its merging criterion?
- a) WordPiece uses character frequency while BPE uses word frequency
- b) WordPiece uses likelihood-based scoring while BPE uses simple frequency counting
- c) WordPiece requires pre-segmentation while BPE does not
- d) WordPiece only works with Latin scripts while BPE is universal

**Answer: b) WordPiece uses likelihood-based scoring while BPE uses simple frequency counting**

**Explanation:** The key difference is in the merging criterion: BPE merges the most frequent adjacent pair, while WordPiece merges the pair that maximizes the likelihood of the training data using the formula: score(pair) = freq(pair) / (freq(left) × freq(right)). This likelihood-based approach tends to create more linguistically meaningful subwords by favoring strongly associated pairs over merely frequent ones.

---

### 3. What is the main innovation of SentencePiece compared to BPE and WordPiece?
- a) It uses a completely different algorithm that is faster than BPE
- b) It treats text as a sequence of Unicode characters without pre-tokenization, making it truly language-agnostic
- c) It only works with character-level tokenization
- d) It requires less training data than other methods

**Answer: b) It treats text as a sequence of Unicode characters without pre-tokenization, making it truly language-agnostic**

**Explanation:** SentencePiece's key innovation is that it doesn't assume word boundaries exist (no pre-tokenization on spaces). It treats the entire text as a sequence of Unicode characters, making it work equally well for languages with clear word boundaries (English) and those without (Chinese, Japanese). It also preserves spaces as special tokens (▁), enabling perfect text reconstruction.

---

### 4. Why is the "##" prefix used in WordPiece tokenization?
- a) To indicate the start of a new word
- b) To mark continuation tokens that are not at the beginning of a word
- c) To represent special characters that cannot be tokenized
- d) To indicate tokens that appear frequently in the training data

**Answer: b) To mark continuation tokens that are not at the beginning of a word**

**Explanation:** The "##" prefix in WordPiece indicates that a token is a continuation of the previous token within the same word. For example, "playing" might be tokenized as ["play", "##ing"], where "##ing" indicates that "ing" is not a standalone word but continues "play". This helps preserve word boundaries during decoding and makes the tokenization reversible.

---

### 5. What is subword regularization and why is it useful?
- a) A method to reduce the vocabulary size by removing rare tokens
- b) A technique that uses multiple possible segmentations during training to improve model robustness
- c) A way to normalize text before tokenization
- d) A method to speed up tokenization by caching results

**Answer: b) A technique that uses multiple possible segmentations during training to improve model robustness**

**Explanation:** Subword regularization introduces multiple possible segmentations of the same text during training, acting as a form of data augmentation. This makes the model less sensitive to specific tokenization choices and more robust to tokenization errors, typos, and rare words. It's particularly useful in low-resource settings and helps improve generalization.

---

### 6. What is the primary challenge when tokenizing Chinese, Japanese, and Korean (CJK) text?
- a) These languages have too many characters to fit in a reasonable vocabulary
- b) These languages have no clear word boundaries, making pre-tokenization difficult
- c) These languages cannot be represented in Unicode
- d) These languages require character-level tokenization only

**Answer: b) These languages have no clear word boundaries, making pre-tokenization difficult**

**Explanation:** CJK languages don't use spaces to separate words, making it impossible to use simple whitespace-based pre-tokenization. This is why methods like SentencePiece, which don't rely on pre-tokenization, work better for these languages. The challenge is determining where one semantic unit ends and another begins without explicit word boundary markers.

---

### 7. In production systems, what is the most important consideration for tokenizer performance?
- a) Using the largest possible vocabulary size
- b) Implementing efficient encoding with caching and batch processing capabilities
- c) Always using character-level tokenization for consistency
- d) Avoiding any form of text normalization

**Answer: b) Implementing efficient encoding with caching and batch processing capabilities**

**Explanation:** Production tokenizers need to handle high throughput efficiently. This requires: fast lookup structures (like tries), caching of frequent tokenizations, batch processing capabilities, and optimized implementations (like Rust-based tokenizers). Memory management and the ability to process multiple texts simultaneously are crucial for real-world deployment at scale.

---

### 8. What does the compression ratio measure in tokenization evaluation?
- a) How much smaller the tokenized file is compared to the original
- b) The average number of characters per token, indicating tokenization efficiency
- c) The percentage of unknown tokens in the output
- d) The speed of the tokenization process

**Answer: b) The average number of characters per token, indicating tokenization efficiency**

**Explanation:** Compression ratio measures how efficiently a tokenizer represents text by calculating the ratio of total characters to total tokens. A higher compression ratio means fewer tokens are needed to represent the same text, which is generally better for model efficiency. However, this must be balanced with semantic meaningfulness - very high compression might lose important linguistic information.

---

### 9. What is the main benefit of using continuation markers (like "##" in WordPiece) in subword tokenization?
- a) They reduce the total vocabulary size needed
- b) They enable perfect reconstruction of the original text with proper word boundaries
- c) They make tokenization faster by reducing the number of lookups
- d) They help identify the most important tokens in a sequence

**Answer: b) They enable perfect reconstruction of the original text with proper word boundaries**

**Explanation:** Continuation markers preserve information about word boundaries during tokenization. Without them, it would be impossible to know whether adjacent tokens belong to the same word or different words during decoding. For example, ["play", "##ing"] can be correctly decoded to "playing", while ["play", "ing"] without markers might be incorrectly decoded as "play ing" (two words).

---

### 10. When extending a pre-trained tokenizer's vocabulary for a new domain, what is the most important consideration?
- a) Adding as many new tokens as possible to maximize coverage
- b) Only adding tokens that are poorly handled by the existing tokenizer and appear frequently in the domain
- c) Replacing the entire vocabulary with domain-specific terms
- d) Using only character-level tokens for the new domain

**Answer: b) Only adding tokens that are poorly handled by the existing tokenizer and appear frequently in the domain**

**Explanation:** When extending a tokenizer vocabulary, you should focus on terms that: (1) are poorly tokenized by the existing tokenizer (resulting in many subword pieces or unknown tokens), and (2) appear frequently enough in the domain to justify a dedicated token. Adding too many tokens increases model size and training time, while adding rare tokens provides little benefit. The goal is to improve tokenization efficiency for domain-specific content without bloating the vocabulary.

---

## Answer Key
1. b) It handles out-of-vocabulary words and reduces vocabulary size while maintaining semantic meaning
2. b) WordPiece uses likelihood-based scoring while BPE uses simple frequency counting
3. b) It treats text as a sequence of Unicode characters without pre-tokenization, making it truly language-agnostic
4. b) To mark continuation tokens that are not at the beginning of a word
5. b) A technique that uses multiple possible segmentations during training to improve model robustness
6. b) These languages have no clear word boundaries, making pre-tokenization difficult
7. b) Implementing efficient encoding with caching and batch processing capabilities
8. b) The average number of characters per token, indicating tokenization efficiency
9. b) They enable perfect reconstruction of the original text with proper word boundaries
10. b) Only adding tokens that are poorly handled by the existing tokenizer and appear frequently in the domain

## Scoring
- 9-10 correct: Excellent understanding of tokenization strategies
- 7-8 correct: Good grasp with minor gaps
- 5-6 correct: Basic understanding, review key concepts
- Below 5: Revisit the material and practice implementations
