# Day 53 Quiz: RAG Evaluation with RAGAS - Metrics, Quality Assessment & Optimization

## Instructions
Answer all questions. Each question has one correct answer.

---

1. **What does the RAGAS faithfulness metric measure?**
   - A) How relevant the retrieved contexts are to the question
   - B) How factually accurate the generated answer is based on the retrieved context
   - C) How similar the generated answer is to the original question
   - D) How diverse the retrieved contexts are

2. **In RAGAS, how is Answer Relevancy typically calculated?**
   - A) By comparing the answer directly to the question using keyword matching
   - B) By generating questions from the answer and measuring similarity to the original question
   - C) By counting the number of words that overlap between question and answer
   - D) By measuring the length of the generated answer

3. **What is Context Precision in RAG evaluation?**
   - A) The accuracy of the generated answer
   - B) The speed of context retrieval
   - C) The proportion of relevant contexts among all retrieved contexts
   - D) The total number of contexts retrieved

4. **Which metric evaluates whether all relevant information needed to answer the question was retrieved?**
   - A) Faithfulness
   - B) Answer Relevancy
   - C) Context Precision
   - D) Context Recall

5. **What is the main advantage of RAGAS over traditional evaluation methods?**
   - A) It requires ground truth answers for all evaluations
   - B) It provides reference-free evaluation without requiring ground truth answers
   - C) It only works with specific language models
   - D) It focuses solely on retrieval quality, ignoring generation

6. **In a production RAG system, what should trigger a quality alert?**
   - A) High response times only
   - B) Low faithfulness or relevancy scores below defined thresholds
   - C) Increased number of user queries
   - D) Changes in document corpus size

7. **What is the purpose of A/B testing in RAG evaluation?**
   - A) To test different user interfaces
   - B) To compare the performance of two different RAG system configurations
   - C) To evaluate individual document quality
   - D) To measure system uptime

8. **Which approach is most effective for creating evaluation datasets when ground truth is limited?**
   - A) Manual annotation of all possible questions
   - B) Using only existing FAQ data
   - C) Synthetic dataset generation using LLMs to create question-answer pairs from documents
   - D) Random sampling of user queries

9. **What does inter-annotator agreement measure in human evaluation of RAG systems?**
   - A) How fast human evaluators can assess responses
   - B) The consistency of quality judgments between different human evaluators
   - C) The number of evaluators needed for assessment
   - D) The cost of human evaluation

10. **Which combination of metrics provides the most comprehensive RAG system evaluation?**
    - A) Only faithfulness and answer relevancy
    - B) Faithfulness, answer relevancy, context precision, context recall, and custom domain-specific metrics
    - C) Response time and token usage only
    - D) User satisfaction scores alone

---

## Answer Key

**1. B** - RAGAS faithfulness measures how factually accurate the generated answer is based on the retrieved context. It calculates the proportion of claims in the answer that are supported by the provided contexts, ensuring the response doesn't hallucinate information.

**2. B** - Answer Relevancy in RAGAS is calculated by generating questions from the answer using an LLM and then measuring the semantic similarity between these generated questions and the original question. This approach evaluates how well the answer addresses the specific question asked.

**3. C** - Context Precision measures the proportion of relevant contexts among all retrieved contexts. It evaluates the quality of the retrieval system by determining how many of the retrieved documents actually contain information relevant to answering the question.

**4. D** - Context Recall evaluates whether all relevant information needed to answer the question was retrieved. It measures the proportion of relevant contexts that were successfully retrieved compared to all contexts that should have been retrieved for a complete answer.

**5. B** - The main advantage of RAGAS is that it provides reference-free evaluation without requiring ground truth answers. This makes it practical for production systems where creating comprehensive ground truth datasets is expensive or impossible.

**6. B** - Quality alerts should be triggered when faithfulness or relevancy scores fall below defined thresholds. These metrics directly indicate when the RAG system is producing inaccurate or irrelevant responses, which impacts user experience and system reliability.

**7. B** - A/B testing in RAG evaluation compares the performance of two different RAG system configurations (different retrieval methods, models, or parameters) to determine which performs better according to evaluation metrics and user satisfaction.

**8. C** - Synthetic dataset generation using LLMs to create question-answer pairs from documents is most effective when ground truth is limited. This approach can scale to create large evaluation datasets while maintaining quality and coverage of the document corpus.

**9. B** - Inter-annotator agreement measures the consistency of quality judgments between different human evaluators. High agreement indicates that the evaluation criteria are clear and the quality assessments are reliable and reproducible.

**10. B** - The most comprehensive RAG evaluation combines faithfulness, answer relevancy, context precision, context recall, and custom domain-specific metrics. This multi-dimensional approach captures different aspects of system performance including accuracy, relevance, retrieval quality, and domain-specific requirements.
