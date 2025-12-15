# Day 34: A/B Testing for ML - Quiz

## Questions

### 1. What is the primary advantage of using consistent hashing for user assignment in A/B tests?
- a) It provides better statistical power
- b) It ensures users get the same variant across sessions
- c) It improves model accuracy
- d) It reduces computational overhead

**Answer: b) It ensures users get the same variant across sessions**

**Explanation:** Consistent hashing ensures that the same user always gets assigned to the same experiment variant across different sessions and interactions. This is crucial for maintaining experiment integrity and avoiding bias that could occur if users switched between variants. Statistical power, model accuracy, and computational overhead are not directly related to the assignment mechanism.

### 2. In multi-armed bandit algorithms, what does the "exploration vs exploitation" trade-off refer to?
- a) Balancing training time vs inference time
- b) Choosing between trying new variants vs using the best known variant
- c) Deciding between statistical significance vs practical significance
- d) Optimizing for accuracy vs interpretability

**Answer: b) Choosing between trying new variants vs using the best known variant**

**Explanation:** The exploration vs exploitation trade-off is fundamental to multi-armed bandits. Exploration means trying different variants (arms) to learn about their performance, while exploitation means using the variant that currently appears to be the best. Too much exploration wastes traffic on poor variants, while too much exploitation might miss discovering better variants. This trade-off is not about training/inference time, statistical concepts, or model properties.

### 3. What is the main purpose of guardrail metrics in A/B testing?
- a) To increase statistical power
- b) To prevent degradation of critical business metrics
- c) To reduce experiment duration
- d) To improve user assignment accuracy

**Answer: b) To prevent degradation of critical business metrics**

**Explanation:** Guardrail metrics are safety measures that monitor critical business metrics (like revenue, user satisfaction, or system performance) to ensure they don't degrade significantly during an experiment. If guardrails are violated, the experiment should be stopped to prevent business harm. They don't directly affect statistical power, experiment duration, or assignment accuracy - they're purely protective measures.

### 4. Which statistical test is most appropriate for analyzing binary metrics like click-through rates?
- a) T-test
- b) Chi-square test
- c) Z-test for proportions
- d) ANOVA

**Answer: c) Z-test for proportions**

**Explanation:** Z-test for proportions is specifically designed for comparing binary outcomes (success/failure rates) between groups, making it ideal for metrics like click-through rates, conversion rates, or any percentage-based metrics. T-tests are for continuous variables, chi-square tests are for independence/goodness of fit, and ANOVA is for comparing multiple group means.

### 5. What does statistical power represent in A/B testing?
- a) The probability of detecting a true effect when it exists
- b) The probability of making a Type I error
- c) The size of the effect being tested
- d) The confidence level of the test

**Answer: a) The probability of detecting a true effect when it exists**

**Explanation:** Statistical power is the probability of correctly rejecting a false null hypothesis, i.e., detecting a true effect when it actually exists. It's calculated as 1 - β (where β is the Type II error rate). Type I error probability is the significance level (α), effect size is a separate concept measuring the magnitude of difference, and confidence level is related to but distinct from power.

### 6. In the Upper Confidence Bound (UCB) algorithm, what does the confidence bonus term represent?
- a) The statistical significance of the result
- b) The uncertainty about an arm's true performance
- c) The sample size of the experiment
- d) The effect size of the treatment

**Answer: b) The uncertainty about an arm's true performance**

**Explanation:** The confidence bonus in UCB represents the uncertainty or confidence interval around an arm's estimated performance. Arms with fewer observations get larger bonuses, encouraging exploration of less-tested variants. This balances exploitation (choosing the best-performing arm) with exploration (trying uncertain arms that might be better). It's not about statistical significance, sample size directly, or effect size.

### 7. What is the primary benefit of sequential testing over fixed-sample testing?
- a) Better statistical power
- b) Ability to stop experiments early for efficacy or futility
- c) Simpler statistical analysis
- d) Lower Type I error rates

**Answer: b) Ability to stop experiments early for efficacy or futility**

**Explanation:** Sequential testing allows for interim analyses during an experiment, enabling early stopping when there's sufficient evidence of efficacy (treatment works) or futility (treatment unlikely to work). This can save time and resources compared to fixed-sample tests that run to completion regardless of interim results. It doesn't necessarily provide better power, simpler analysis, or lower error rates - it provides flexibility in experiment duration.

### 8. Why is stratified randomization important in A/B testing?
- a) To increase the sample size
- b) To ensure balanced allocation across important user segments
- c) To reduce computational complexity
- d) To improve model accuracy

**Answer: b) To ensure balanced allocation across important user segments**

**Explanation:** Stratified randomization ensures that important user characteristics (like country, device type, or user tier) are balanced across experiment variants. This reduces confounding variables and increases the precision of treatment effect estimates. Without stratification, random chance might lead to imbalanced groups that could bias results. It doesn't directly affect sample size, computational complexity, or model accuracy.

### 9. What is the minimum detectable effect (MDE) in experiment design?
- a) The largest effect the experiment can measure
- b) The smallest effect size that would be practically significant
- c) The current baseline performance metric
- d) The confidence interval width

**Answer: b) The smallest effect size that would be practically significant**

**Explanation:** The minimum detectable effect (MDE) is the smallest improvement that would be worth detecting and acting upon from a business perspective. It's used in power analysis to determine required sample sizes - the experiment should be powered to detect effects at least as large as the MDE. It's not the maximum measurable effect, baseline performance, or confidence interval width, but rather the threshold for practical significance.

### 10. In Thompson Sampling, what probability distribution is commonly used for binary reward problems?
- a) Normal distribution
- b) Poisson distribution
- c) Beta distribution
- d) Exponential distribution

**Answer: c) Beta distribution**

**Explanation:** Thompson Sampling for binary rewards (success/failure) typically uses the Beta distribution because it's the conjugate prior for the Bernoulli likelihood. This means that when you observe successes and failures, you can easily update the Beta distribution parameters (alpha for successes, beta for failures). The Beta distribution naturally represents uncertainty about a probability parameter, making it ideal for modeling click-through rates, conversion rates, and other binary outcomes.