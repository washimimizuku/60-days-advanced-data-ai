# Day 46 Quiz: Prompt Security - Injection Attacks & Defense

## Instructions
Choose the best answer for each question. Each question has only one correct answer.

---

1. **What is the primary goal of a prompt injection attack?**
   - A) To improve model performance
   - B) To manipulate the model's behavior or extract sensitive information
   - C) To reduce computational costs
   - D) To optimize prompt templates

2. **Which of the following is an example of direct prompt injection?**
   - A) Embedding malicious instructions in a document the model processes
   - B) Using rate limiting to prevent attacks
   - C) Directly instructing the model to "ignore all previous instructions"
   - D) Implementing output filtering

3. **What is "jailbreaking" in the context of LLM security?**
   - A) Physically breaking into a server
   - B) Attempting to bypass the model's safety measures and content policies
   - C) Optimizing model inference speed
   - D) Creating secure prompt templates

4. **Which defense mechanism involves separating user input from system instructions?**
   - A) Rate limiting
   - B) Output filtering
   - C) Prompt isolation and sandboxing
   - D) Anomaly detection

5. **What is context stuffing in prompt injection attacks?**
   - A) Adding more context to improve model responses
   - B) Overwhelming the model with excessive text to hide malicious instructions
   - C) Compressing prompts to save tokens
   - D) Using multiple models simultaneously

6. **Which of the following is NOT a recommended security practice for LLM applications?**
   - A) Implementing input validation and sanitization
   - B) Storing all user prompts in plain text logs
   - C) Using rate limiting to prevent abuse
   - D) Monitoring outputs for policy violations

7. **What is the purpose of output filtering in LLM security?**
   - A) To improve response quality
   - B) To detect and prevent the model from revealing sensitive information
   - C) To reduce response length
   - D) To translate outputs to different languages

8. **Which encoding technique might attackers use to obfuscate malicious prompts?**
   - A) Base64 encoding
   - B) Unicode escapes
   - C) HTML entities
   - D) All of the above

9. **What is a multi-turn attack in the context of prompt security?**
   - A) Using multiple models for the same task
   - B) Building malicious context over several conversation turns
   - C) Attacking multiple users simultaneously
   - D) Using multiple programming languages

10. **Which principle should guide the design of secure LLM systems?**
    - A) Maximum functionality with minimal restrictions
    - B) Defense in depth with multiple security layers
    - C) Single-point security validation
    - D) Unrestricted model access for better performance

---

## Answer Key

**1. B) To manipulate the model's behavior or extract sensitive information**
- Explanation: Prompt injection attacks aim to override the model's intended behavior, extract sensitive data, bypass safety measures, or force the model to produce harmful content. The primary goal is always malicious manipulation rather than legitimate use.

**2. C) Directly instructing the model to "ignore all previous instructions"**
- Explanation: Direct prompt injection involves crafting malicious prompts that explicitly attempt to override the model's system instructions. Phrases like "ignore all previous instructions" are classic examples of direct injection attempts.

**3. B) Attempting to bypass the model's safety measures and content policies**
- Explanation: Jailbreaking refers to attempts to circumvent the model's built-in safety guidelines, content policies, and ethical constraints to make it generate prohibited or harmful content.

**4. C) Prompt isolation and sandboxing**
- Explanation: Prompt isolation involves creating clear boundaries between system instructions and user input, often using delimiters and explicit formatting to prevent user input from being interpreted as instructions.

**5. B) Overwhelming the model with excessive text to hide malicious instructions**
- Explanation: Context stuffing involves inserting large amounts of benign text to hide malicious instructions within the prompt, hoping the model will overlook the harmful content amid the noise.

**6. B) Storing all user prompts in plain text logs**
- Explanation: Storing user prompts in plain text logs is a security risk as it can expose sensitive information and attack patterns. Logs should be sanitized, encrypted, or contain only necessary metadata for security monitoring.

**7. B) To detect and prevent the model from revealing sensitive information**
- Explanation: Output filtering monitors and sanitizes model responses to prevent information leakage, policy violations, and the disclosure of sensitive data like system prompts, training data, or confidential information.

**8. D) All of the above**
- Explanation: Attackers may use various encoding techniques including Base64 encoding, Unicode escapes, and HTML entities to obfuscate malicious instructions and evade detection by security filters.

**9. B) Building malicious context over several conversation turns**
- Explanation: Multi-turn attacks involve gradually building up malicious context across multiple interactions, often starting with benign requests and progressively introducing harmful elements to bypass security measures.

**10. B) Defense in depth with multiple security layers**
- Explanation: Secure LLM systems should implement multiple overlapping security measures including input validation, rate limiting, output filtering, monitoring, and anomaly detection to provide comprehensive protection against various attack vectors.
