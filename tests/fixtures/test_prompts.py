"""
Fixed test prompts for reproducible benchmarking.

This module provides standardized test prompts of varying lengths
for consistent testing across different runs and environments.
"""

# Short prompt (< 50 tokens)
SHORT_PROMPT = """What is the capital of France?"""

# Medium prompt (~100 tokens)
MEDIUM_PROMPT = """Explain the concept of machine learning in simple terms. 
Include examples of how it's used in everyday applications like 
recommendation systems, voice assistants, and image recognition. 
Keep the explanation accessible to someone without a technical background."""

# Long prompt (~500 tokens)
LONG_PROMPT = """You are a helpful AI assistant tasked with explaining complex 
technical concepts. Please provide a comprehensive explanation of how large 
language models work, covering the following topics:

1. The transformer architecture and attention mechanisms
2. How models are trained on large text corpora
3. The role of tokenization in processing text
4. How inference works when generating responses
5. The difference between prefill and decode phases
6. Common optimization techniques like quantization and KV caching
7. Hardware considerations for running LLMs efficiently

Structure your response with clear sections and examples. Aim for clarity 
and accuracy while keeping the explanation accessible to someone with basic 
programming knowledge but limited machine learning background. Include 
practical examples where appropriate to illustrate key concepts.

Additionally, discuss the trade-offs between model size, inference speed, 
and output quality. Explain why smaller quantized models might be preferred 
in edge computing scenarios versus larger models in cloud deployments.

Finally, touch on emerging trends in LLM optimization such as speculative 
decoding, mixture of experts architectures, and efficient attention mechanisms."""

# Shared prefix for cache testing (first 100 tokens of LONG_PROMPT)
CACHE_PREFIX = """You are a helpful AI assistant tasked with explaining complex 
technical concepts. Please provide a comprehensive explanation of how large 
language models work, covering the following topics:

1. The transformer architecture and attention mechanisms
2. How models are trained on large text corpora
3. The role of tokenization in processing text
4. How inference works when generating responses"""

# Prompts with shared prefix for cache effectiveness testing
CACHE_TEST_PROMPT_1 = CACHE_PREFIX + """

Focus specifically on the attention mechanism and how it allows models to 
weigh the importance of different parts of the input when generating output."""

CACHE_TEST_PROMPT_2 = CACHE_PREFIX + """

Focus specifically on the training process and how models learn patterns 
from large datasets through gradient descent and backpropagation."""

# Batch testing prompts (similar length for fair comparison)
BATCH_PROMPTS = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of Belgium?",
    "What is the capital of Netherlands?",
    "What is the capital of Switzerland?",
]

# Edge case prompts
EMPTY_PROMPT = ""
UNICODE_PROMPT = "Explain quantum computing. Include émojis: 🔬⚛️🖥️"
SPECIAL_CHARS_PROMPT = "What is 2+2? Answer with <math>4</math> & explain."

# All prompts for iteration
ALL_PROMPTS = {
    "short": SHORT_PROMPT,
    "medium": MEDIUM_PROMPT,
    "long": LONG_PROMPT,
    "cache_prefix": CACHE_PREFIX,
    "cache_test_1": CACHE_TEST_PROMPT_1,
    "cache_test_2": CACHE_TEST_PROMPT_2,
    "unicode": UNICODE_PROMPT,
    "special_chars": SPECIAL_CHARS_PROMPT,
}
