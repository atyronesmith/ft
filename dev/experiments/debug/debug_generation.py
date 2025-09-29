#!/usr/bin/env python3
"""Debug generation issue by analyzing training vs inference formats."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

# Test the format used in training
messages = [
    {'role': 'system', 'content': 'You are a helpful geography assistant who provides accurate, concise answers about world capitals.'},
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': 'Paris'}
]

training_text = tokenizer.apply_chat_template(messages, tokenize=False)
print('=== TRAINING TEXT ===')
print(repr(training_text))

# Tokenize it
tokens = tokenizer.encode(training_text, add_special_tokens=False)
print('\n=== TOKENS ===')
for i, token in enumerate(tokens):
    decoded = tokenizer.decode([token])
    print(f'{i:2d}: {token:5d} -> {repr(decoded)}')

# Find where assistant starts
assistant_marker = '<|assistant|>'
assistant_pos = training_text.find(assistant_marker)
if assistant_pos != -1:
    prefix_text = training_text[:assistant_pos + len(assistant_marker) + 1]  # Include newline
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    print(f'\n=== ASSISTANT STARTS AT TOKEN {len(prefix_tokens)} ===')
    print(f'Training should focus on tokens {len(prefix_tokens)}+ for the answer')

    # Show what the target sequence should be
    answer_start = len(prefix_tokens)
    answer_tokens = tokens[answer_start:]
    print(f'Answer tokens: {answer_tokens}')
    answer_text = tokenizer.decode(answer_tokens)
    print(f'Answer text: {repr(answer_text)}')

# Test inference format
inference_messages = messages[:-1]  # Remove assistant response
inference_text = tokenizer.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True)
print('\n=== INFERENCE TEXT ===')
print(repr(inference_text))

# What should the model generate?
expected_generation = training_text[len(inference_text):]
print('\n=== EXPECTED GENERATION ===')
print(repr(expected_generation))
