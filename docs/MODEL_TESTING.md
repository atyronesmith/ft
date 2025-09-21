# Model Testing and Cleanup

This document explains the automatic cleanup functionality and generation testing for trained models.

## Automatic Training Run Cleanup

### Overview
Training runs are automatically cleaned up to keep only the 3 most recent runs, preventing disk space issues.

### How It Works
- **When**: Cleanup runs automatically after each training completes
- **What**: Removes entire training run directories (including all checkpoints)
- **How Many**: Keeps the 3 most recent training runs by modification time
- **Location**: Searches in the parent directory of the current output directory

### Cleanup Logic
```python
# 1. Find all directories containing final_model/lora_weights.npz
# 2. Sort by modification time (newest first)
# 3. Keep the 3 most recent
# 4. Remove older directories completely
```

### Example Output
```
2025-09-20 22:45:30.123 | INFO | trainer:cleanup_old_training_runs:417 - Cleaning up 2 old training runs (keeping 3 most recent)
2025-09-20 22:45:30.125 | INFO | trainer:cleanup_old_training_runs:422 - Removing old training run: pytest-160 (from 2025-09-20 22:30:15)
2025-09-20 22:45:30.180 | INFO | trainer:cleanup_old_training_runs:422 - Removing old training run: pytest-159 (from 2025-09-20 22:25:10)
```

### Configuration
The cleanup behavior can be customized by modifying the `max_runs` parameter in the trainer:

```python
# In trainer.py, change this line:
self.cleanup_old_training_runs(max_runs=3)  # Change 3 to desired number
```

## Generation Testing

### Overview
The generation test script automatically finds the most recent trained model and tests it on 100 world capital questions.

### Scripts Available

#### 1. Main Test Script
```bash
python scripts/test_model_generation.py \
    --base-model microsoft/DialoGPT-small \
    --test-dir training \
    --output results.json
```

#### 2. Simple Wrapper Script
```bash
./scripts/test_latest_model.sh [base_model] [test_dir]
```

### Test Process

1. **Find Latest Model**: Searches for the most recent training run with valid LoRA weights
2. **Load Model**: Loads base model + applies LoRA weights
3. **Generate Questions**: Creates 100 world capital questions
4. **Test Generation**: Runs generation on all questions
5. **Evaluate Accuracy**: Checks answers against expected results
6. **Report Results**: Provides detailed accuracy and performance metrics

### Sample Output

```
🧪 Model Generation Test
==================================================
🔍 Searching for training runs in: training
📁 Most recent training run: run-163
📅 Model path: training/run-163/final_model

🤖 Loading base model: microsoft/DialoGPT-small
📥 Loading LoRA weights from: /tmp/.../final_model/lora_weights.npz
✅ Model loaded with LoRA weights applied

📝 Generating 100 capital questions...
✅ Generated 100 questions

🚀 Starting generation test...
   Progress: 0/100 questions processed...
   Progress: 20/100 questions processed...
   Progress: 40/100 questions processed...
   Progress: 60/100 questions processed...
   Progress: 80/100 questions processed...

⏱️  Generation completed in 45.23 seconds
📊 Average time per question: 0.452 seconds

📈 Evaluating accuracy...

🎯 Generation Test Results
==============================
Total Questions: 100
Correct Answers: 78 (78.0%)
Partial Matches: 15 (15.0%)
Incorrect: 7
Generation Speed: 2.21 questions/sec

📋 Sample Results:
1. Q: What is the capital of France?
   Expected: Paris
   Generated: The capital of France is Paris.
   Status: ✅ CORRECT

2. Q: What is the capital of Germany?
   Expected: Berlin
   Generated: Berlin is the capital city of Germany.
   Status: ✅ CORRECT

📊 Detailed results saved to: generation_test_results.json
🎉 Test PASSED! Accuracy: 78.0%
```

### Accuracy Evaluation

The script evaluates answers using multiple criteria:

- **✅ CORRECT**: Expected answer found in generated text (case-insensitive)
- **🟡 PARTIAL**: Some words from expected answer found
- **❌ INCORRECT**: No match found

### Test Results

Results are saved to a JSON file containing:

```json
{
  "total_questions": 100,
  "correct": 78,
  "partial": 15,
  "incorrect": 7,
  "accuracy_percent": 78.0,
  "partial_percent": 15.0,
  "sample_results": [...]
}
```

### Exit Codes

- **0**: Test passed (accuracy ≥ 40%) or partial success
- **1**: Test failed (accuracy < 40%) or error occurred

## Integration with Training

### Automatic Workflow

1. **Training Completes**: LoRA fine-tuning finishes
2. **Save Final Model**: Checkpoint saved to `final_model/`
3. **Cleanup Old Runs**: Automatically removes old training runs
4. **Optional Testing**: Can immediately test the new model

### Manual Testing Workflow

After training completes:

```bash
# Simple test with defaults
./scripts/test_latest_model.sh

# Custom test
python scripts/test_model_generation.py \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --test-dir ./output \
    --output my_test_results.json
```

## File Structure

After training and cleanup, you'll have:

```
training/
├── pytest-163/          # Most recent (kept)
│   └── .../final_model/
│       ├── lora_weights.npz
│       ├── training_state.json
│       └── lora_config.json
├── pytest-162/          # Second most recent (kept)
│   └── .../final_model/
└── pytest-161/          # Third most recent (kept)
    └── .../final_model/
# pytest-160, pytest-159, etc. automatically removed
```

## Troubleshooting

### No Training Runs Found
```
❌ No training runs found!
   Searched in: training
   Looking for directories with: final_model/lora_weights.npz
```

**Solutions**:
- Check the correct test directory path
- Ensure training completed successfully
- Verify LoRA weights were saved

### Model Loading Errors
```
❌ Failed to load model: [Error details]
```

**Solutions**:
- Ensure base model name is correct
- Check internet connection for HuggingFace downloads
- Verify LoRA weights file is not corrupted

### Low Accuracy Results
```
❌ Test FAILED! Accuracy: 25.0%
```

**Possible Causes**:
- Model not sufficiently trained
- Base model not suitable for the task
- LoRA configuration issues
- Generation parameters need tuning

## Configuration Options

### Cleanup Settings
```python
# In trainer.py
self.cleanup_old_training_runs(max_runs=3)  # Keep 3 most recent
self.cleanup_old_training_runs(max_runs=5)  # Keep 5 most recent
```

### Generation Settings
```python
# In test script, modify GenerationConfig:
config = GenerationConfig(
    max_tokens=50,        # Longer responses
    temperature=0.0,      # Deterministic (greedy)
    top_p=0.95,          # Nucleus sampling
    verbose=False        # Debug output
)
```

This testing framework provides comprehensive evaluation of your trained models while automatically managing disk space usage.