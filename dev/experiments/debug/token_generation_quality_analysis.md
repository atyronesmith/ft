# Root Cause Analysis: Token Generation Quality Difference

**Date:** 2025-01-01
**Issue:** `lora.py` generates coherent text while `scripts/test_model_generation.py` generates random output despite identical model and LoRA weights loading.

## Summary

After detailed comparison of both scripts, I've identified **multiple critical issues** that explain why `lora.py` generates coherent text while `scripts/test_model_generation.py` generates random output.

## 🔴 **CRITICAL BUG #1: Broken Context Handling in Generation Loop**

**Location:** `scripts/test_model_generation.py:420`

```python
# BROKEN: Only passes single token in subsequent iterations
y = mx.array([token_id])
```

**Working approach (`lora.py` via `lora_utils.generate()`):**
- Passes the new token only (`y = sample(logits)`) but relies on cache for full context
- Cache maintains all previous context correctly

**Broken approach (`test_model_generation.py`):**
- First iteration: passes full prompt correctly
- **Subsequent iterations: only passes single new token without proper context building**
- This destroys the sequence context, leading to incoherent generation

### Code Comparison

**Working (`lora_utils.generate()`):**
```python
y = prompt
cache = None
while True:
    logits, cache = model(y[None], cache=cache)
    logits = logits[:, -1, :]
    y = sample(logits)  # Single token, cache maintains context
    yield y
```

**Broken (`test_model_generation.py`):**
```python
y = prompt_array
cache = None
tokens = []

for step in range(max_tokens):
    logits, cache = model(y[None], cache=cache)
    logits = logits[:, -1, :]

    next_token = simple_sample(logits)
    token_id = int(next_token.item())
    tokens.append(token_id)
    y = mx.array([token_id])  # ❌ CONTEXT LOST HERE
```

## 🔴 **CRITICAL ISSUE #2: Completely Different Model Systems**

**Working script (`lora.py`):**
- Uses MLX example's native model loader: `lora_utils.load()`
- Model structure: `Model -> model (LlamaModel) -> layers`
- Accesses layers via: `model.model.layers`
- Simple, battle-tested MLX implementation

**Broken script (`test_model_generation.py`):**
- Uses custom ModelManager: `manager.load_model()`
- Model structure: `MLXLlamaModel -> layers` (direct)
- Accesses layers via: `model.layers`
- Complex custom implementation with potential incompatibilities

### LoRA Application Differences

**Working script:**
```python
for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
    l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
    l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
```

**Broken script:**
```python
layers = model.layers  # Direct access
for layer_idx in range(start_layer, len(layers)):
    layer = layers[layer_idx]
    # Complex LoRA setup with dtype conversions and parameter name fixing
```

## 🟡 **Issue #3: Prompt Formatting Problems**

**Working script:**
- Uses exact user-provided prompt: `"table: 1-10015132-16\ncolumns: Player, No., Nationality, Position, Years in Toronto, School/Club Team\nQ: What is terrence ross' nationality\nA: "`

**Broken script:**
- Extracts questions from training data (table `1-10015132-11`)
- Applies them to different table context (`1-10015132-16`)
- Results in nonsensical questions like asking about players that don't exist in the target table

### Training Data Format
```json
{"text": "table: 1-10015132-11\ncolumns: Player, No., Nationality, Position, Years in Toronto, School/Club Team\nQ: What position does the player who played for butler cc (ks) play?\nA: SELECT Position FROM 1-10015132-11 WHERE School/Club Team = 'Butler CC (KS)'"}
```

**Problem:** Test script extracts questions from `1-10015132-11` data but applies them to `1-10015132-16` table context.

## 🟡 **Issue #4: Generation Parameter Differences**

| Parameter | Working Script | Broken Script |
|-----------|---------------|---------------|
| Max tokens | 50 | 20 (too low) |
| Temperature calculation | `logits * (1/temp)` | `logits / temp` (equivalent but different implementation) |
| Stopping logic | Simple external EOS check | Complex internal stopping |
| Top-p sampling | None | 0.95 |
| Temperature value | 0.8 | 0.7 |

### Working Generation Parameters
```python
# lora.py command line
--temp 0.8 --max-tokens 50

# lora_utils.generate()
def sample(logits: mx.array) -> mx.array:
    return (
        mx.argmax(logits, axis=-1)
        if temp == 0
        else mx.random.categorical(logits * (1 / temp))
    )
```

### Broken Generation Parameters
```python
# test_model_generation.py
config = GenerationConfig(
    max_tokens=20,  # Too low
    temperature=0.7,
    top_p=0.95,
    verbose=debug,
    stop_on_eos=True,
    stop_on_special_tokens=True
)

def simple_sample(logits):
    if config.temperature <= 1e-6:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits / config.temperature)
```

## 🔍 **The Primary Root Cause**

The **context handling bug** (Issue #1) is the primary cause of random text generation. Even if all other issues were fixed, the broken context handling alone would cause incoherent output because the model loses track of what it's generating after the first token.

### Why This Breaks Generation

1. **First iteration:** Model receives full prompt context and generates appropriate first token
2. **Second iteration:** Model only receives the single previous token, loses all prompt context
3. **Subsequent iterations:** Model has no idea what the original question was about
4. **Result:** Random, incoherent text generation

## 🔧 **Recommended Fixes**

### Priority 1: Fix Context Handling
```python
# Current (BROKEN):
y = mx.array([token_id])

# Option 1: Rely on cache (like working script)
# Keep y as single token, ensure cache properly maintains context

# Option 2: Maintain full sequence
# y = mx.concatenate([y, mx.array([token_id])])
```

### Priority 2: Use Consistent Model Loading
Consider switching to the MLX-native model loading approach used by the working script for maximum compatibility.

### Priority 3: Fix Prompt Formatting
Ensure questions are applied to the correct table context, or use the exact same prompt format as the working script.

### Priority 4: Align Generation Parameters
Use the same temperature (0.8), max_tokens (50), and sampling approach as the working script.

## 🧪 **Test Validation**

To validate fixes:
1. Run both scripts with identical prompts
2. Compare first few tokens generated - they should be identical
3. Verify context is maintained by checking that later tokens are contextually appropriate
4. Compare final output quality and coherence

## 🗂️ **File Locations**

- **Working script:** `/Users/aasmith/Dev/ft/dev/experiments/mlx_official_comparison/lora.py`
- **Broken script:** `/Users/aasmith/Dev/ft/scripts/test_model_generation.py`
- **Working utils:** `/Users/aasmith/Dev/ft/dev/experiments/mlx_official_comparison/utils.py`
- **Training data:** `/Users/aasmith/Dev/ft/data/mlx_examples/valid.jsonl`

## 📝 **Notes**

- Both scripts successfully load identical models and LoRA weights
- The issue is purely in the token generation logic, not in model or weight loading
- The working script uses the official MLX examples approach, which should be considered the reference implementation
- The broken script's custom generation logic introduces multiple points of failure