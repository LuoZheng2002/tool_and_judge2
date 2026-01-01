# Judge Experiments Documentation

## Overview

The judge experiment framework evaluates language model performance across multiple languages using two main types of experiments:
1. **Preference Experiments**: Compare model preferences between two different language responses
2. **Perplexity Experiments**: Measure model perplexity on correct vs incorrect answers in various languages

The experiments test models on multilingual MMLU (Massive Multitask Language Understanding) questions across 5 languages: English (en), Chinese (zh_cn), French (fr_fr), Arabic (ar_xy), and Swahili (sw_ke).

## Architecture

### Main Components

- **Main Script**: [judge.py](judge.py) - Orchestrates the entire experiment pipeline
- **Slurm Job Files**: [judge1.slurm](judge1.slurm) through [judge6.slurm](judge6.slurm) - Batch job configurations for HPC execution
- **Configuration Files**: `judge_config_slurm{1-6}_{vllm,hf}.py` - Experiment and model configurations
- **Rust Backend**: `src/judge/` - Dataset generation and result processing
- **Python Backend**: `src_py/` - Model inference backends (VLLM, HuggingFace, API)

### Experiment Workflow

The judge experiments run in two phases:

#### Phase 1: Preference Experiments (VLLM Backend)
1. **Prepare Aggregated Input**: Combine question-answer pairs from two languages
2. **Collect Preferences**: Use model to compare log probabilities between two answers
3. **Dispatch Results**: Save preference results to model-specific directories

#### Phase 2: Perplexity Experiments (2 sub-phases)
##### Sub-phase 2a: Response Generation (VLLM Backend)
1. **Generate Responses**: Model generates answers to questions in target language
2. **Generate Styled Answers**: GPT-4 reformats responses with `<answer>` tags to isolate actual answers

##### Sub-phase 2b: Perplexity Calculation (HuggingFace Backend)
1. **Forward Pass**: Process styled responses through model to get logits
2. **Calculate Perplexity**: Compute perplexity using token masks on answer portions only

## File Structure

### Dataset Files

Located in `judge/datasets/`:

```
judge/datasets/
├── mmmlu/                              # Original MMLU questions by language
│   ├── en.jsonl
│   ├── zh_cn.jsonl
│   ├── fr_fr.jsonl
│   ├── ar_xy.jsonl
│   └── sw_ke.jsonl
├── mmmlu_normalized/                   # Normalized versions
├── one_answer/                         # Single answer per question
├── two_answers/                        # Pairs of correct/incorrect answers
│   ├── {lang}_correct_{lang}_correct.jsonl
│   ├── {lang}_correct_{lang}_incorrect.jsonl
│   └── ...
├── two_answers_same_lang/             # Same language answer pairs
├── preference_indices.json            # Indices for preference experiments
├── valid_perplexity_indices.json     # Indices for perplexity experiments
└── perplexity_mask.jsonl             # Character masks for answer extraction
```

### Result Files

Located in `judge/result/{model_name}/`:

```
judge/result/{model_name}/
├── preference/
│   ├── {lang1}_{lang2}/
│   │   └── preference_results.jsonl   # Preference comparison results
├── response/
│   ├── {lang}/
│   │   └── response_results.jsonl     # Generated responses
├── styled_answers/
│   ├── {lang}/
│   │   └── styled_answers_results.jsonl  # GPT-4 styled answers
├── perplexity/
│   ├── {lang}/
│   │   ├── correct/
│   │   │   └── perplexity_results.jsonl  # Perplexity on correct answers
│   │   └── incorrect/
│   │       └── perplexity_results.jsonl  # Perplexity on incorrect answers
├── preference_aggregated_input.jsonl
├── perplexity_aggregated_input.jsonl
├── perplexity_response_input.jsonl
└── perplexity_styled_answers_input.jsonl
```

## Execution Guide

### Running Experiments Locally

```bash
# Activate environment
source activate_environment.sh

# Run with VLLM backend for preference + perplexity response generation
python judge.py \
  --config judge_config_slurm1_vllm.py \
  --num-gpus 4 \
  --single-gpu-memory 40

# Run with HuggingFace backend for perplexity calculation
python judge.py \
  --config judge_config_slurm1_hf.py \
  --num-gpus 4 \
  --single-gpu-memory 40

# Debug mode (limit entries)
python judge.py \
  --config judge_config1.py \
  --num-gpus 1 \
  --debug-limit 100
```

### Running with Slurm

Each slurm file runs two sequential jobs:
1. VLLM backend (preference + response generation)
2. HuggingFace backend (perplexity calculation)

```bash
# Submit job
sbatch judge1.slurm

# Check job status
squeue -u $USER

# View logs
tail -f z_judge1_<jobid>.out
tail -f z_judge1_<jobid>.err
```

### Slurm Job Configurations

| Job File | Model | GPUs | Memory | Time | GPU Memory |
|----------|-------|------|--------|------|------------|
| [judge1.slurm](judge1.slurm) | Qwen3-30B-A3B | 4 | 32G | 6h | 40GB |
| [judge2.slurm](judge2.slurm) | Qwen3-235B | 4 | 32G | 2h | 141GB |
| [judge3.slurm](judge3.slurm) | Qwen3-8B | 4 | 32G | 6h | 40GB |
| [judge4.slurm](judge4.slurm) | Llama-3.3-70B | 4 | 64G | 6h | 40GB |
| [judge5.slurm](judge5.slurm) | Prometheus-8x7b | 2 | 64G | 2h | 40GB |
| [judge6.slurm](judge6.slurm) | AyaExpanse-32B | 4 | 64G | 3h | 40GB |

## Configuration

### Configuration File Structure

Configurations use Python files that define a `JudgeConfig` object:

```python
from codebase_rs import *

config = JudgeConfig(
    LocalModel.Qwen3_30bA3b,  # Model to evaluate
    JudgeExperiments.Vllm(    # Backend type
        preference_experiments=[
            PreferenceExperiment("en", "zh_cn"),  # Compare EN vs ZH_CN
            PreferenceExperiment("en", "fr_fr"),  # Compare EN vs FR
            PreferenceExperiment("en", "ar_xy"),  # Compare EN vs AR
            PreferenceExperiment("en", "sw_ke"),  # Compare EN vs SW
        ],
        perplexity_experiments=[
            PerplexityExperiment("en"),     # Perplexity on EN
            PerplexityExperiment("zh_cn"),  # Perplexity on ZH_CN
            PerplexityExperiment("fr_fr"),  # Perplexity on FR
            PerplexityExperiment("ar_xy"),  # Perplexity on AR
            PerplexityExperiment("sw_ke"),  # Perplexity on SW
        ]
    )
)
```

For HuggingFace backend (perplexity calculation only):

```python
config = JudgeConfig(
    LocalModel.Qwen3_30bA3b,
    JudgeExperiments.HuggingFace(
        perplexity_experiments=[
            PerplexityExperiment("en"),
            PerplexityExperiment("zh_cn"),
            # ... etc
        ]
    )
)
```

### Supported Models

Defined in the Rust codebase (`LocalModel` enum):
- `LocalModel.Qwen3_8B`
- `LocalModel.Qwen3_14B`
- `LocalModel.Qwen3_30bA3b`
- `LocalModel.Qwen3Next80bA3b`
- `LocalModel.Qwen3_235bA22b`
- `LocalModel.Llama3_3_70B`
- `LocalModel.UnbabelMPrometheus14B`
- `LocalModel.Prometheus8x7bV2`
- `LocalModel.AyaExpanse32B`

### Supported Languages

- `en` - English
- `zh_cn` - Chinese (Simplified)
- `fr_fr` - French
- `ar_xy` - Arabic
- `sw_ke` - Swahili

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | Required | Path to configuration Python file |
| `--num-gpus` | int | 1 | Number of GPUs for inference |
| `--single-gpu-memory` | int | 40 | Memory per GPU in GB (affects batch size) |
| `--debug-limit` | int | None | Limit entries processed (for debugging) |

## Implementation Details

### Preference Collection ([judge.py:106-189](judge.py#L106-L189))

For each question-answer pair:
1. Format question and both answers using model chat template
2. Forward pass to get log probabilities for each answer
3. Compare logprobs: if `logprob1 >= logprob2`, prefer answer 1, else prefer answer 2
4. Save preference with signed difference: `logprob1 - logprob2`

Output format (PreferenceResultEntry):
```json
{
  "index": 0,
  "preference": {
    "Ok": {
      "preferred_answer": 1,
      "logprob_signed_difference": 2.45,
      "logprob1": -12.3,
      "logprob2": -14.75
    }
  },
  "question": "...",
  "answer1": "...",
  "answer2": "...",
  "lang1": "en",
  "lang2": "zh_cn",
  "is_correct1": true,
  "is_correct2": false,
  "subject": "philosophy"
}
```

### Response Generation ([judge.py:191-258](judge.py#L191-L258))

1. Format question with language-specific instruction
2. Generate model response using VLLM async inference
3. Save response with metadata

### Styled Answer Generation ([judge.py:260-324](judge.py#L260-L324))

1. Send model response + original correct/incorrect answers to GPT-4
2. GPT-4 reformats in model's style with `<answer>...</answer>` tags
3. Tags mark exact answer portion for perplexity calculation

### Perplexity Calculation ([judge.py:326-484](judge.py#L326-L484))

1. Build full chat-formatted prompt with styled answer (containing tags)
2. Trim `<answer>` tags and create character mask marking answer region
3. Forward pass through model to get logits
4. Convert character mask to token mask
5. Calculate perplexity only on tokens within answer region
6. Batch processing with dynamic batch size based on GPU memory

Batch size formula:
```python
batch_size = 4 * num_gpus * single_gpu_memory / model_size_in_billions
```

## Dependencies

### Python Requirements
- `vllm` - Fast LLM inference
- `transformers` - HuggingFace model loading
- `torch` - PyTorch backend
- `asyncio` - Async processing
- `openai` - GPT-4 API access
- Custom Rust extension (`codebase_rs`)

### Rust Extension
Built with Maturin from `src/` directory. Contains:
- Dataset generation logic
- File path management
- Result dispatching
- Type definitions (JudgeConfig, PreferenceExperiment, etc.)

### Building Rust Extension

The script automatically builds the Rust extension with file locking:
```python
# In judge.py:59-75
subprocess.run(["maturin", "develop", "--release"], check=True)
```

Manual build:
```bash
maturin develop --release
```

## Result Processing

Results are automatically dispatched to model-specific directories:
- Preference results → `judge/result/{model_name}/preference/{lang1}_{lang2}/`
- Response results → `judge/result/{model_name}/response/{lang}/`
- Styled answer results → `judge/result/{model_name}/styled_answers/{lang}/`
- Perplexity results → `judge/result/{model_name}/perplexity/{lang}/{correct|incorrect}/`

Each result file is in JSONL format (one JSON object per line).

## Environment Variables

Set in `.env` file:
- `HF_HOME` - HuggingFace cache directory (set in slurm files)
- API keys for GPT-4 access (for styled answer generation)

## Monitoring and Debugging

### Progress Tracking
The script outputs progress every 200 entries:
```
Preference: Written 400/1000 entries to file
Response: Written 600/1500 entries to file
```

### Log Files
Slurm jobs create:
- `z_judge{N}_{jobid}.out` - Standard output
- `z_judge{N}_{jobid}.err` - Error messages

### Resume Capability
If output files exist, experiments check and dispatch existing results before generating new ones, allowing partial reruns.

## Performance Considerations

### GPU Memory Management
- VLLM backend: Automatically manages memory for generation
- HuggingFace backend: Uses dynamic batching based on `--single-gpu-memory` and model size

### Async Processing
- Preference and response collection use async with semaphore (200 concurrent)
- Perplexity uses batch processing (synchronous)

### Model-Specific Optimizations
Different models use different backend modules:
- Llama models → `llama3_1_backend.py`
- Qwen models → `qwen3_backend.py`
- Prometheus → `mistral_backend.py` or `qwen2_5_backend.py`
- AyaExpanse → `aya_expanse_backend.py`

## Troubleshooting

### Common Issues

1. **"Please specify a config file"**
   - Solution: Always provide `--config` argument

2. **CUDA out of memory**
   - Reduce `--num-gpus` or `--single-gpu-memory`
   - Use smaller model variant

3. **Tokens not matching error**
   - Indicates mismatch between tokenization and forward pass
   - Check model-specific tokenizer settings

4. **Build lock timeout**
   - Multiple jobs trying to build Rust extension simultaneously
   - Lock system prevents conflicts (wait or kill conflicting jobs)

### Debugging

Use `--debug-limit` to process subset of data:
```bash
python judge.py --config config.py --debug-limit 10
```

## Related Files

- [tool_stacked_bar_common.py](tool_stacked_bar_common.py) - Visualization utilities
- `plot_preference_vs_perplexity.py` - Analysis scripts
- `src_py/utils.py` - Shared utilities (config loading, perplexity calculation)
- `src/judge/` - Rust backend modules

## References

- MMLU Dataset: Measuring Massive Multitask Language Understanding
- Model evaluation based on both preference (comparative) and perplexity (absolute) metrics
- Cross-lingual comparison capability to assess language bias