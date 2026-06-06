  ---
name: create-qa-spectrum
description:
  Streams, filters, and selects diverse QA datasets across multiple reasoning families, and pushes the final dataset to Hugging Face.
---

# Create Diverse QA Spectrum Reasoning Dataset

## Overview

curing a diverse and representative dataset is crucial for evaluating language models on their multi-dimensional reasoning capabilities. To evaluate models accurately, a QA dataset should span a wide "spectrum" of difficulty and cognitive demands, from simple factual retrieval to multi-step logical deduction.

This skill implements a highly modular and automated curation workflow to select exactly 1,000 diverse reasoning examples.

### Curation Philosophy and Core Research

#### 1. Target Reasoning Families
We define six distinct families of reasoning to benchmark different model capabilities:
- **Factual Reasoning** (Memorization & World Knowledge): WebQuestions / TriviaQA.
- **Commonsense / Implicit Reasoning** (Social & Physical Logic): CommonsenseQA / StrategyQA.
- **Science QA** (Reading Comprehension & Facts): ARC / SciQ / OpenBookQA.
- **Arithmetic Word Problems** (Multi-step Logic & Math): GSM8K / ASDiv.
- **Grounded Multi-hop** (Information Synthesis across sources): HotpotQA / MuSiQue.
- **Extractive Control** (Simple span extraction checks): SQuAD / QA-SRL.

#### 2. Enhancing Diversity & Curation Techniques
To make the dataset as diverse as possible, the curation process is divided into two phases: **Cheap Prior Filtering** and **Semantic Diversity Selection**.

- **Cheap Prior Filtering (Syntactic and Structural)**:
  - **Factual Entity Limiting**: To prevent the dataset from being dominated by a few famous figures (e.g. popular pop singers, historical events), candidate factual questions are limited to a maximum of 3 per named entity.
  - **Arithmetic Template Normalization**: Math word problems are highly prone to template repetition (e.g., "John has $X$ apples, Betty has $Y$ oranges..."). We normalize candidate math questions by substituting numbers, gendered pronouns, and names with placeholders (`<NUM>`, `<PRONOUN>`, `<NAME>`). Structural duplicates are then strictly rejected.
  - **Quality Filtering**: Automatic rejection of questions requiring unavailable external artifacts (e.g., images, tables, external charts) or containing non-English character blocks.

- **Semantic Diversity Curation (Embedding Space)**:
  - We run **Global Semantic Deduplication** across all source datasets to prevent cross-dataset redundancy.
  - **MaxMin Greedy Selection**: Using sentence embeddings (`all-MiniLM-L6-v2` or similar), we greedily choose candidates. In each iteration, we select the candidate that has the *minimum* similarity to the currently selected set:
    $$c^* = \arg\min_{c \in \text{Candidates}} \max_{s \in \text{Selected}} \text{CosineSimilarity}(c, s)$$
    This pushes the selected examples as far apart as possible in embedding space, maximizing coverage of the semantic domain.
  - **Hard Similarity Thresholds**: Cosine similarity $\ge 0.90$ is automatically rejected. Similarity between $0.84$ and $0.90$ is rejected unless their gold answers differ, confirming they ask distinct questions.

---

## Dependencies
This skill is designed for **Linux** environments with `uv` installed.
- `uv` (standard python virtualenv and script runner)
- `datasets`
- `huggingface_hub`
- `sentence-transformers`
- `tqdm`
- `numpy`
- `scipy`

---

## Quick Start

Execute the complete modular pipeline to collect, select, validate, and upload a QA spectrum dataset:

```bash
# 1. Collect candidates
uv run scripts/create_spectrum_dataset.py collect --output candidates.json

# 2. Select diverse examples (uses CPU/GPU automatically)
uv run scripts/create_spectrum_dataset.py select --candidates candidates.json --output selected.json

# 3. Validate constraints
uv run scripts/create_spectrum_dataset.py validate --dataset-file selected.json

# 4. Upload to Hugging Face
export HF_TOKEN="your_hf_token_here"
uv run scripts/create_spectrum_dataset.py upload --dataset-file selected.json --repo-id my-username/reasoning-spectrum-qa
```

---

## Utility Scripts (CLI-based)

The `scripts/create_spectrum_dataset.py` orchestrator supports the following subcommands:

### 1. `collect`
Streams candidates from Hugging Face datasets and runs cheap syntactic/structural filters.
```bash
uv run scripts/create_spectrum_dataset.py collect \
  --output candidates.json \
  --factual-limit 480 \
  --commonsense-limit 480 \
  --science-limit 1000 \
  --arithmetic-limit 800 \
  --multihop-limit 880 \
  --extractive-limit 360
```

### 2. `select`
Loads candidate JSON, computes sentence embeddings, and performs greedy global MaxMin selection to achieve targets.
```bash
uv run scripts/create_spectrum_dataset.py select \
  --candidates candidates.json \
  --output selected.json \
  --model-name "all-MiniLM-L6-v2" \
  --factual-target 120 \
  --commonsense-target 120 \
  --science-target 250 \
  --arithmetic-target 200 \
  --multihop-target 220 \
  --extractive-target 90
```
> [!TIP]
> To use a better or larger embedding model in the future (e.g., `all-mpnet-base-v2` or `BAAI/bge-large-en-v1.5`), simply override the `--model-name` argument.

### 3. `validate`
Ensures all dataset constraints are strictly met: exactly 1000 rows, valid schemas, and no near-duplicate pairs (similarity $\ge 0.90$).
```bash
uv run scripts/create_spectrum_dataset.py validate --dataset-file selected.json
```

### 4. `upload`
Validates and pushes the dataset to Hugging Face Hub (as private by default, unless `--public` is specified). Generates `README.md` and `stats.json`.
```bash
uv run scripts/create_spectrum_dataset.py upload \
  --dataset-file selected.json \
  --repo-id my-org/my-dataset \
  --public
```

---

## Rate Limiting & Network Resilience
- Loading and streaming datasets from Hugging Face does not impose strict rate-limiting for these payload sizes, but the utility script retries on network transient failures.
- HF Hub dataset pushing uses direct Chunk uploads and handles validation.

---

## Common Mistakes & Troubleshooting

1. **Hugging Face Single-word Repository Error**:
   Older codebases or API versions might parse dataset IDs like `squad` or `gsm8k` incorrectly, throwing a `Repository id must be 'namespace/name'` validation error.
   *Solution*: Always use fully namespaced IDs: `rajpurkar/squad`, `openai/gsm8k`, `allenai/ai2_arc`, `allenai/sciq`, `stanfordnlp/web_questions`, `tau/commonsense_qa`, `hotpotqa/hotpot_qa`.

2. **CUDA Out-Of-Memory / CPU fallback**:
   Running embedding generation with large models or batch sizes on minimal environments might cause CUDA out-of-memory errors.
   *Solution*: The sentence-transformer library naturally falls back to CPU. The batch size is configured to 64 to keep peak RAM usage low.

3. **Hugging Face Hub Authentication**:
   Using `push_to_hub` without providing the write token or setting `HF_TOKEN` will fail.
   *Solution*: Read the token explicitly from the `HF_TOKEN` environment variable and pass it to functions. Never commit, log, or hardcode it.
