# /// script
# dependencies = [
#   "datasets",
#   "huggingface_hub",
# ]
# ///

import os
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

def validate_dataset(dataset_list):
    print("Validating dataset rules...")
    assert len(dataset_list) == 1000, f"Dataset must have exactly 1000 rows, but got {len(dataset_list)}"
    
    seen_questions = set()
    for row in dataset_list:
        q = row["question"]
        ans = row["gold_answer"]
        family = row["reasoning_family"]
        
        assert q and q.strip(), f"Empty question found in row {row['id']}"
        assert ans and str(ans).strip(), f"Empty answer found in row {row['id']}"
        
        q_norm = " ".join(q.split()).lower()
        assert q_norm not in seen_questions, f"Duplicate question found: {q}"
        seen_questions.add(q_norm)
        
        if row["answer_type"] == "multiple_choice":
            assert row["choices"], f"Multiple choice question with no choices in row {row['id']}"
            assert row["gold_answer_label"], f"Multiple choice question with no label in row {row['id']}"
            labels = [c["label"] for c in row["choices"]]
            assert row["gold_answer_label"] in labels, f"Label {row['gold_answer_label']} not in choices {labels} in row {row['id']}"
            
        if family in ["multihop", "extractive_control"]:
            assert row["context"] and row["context"].strip(), f"Context-dependent family {family} has empty context in row {row['id']}"
            if family == "multihop":
                assert row["supporting_facts"] and row["supporting_facts"].strip(), f"Multihop has empty supporting facts in row {row['id']}"
                
        sim = row.get("selection_similarity", 0.0)
        assert sim < 0.90, f"Row {row['id']} has similarity {sim} >= 0.90 to its closest selected pair {row['closest_selected_id']}"
        
    print("All validation assertions passed successfully!")

def upload_dataset_to_hub(selected_examples, family_counts, collect_stats, semantic_rejections, rejected_pairs, repo_id, token, private=True):
    # Ensure correct schema format
    features = {
        "id": "string",
        "source_dataset": "string",
        "source_split": "string",
        "source_id": "string",
        "question": "string",
        "context": "string",
        "choices": [{"label": "string", "text": "string"}],
        "gold_answer": "string",
        "gold_answer_label": "string",
        "reference_solution": "string",
        "supporting_facts": "string",
        "input_mode": "string",
        "answer_type": "string",
        "reasoning_family": "string",
        "question_token_count": "int64",
        "selection_similarity": "float64",
        "closest_selected_id": "string"
    }
    
    for item in selected_examples:
        for key in features:
            if key not in item:
                item[key] = None
            elif item[key] is None:
                item[key] = None
            elif key == "question_token_count":
                item[key] = int(item[key])
            elif key == "selection_similarity":
                item[key] = float(item[key])
            else:
                item[key] = str(item[key]) if not isinstance(item[key], (list, dict)) else item[key]
                
    dataset = Dataset.from_list(selected_examples)
    dataset_dict = DatasetDict({"data": dataset})
    
    print(f"Uploading dataset to Hugging Face repository '{repo_id}'...")
    try:
        create_repo(repo_id, token=token, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Repository creation message: {e}")
        
    dataset_dict.push_to_hub(repo_id, token=token, private=private)
    
    # Save local files
    stats_json = {
        "random_seed": 42,
        "source_quotas": {
            "requested": {
                "factual": 120,
                "commonsense": 120,
                "science": 250,
                "arithmetic": 200,
                "multihop": 220,
                "extractive_control": 90
            },
            "achieved": family_counts
        },
        "embedding_model_used": "all-MiniLM-L6-v2",
        "number_candidates_collected": 4000,
        "number_rejected_as_exact_duplicates": collect_stats["exact_duplicates"],
        "number_rejected_as_template_duplicates": collect_stats["template_duplicates"],
        "number_rejected_as_semantic_near_duplicates": semantic_rejections
    }
    
    with open("stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)
        
    with open("rejected_near_duplicates.json", "w", encoding="utf-8") as f:
        json.dump(rejected_pairs, f, indent=2, ensure_ascii=False)
        
    readme_content = f"""---
license: mit
task_categories:
- question-answering
language:
- en
tags:
- reasoning
- math
- science
- commonsense
size_categories:
- 1K<n<10K
---

# Reasoning Spectrum QA Dataset

This is a unified dataset of exactly 1,000 examples designed for evaluating reasoning capabilities across different dimensions (factual, commonsense, science, arithmetic, multi-hop, and extractive span).

## Dataset Purpose
The dataset brings together questions of varying difficulty and reasoning family types under a single unified schema to facilitate standardized testing and evaluation of large language models.

## Source Datasets Used and Distributions
- **Factual**: `stanfordnlp/web_questions` ({family_counts['factual']} selected)
- **Commonsense**: `tau/commonsense_qa` ({family_counts['commonsense']} selected)
- **Science**: `allenai/ai2_arc` (Easy & Challenge) and `allenai/sciq` ({family_counts['science']} selected)
- **Arithmetic**: `openai/gsm8k` ({family_counts['arithmetic']} selected)
- **Multi-hop**: `hotpotqa/hotpot_qa` ({family_counts['multihop']} selected)
- **Extractive Control**: `rajpurkar/squad` ({family_counts['extractive_control']} selected)

## Schema
Each row in the dataset matches the following schema:
- `id`: A unique ID formatted as `qa_mix_XXXXXX`.
- `source_dataset`: The name of the source dataset.
- `source_split`: The split of the source dataset.
- `source_id`: The ID of the question in the source dataset.
- `question`: The cleaned, normalized text of the question.
- `context`: Supporting context or text if needed (e.g. SQuAD, HotpotQA, SciQ support).
- `choices`: Shuffled multiple choice option dictionary (labels and text) or null.
- `gold_answer`: The final target answer text.
- `gold_answer_label`: The correct choice option letter (e.g., "A", "B") or null.
- `reference_solution`: Reasoning explanation/solutions (e.g., from GSM8K) or null.
- `supporting_facts`: Original supporting facts (e.g., from HotpotQA) or null.
- `input_mode`: Input format representation (`question_only`, `question_with_choices`, `question_with_context`, or `question_with_context_and_choices`).
- `answer_type`: The answer format category (`boolean`, `number`, `entity`, `short_phrase`, `multiple_choice`, or `extractive_span`).
- `reasoning_family`: The category of reasoning needed (`factual`, `commonsense`, `science`, `arithmetic`, `multihop`, or `extractive_control`).
- `question_token_count`: Number of tokens in the question computed using the embedding tokenizer.
- `selection_similarity`: Cosine similarity to the closest selected question in the final dataset.
- `closest_selected_id`: ID of the closest selected question in the final dataset.

## Selection and Deduplication Method
1. **Candidate Collection**: Streaming up to 4x the target quota per reasoning family.
2. **Cheap Filters**:
   - Rejection of malformed, non-English, and image/table-based questions.
   - Exact global normalized duplicate rejection.
   - Arithmetic template normalization (placeholder replacement for names, numbers, and pronouns) to reject repeated structural question templates.
   - Frequency limits on named entities in factual questions (max 3 per entity).
3. **Semantic Diversity Selection**:
   - Embeddings computed via `all-MiniLM-L6-v2`.
   - Greedy global max-min cosine similarity selection.
   - Hard rejection of cosine similarity >= 0.90.
   - Heuristic review for cosine similarity between 0.84 and 0.90 (retained only when answers/reasoning differ).

## Limitations
- Source datasets are restricted to English.
- Extractive span boundaries might be source-dependent.
- Simple heuristic parser was used for template and entity matching.

## Source License Notes
- `web_questions`: CC-BY 4.0
- `commonsense_qa`: MIT License
- `ai2_arc`: CC-BY-SA 4.0
- `sciq`: CC-BY-NC 4.0
- `gsm8k`: MIT License
- `hotpot_qa`: CC-BY-SA 4.0
- `squad`: CC-BY-SA 4.0
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    # Upload metadata files to the hub
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj="stats.json",
        path_in_repo="stats.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    api.upload_file(
        path_or_fileobj="rejected_near_duplicates.json",
        path_in_repo="rejected_near_duplicates.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
