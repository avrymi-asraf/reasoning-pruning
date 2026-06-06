# /// script
# dependencies = [
#   "sentence-transformers",
#   "numpy",
#   "tqdm",
# ]
# ///

import random
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from arithmetic_normalizer import clean_text

def should_keep_intermediate_similarity(c1, c2):
    if c1['reasoning_family'] != c2['reasoning_family']:
        return True
    
    ans1 = str(c1['gold_answer']).strip().lower()
    ans2 = str(c2['gold_answer']).strip().lower()
    
    if ans1 != ans2 and ans1 and ans2:
        return True
        
    return False

def select_diverse_subset(
    candidates,
    model_name="all-MiniLM-L6-v2",
    factual_target=120,
    commonsense_target=120,
    science_target=250,
    arithmetic_target=200,
    multihop_target=220,
    extractive_target=90
):
    print("Starting semantic diversity selection...")
    
    family_targets = {
        "factual": factual_target,
        "commonsense": commonsense_target,
        "science": science_target,
        "arithmetic": arithmetic_target,
        "multihop": multihop_target,
        "extractive_control": extractive_target
    }
    
    total_target = sum(family_targets.values())
    
    # Shuffle deterministically
    rng = random.Random(42)
    rng.shuffle(candidates)
    
    # Load model and compute embeddings
    print(f"Loading SentenceTransformer model '{model_name}' and generating embeddings...")
    model = SentenceTransformer(model_name)
    questions = [c["question"] for c in candidates]
    embeddings = model.encode(questions, show_progress_bar=True, batch_size=64)
    
    # Normalize embeddings for dot product cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    
    N = len(candidates)
    
    selected_indices = []
    family_counts = {fam: 0 for fam in family_targets}
    
    S_sim = np.zeros(N) - 1.0
    closest_selected = [-1] * N
    rejected_near_duplicates = []
    
    # Count tokens
    print("Counting question tokens using model tokenizer...")
    for c in candidates:
        tokens = model.tokenizer.tokenize(c["question"])
        c["question_token_count"] = len(tokens)
        
    # Select first element
    first_idx = 0
    selected_indices.append(first_idx)
    family_counts[candidates[first_idx]["reasoning_family"]] += 1
    
    for i in range(N):
        if i == first_idx:
            S_sim[i] = 1.0
            closest_selected[i] = first_idx
        else:
            sim = float(np.dot(embeddings[i], embeddings[first_idx]))
            S_sim[i] = sim
            closest_selected[i] = first_idx
            
    # Selection loop
    for step in tqdm(range(1, total_target), desc="Selecting diverse examples"):
        best_candidate_idx = -1
        best_candidate_sim = 1.0
        
        for i in range(N):
            if i in selected_indices:
                continue
                
            fam = candidates[i]["reasoning_family"]
            if family_counts[fam] >= family_targets[fam]:
                continue
                
            sim = S_sim[i]
            
            # Constraints check
            if sim >= 0.90:
                continue
                
            if 0.84 <= sim < 0.90:
                closest_idx = closest_selected[i]
                if not should_keep_intermediate_similarity(candidates[i], candidates[closest_idx]):
                    continue
            
            if sim < best_candidate_sim:
                best_candidate_sim = sim
                best_candidate_idx = i
                
        if best_candidate_idx == -1:
            # Fallback
            best_fallback_idx = -1
            best_fallback_sim = 1.0
            for i in range(N):
                if i in selected_indices:
                    continue
                fam = candidates[i]["reasoning_family"]
                if family_counts[fam] >= family_targets[fam]:
                    continue
                sim = S_sim[i]
                if sim < best_fallback_sim:
                    best_fallback_sim = sim
                    best_fallback_idx = i
            if best_fallback_idx != -1:
                best_candidate_idx = best_fallback_idx
                best_candidate_sim = best_fallback_sim
            else:
                break
                
        selected_indices.append(best_candidate_idx)
        family_counts[candidates[best_candidate_idx]["reasoning_family"]] += 1
        
        new_emb = embeddings[best_candidate_idx]
        for i in range(N):
            if i in selected_indices:
                continue
            sim_to_new = float(np.dot(embeddings[i], new_emb))
            if sim_to_new > S_sim[i]:
                S_sim[i] = sim_to_new
                closest_selected[i] = best_candidate_idx
                
    selected_examples = []
    for idx in selected_indices:
        selected_examples.append(candidates[idx])
        
    # Assign IDs and globally correct closest selected pairs
    selected_embeddings = embeddings[selected_indices]
    for idx, cand in enumerate(selected_examples):
        cand["id"] = f"qa_mix_{idx+1:06d}"
        
    for idx, cand in enumerate(selected_examples):
        emb_cand = selected_embeddings[idx]
        max_sim = -1.0
        closest_idx = -1
        for jdx in range(len(selected_examples)):
            if idx == jdx:
                continue
            sim = float(np.dot(emb_cand, selected_embeddings[jdx]))
            if sim > max_sim:
                max_sim = sim
                closest_idx = jdx
        cand["selection_similarity"] = max_sim
        cand["closest_selected_id"] = selected_examples[closest_idx]["id"] if closest_idx != -1 else None

    # Track rejections
    semantic_rejections_count = 0
    for i in range(N):
        if i in selected_indices:
            continue
        sim = S_sim[i]
        closest_idx = closest_selected[i]
        if closest_idx != -1:
            closest_cand = candidates[closest_idx]
            if sim >= 0.90 or (0.84 <= sim < 0.90 and not should_keep_intermediate_similarity(candidates[i], closest_cand)):
                semantic_rejections_count += 1
                rejected_near_duplicates.append({
                    "candidate_question": candidates[i]["question"],
                    "candidate_family": candidates[i]["reasoning_family"],
                    "closest_selected_question": closest_cand["question"],
                    "closest_selected_family": closest_cand["reasoning_family"],
                    "similarity": sim
                })

    print(f"Selected {len(selected_examples)} examples. Family counts achieved: {family_counts}")
    return selected_examples, family_counts, semantic_rejections_count, rejected_near_duplicates
