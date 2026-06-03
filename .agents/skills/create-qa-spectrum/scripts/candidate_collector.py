# /// script
# dependencies = [
#   "datasets",
# ]
# ///

import ast
import json
import random
from datasets import load_dataset
from arithmetic_normalizer import clean_text, get_arithmetic_template

def collect_candidates(
    factual_limit=480,
    commonsense_limit=480,
    science_limit=1000,
    arithmetic_limit=800,
    multihop_limit=880,
    extractive_limit=360
):
    print("Starting candidate collection...")
    candidates = []
    
    # Rejection statistics tracking
    stats = {
        "exact_duplicates": 0,
        "template_duplicates": 0,
        "entity_duplicates": 0,
        "invalid_or_malformed": 0
    }
    
    global_seen_questions = set()
    seen_arith_templates = set()
    factual_entity_counts = {}
    
    family_counts = {
        "factual": 0,
        "commonsense": 0,
        "science": 0,
        "arithmetic": 0,
        "multihop": 0,
        "extractive_control": 0
    }
    
    # ------------------ FACTUAL (WebQuestions) ------------------
    print("Collecting Factual candidates from web_questions...")
    webq_stream = load_dataset("stanfordnlp/web_questions", split="train", streaming=True)
    for item in webq_stream:
        if family_counts["factual"] >= factual_limit:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        # Parse entity from Freebase URL
        url = item.get("url", "")
        entity = url.rstrip('/').split('/')[-1] if url else "unknown"
        if entity not in factual_entity_counts:
            factual_entity_counts[entity] = 0
            
        # Entity duplicate check (max 3 per entity)
        if factual_entity_counts[entity] >= 3:
            stats["entity_duplicates"] += 1
            continue
            
        # Parse answers
        answers = item.get("answers", [])
        if isinstance(answers, str):
            try:
                answers = ast.literal_eval(answers)
            except:
                answers = [answers]
        gold_answer = answers[0] if answers else ""
        
        cand = {
            "source_dataset": "web_questions",
            "source_split": "train",
            "source_id": entity,
            "question": q,
            "context": None,
            "choices": None,
            "gold_answer": gold_answer,
            "gold_answer_label": None,
            "reference_solution": None,
            "supporting_facts": None,
            "input_mode": "question_only",
            "answer_type": "short_phrase",
            "reasoning_family": "factual"
        }
        
        # Cheap validations
        if not q or not gold_answer or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        factual_entity_counts[entity] += 1
        candidates.append(cand)
        family_counts["factual"] += 1
        
    print(f"Collected {family_counts['factual']} Factual candidates.")
    
    # ------------------ COMMONSENSE (CommonsenseQA) ------------------
    print("Collecting Commonsense candidates from commonsense_qa...")
    cqa_stream = load_dataset("tau/commonsense_qa", split="train", streaming=True)
    for item in cqa_stream:
        if family_counts["commonsense"] >= commonsense_limit:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        choices_raw = item.get("choices", {})
        labels = choices_raw.get("label", [])
        texts = choices_raw.get("text", [])
        choices = [{"label": l, "text": t} for l, t in zip(labels, texts)]
        
        ans_key = item.get("answerKey", "")
        gold_answer = ""
        if ans_key in labels:
            gold_answer = texts[labels.index(ans_key)]
            
        cand = {
            "source_dataset": "commonsense_qa",
            "source_split": "train",
            "source_id": item.get("id"),
            "question": q,
            "context": None,
            "choices": choices,
            "gold_answer": gold_answer,
            "gold_answer_label": ans_key,
            "reference_solution": None,
            "supporting_facts": None,
            "input_mode": "question_with_choices",
            "answer_type": "multiple_choice",
            "reasoning_family": "commonsense"
        }
        
        if not q or not gold_answer or not choices or not ans_key or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        candidates.append(cand)
        family_counts["commonsense"] += 1
        
    print(f"Collected {family_counts['commonsense']} Commonsense candidates.")
    
    # ------------------ SCIENCE (ARC + SciQ) ------------------
    print("Collecting Science candidates...")
    
    # We allocate science candidates space based on science_limit
    arc_easy_cap = int(science_limit * 0.4)
    arc_chal_cap = int(science_limit * 0.2)
    sciq_cap = science_limit - arc_easy_cap - arc_chal_cap
    
    # 1. ARC-Easy
    arc_easy_candidates = 0
    print("Streaming ARC-Easy...")
    arc_easy_stream = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train", streaming=True)
    for item in arc_easy_stream:
        if arc_easy_candidates >= arc_easy_cap:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        choices_raw = item.get("choices", {})
        labels = choices_raw.get("label", [])
        texts = choices_raw.get("text", [])
        choices = [{"label": l, "text": t} for l, t in zip(labels, texts)]
        
        ans_key = item.get("answerKey", "")
        gold_answer = ""
        if ans_key in labels:
            gold_answer = texts[labels.index(ans_key)]
            
        cand = {
            "source_dataset": "ai2_arc_easy",
            "source_split": "train",
            "source_id": item.get("id"),
            "question": q,
            "context": None,
            "choices": choices,
            "gold_answer": gold_answer,
            "gold_answer_label": ans_key,
            "reference_solution": None,
            "supporting_facts": None,
            "input_mode": "question_with_choices",
            "answer_type": "multiple_choice",
            "reasoning_family": "science"
        }
        
        if not q or not gold_answer or not choices or not ans_key or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        candidates.append(cand)
        arc_easy_candidates += 1
        family_counts["science"] += 1
        
    # 2. ARC-Challenge
    arc_chal_candidates = 0
    print("Streaming ARC-Challenge...")
    arc_chal_stream = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True)
    for item in arc_chal_stream:
        if arc_chal_candidates >= arc_chal_cap:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        choices_raw = item.get("choices", {})
        labels = choices_raw.get("label", [])
        texts = choices_raw.get("text", [])
        choices = [{"label": l, "text": t} for l, t in zip(labels, texts)]
        
        ans_key = item.get("answerKey", "")
        gold_answer = ""
        if ans_key in labels:
            gold_answer = texts[labels.index(ans_key)]
            
        cand = {
            "source_dataset": "ai2_arc_challenge",
            "source_split": "train",
            "source_id": item.get("id"),
            "question": q,
            "context": None,
            "choices": choices,
            "gold_answer": gold_answer,
            "gold_answer_label": ans_key,
            "reference_solution": None,
            "supporting_facts": None,
            "input_mode": "question_with_choices",
            "answer_type": "multiple_choice",
            "reasoning_family": "science"
        }
        
        if not q or not gold_answer or not choices or not ans_key or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        candidates.append(cand)
        arc_chal_candidates += 1
        family_counts["science"] += 1
        
    # 3. SciQ
    sciq_candidates = 0
    print("Streaming SciQ...")
    sciq_stream = load_dataset("allenai/sciq", split="train", streaming=True)
    for item in sciq_stream:
        if sciq_candidates >= sciq_cap or family_counts["science"] >= science_limit:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        choices_texts = [item["distractor1"], item["distractor2"], item["distractor3"], item["correct_answer"]]
        rng = random.Random(hash(q) + 42)
        rng.shuffle(choices_texts)
        labels = ["A", "B", "C", "D"]
        choices = [{"label": l, "text": t} for l, t in zip(labels, choices_texts)]
        gold_label = labels[choices_texts.index(item["correct_answer"])]
        
        context = clean_text(item.get("support", ""))
        
        cand = {
            "source_dataset": "sciq",
            "source_split": "train",
            "source_id": None,
            "question": q,
            "context": context if context else None,
            "choices": choices,
            "gold_answer": item["correct_answer"],
            "gold_answer_label": gold_label,
            "reference_solution": None,
            "supporting_facts": None,
            "input_mode": "question_with_context_and_choices" if context else "question_with_choices",
            "answer_type": "multiple_choice",
            "reasoning_family": "science"
        }
        
        if not q or not item["correct_answer"] or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        candidates.append(cand)
        sciq_candidates += 1
        family_counts["science"] += 1
        
    print(f"Collected {family_counts['science']} Science candidates (ARC-Easy: {arc_easy_candidates}, ARC-Challenge: {arc_chal_candidates}, SciQ: {sciq_candidates}).")
    
    # ------------------ ARITHMETIC (GSM8K) ------------------
    print("Collecting Arithmetic candidates from gsm8k...")
    gsm_stream = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    for item in gsm_stream:
        if family_counts["arithmetic"] >= arithmetic_limit:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        template = get_arithmetic_template(q)
        if template in seen_arith_templates:
            stats["template_duplicates"] += 1
            continue
            
        ans_raw = item.get("answer", "")
        parts = ans_raw.split("####")
        gold_answer = parts[1].strip() if len(parts) > 1 else ans_raw.strip()
        ref_sol = parts[0].strip() if len(parts) > 1 else None
        
        cand = {
            "source_dataset": "gsm8k",
            "source_split": "train",
            "source_id": None,
            "question": q,
            "context": None,
            "choices": None,
            "gold_answer": gold_answer,
            "gold_answer_label": None,
            "reference_solution": ref_sol,
            "supporting_facts": None,
            "input_mode": "question_only",
            "answer_type": "number",
            "reasoning_family": "arithmetic"
        }
        
        if not q or not gold_answer or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        seen_arith_templates.add(template)
        candidates.append(cand)
        family_counts["arithmetic"] += 1
        
    print(f"Collected {family_counts['arithmetic']} Arithmetic candidates.")
    
    # ------------------ MULTIHOP (HotpotQA) ------------------
    print("Collecting Multi-hop candidates from hotpot_qa...")
    hotpot_stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
    for item in hotpot_stream:
        if family_counts["multihop"] >= multihop_limit:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        context_parts = []
        for title, sents in zip(titles, sentences_list):
            context_parts.append(f"{title}: " + "".join(sents))
        context_str = "\n".join(context_parts)
        
        supp_facts = {
            "title": item["supporting_facts"]["title"],
            "sent_id": item["supporting_facts"]["sent_id"]
        }
        supp_facts_str = json.dumps(supp_facts)
        
        gold_answer = item.get("answer", "").strip()
        ans_type = "boolean" if gold_answer.lower() in ["yes", "no"] else "short_phrase"
        
        cand = {
            "source_dataset": "hotpot_qa",
            "source_split": "train",
            "source_id": item.get("id"),
            "question": q,
            "context": context_str,
            "choices": None,
            "gold_answer": gold_answer,
            "gold_answer_label": None,
            "reference_solution": None,
            "supporting_facts": supp_facts_str,
            "input_mode": "question_with_context",
            "answer_type": ans_type,
            "reasoning_family": "multihop"
        }
        
        if not q or not gold_answer or not context_str or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        candidates.append(cand)
        family_counts["multihop"] += 1
        
    print(f"Collected {family_counts['multihop']} Multi-hop candidates.")
    
    # ------------------ EXTRACTIVE (SQuAD) ------------------
    print("Collecting Extractive candidates from squad...")
    squad_stream = load_dataset("rajpurkar/squad", split="train", streaming=True)
    for item in squad_stream:
        if family_counts["extractive_control"] >= extractive_limit:
            break
            
        q = clean_text(item["question"])
        q_norm = q.lower()
        if q_norm in global_seen_questions:
            stats["exact_duplicates"] += 1
            continue
            
        answers_raw = item.get("answers", {})
        texts = answers_raw.get("text", [])
        gold_answer = texts[0] if texts else ""
        
        cand = {
            "source_dataset": "squad",
            "source_split": "train",
            "source_id": item.get("id"),
            "question": q,
            "context": clean_text(item.get("context", "")),
            "choices": None,
            "gold_answer": gold_answer,
            "gold_answer_label": None,
            "reference_solution": None,
            "supporting_facts": None,
            "input_mode": "question_with_context",
            "answer_type": "extractive_span",
            "reasoning_family": "extractive_control"
        }
        
        if not q or not gold_answer or not cand["context"] or any(kw in q_norm for kw in ["image", "table", "chart", "figure", "diagram", "picture", "photo", "http://", "https://"]):
            stats["invalid_or_malformed"] += 1
            continue
            
        global_seen_questions.add(q_norm)
        candidates.append(cand)
        family_counts["extractive_control"] += 1
        
    print(f"Collected {family_counts['extractive_control']} Extractive candidates.")
    print(f"Total candidates in pool: {len(candidates)}")
    return candidates, stats
