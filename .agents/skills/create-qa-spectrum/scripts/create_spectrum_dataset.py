# /// script
# dependencies = [
#   "datasets",
#   "huggingface_hub",
#   "sentence-transformers",
#   "numpy",
#   "tqdm",
#   "scipy",
# ]
# ///

import os
import sys
import json
import argparse
from candidate_collector import collect_candidates
from semantic_selector import select_diverse_subset
from dataset_uploader import validate_dataset, upload_dataset_to_hub

def cmd_collect(args):
    candidates, collect_stats = collect_candidates(
        factual_limit=args.factual_limit,
        commonsense_limit=args.commonsense_limit,
        science_limit=args.science_limit,
        arithmetic_limit=args.arithmetic_limit,
        multihop_limit=args.multihop_limit,
        extractive_limit=args.extractive_limit
    )
    # Write output candidates
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"candidates": candidates, "stats": collect_stats}, f, indent=2, ensure_ascii=False)
    print(f"Success! Collected candidates written to: {args.output}")

def cmd_select(args):
    # Load candidates
    with open(args.candidates, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidates = data["candidates"]
    collect_stats = data["stats"]
    
    selected, family_counts, semantic_rejections, rejected_pairs = select_diverse_subset(
        candidates,
        model_name=args.model_name,
        factual_target=args.factual_target,
        commonsense_target=args.commonsense_target,
        science_target=args.science_target,
        arithmetic_target=args.arithmetic_target,
        multihop_target=args.multihop_target,
        extractive_target=args.extractive_target
    )
    
    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "selected_examples": selected,
            "family_counts": family_counts,
            "collect_stats": collect_stats,
            "semantic_rejections": semantic_rejections,
            "rejected_pairs": rejected_pairs
        }, f, indent=2, ensure_ascii=False)
    print(f"Success! Selected dataset written to: {args.output}")

def cmd_validate(args):
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected_examples = data["selected_examples"]
    validate_dataset(selected_examples)
    print("Success! Dataset passed all validation checks.")

def cmd_upload(args):
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected_examples = data["selected_examples"]
    family_counts = data["family_counts"]
    collect_stats = data["collect_stats"]
    semantic_rejections = data["semantic_rejections"]
    rejected_pairs = data["rejected_pairs"]
    
    # Validate before upload
    validate_dataset(selected_examples)
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable is not set. Cannot upload.")
        sys.exit(1)
        
    upload_dataset_to_hub(
        selected_examples=selected_examples,
        family_counts=family_counts,
        collect_stats=collect_stats,
        semantic_rejections=semantic_rejections,
        rejected_pairs=rejected_pairs,
        repo_id=args.repo_id,
        token=token,
        private=not args.public
    )
    print(f"Success! Dataset uploaded to Hugging Face Hub under repository: {args.repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Diverse QA Spectrum Dataset Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Collect subcommand
    parser_collect = subparsers.add_parser("collect", help="Stream candidates and apply cheap filtering")
    parser_collect.add_argument("--output", required=True, help="Path to write candidates JSON output")
    parser_collect.add_argument("--factual-limit", type=int, default=480)
    parser_collect.add_argument("--commonsense-limit", type=int, default=480)
    parser_collect.add_argument("--science-limit", type=int, default=1000)
    parser_collect.add_argument("--arithmetic-limit", type=int, default=800)
    parser_collect.add_argument("--multihop-limit", type=int, default=880)
    parser_collect.add_argument("--extractive-limit", type=int, default=360)
    
    # Select subcommand
    parser_select = subparsers.add_parser("select", help="Run semantic diversity greedy selection")
    parser_select.add_argument("--candidates", required=True, help="Path to candidates JSON file")
    parser_select.add_argument("--output", required=True, help="Path to write selected JSON output")
    parser_select.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser_select.add_argument("--factual-target", type=int, default=120)
    parser_select.add_argument("--commonsense-target", type=int, default=120)
    parser_select.add_argument("--science-target", type=int, default=250)
    parser_select.add_argument("--arithmetic-target", type=int, default=200)
    parser_select.add_argument("--multihop-target", type=int, default=220)
    parser_select.add_argument("--extractive-target", type=int, default=90)
    
    # Validate subcommand
    parser_validate = subparsers.add_parser("validate", help="Validate selected dataset schema and constraints")
    parser_validate.add_argument("--dataset-file", required=True, help="Path to selected JSON file")
    
    # Upload subcommand
    parser_upload = subparsers.add_parser("upload", help="Upload selected dataset to Hugging Face Hub")
    parser_upload.add_argument("--dataset-file", required=True, help="Path to selected JSON file")
    parser_upload.add_argument("--repo-id", required=True, help="Hugging Face target repo ID (namespace/repo)")
    parser_upload.add_argument("--public", action="store_true", help="Make repository public (default: private)")
    
    args = parser.parse_args()
    
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "select":
        cmd_select(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "upload":
        cmd_upload(args)

if __name__ == "__main__":
    main()
