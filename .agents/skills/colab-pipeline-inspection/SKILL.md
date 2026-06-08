---
name: colab-pipeline-inspection
description: Use when running pipeline inspection with the real Gemma-4 fine-tune on Colab GPU — iterating D prompts, testing unit-split strategies, or any inspection that requires the full model (not the Gemma-4-API proxy used by scripts/pipeline_inspection.py).
---

# Colab Pipeline Inspection

The local `scripts/pipeline_inspection.py` uses the Gemma-4-API proxy (fast, cheap, inspection-only). For real G — `avreymi/gemma-4-E2B-it-reasoning-pruning` on GPU — use **colab-cli** to provision a T4 session, load the model once, then iterate live without restarting.

**colab-mcp** (`open_colab_browser_connection`) requires an active browser Colab session — it cannot be used by agents operating headlessly. Use colab-cli only.

## Critical Constraints

| Rule | Why |
|------|-----|
| `--timeout 600` for model load | Default `exec` timeout is 10 s; model takes 3-5 min |
| Never use `colab repl` / `colab console` interactively | Both require a TTY and hang in agent context |
| Pipe stdin for all iterative code: `echo "..." \| colab exec` | Only mode that works headlessly |
| Kernel state persists across `exec` calls | Load G once; rebuild D fast with stdin snippets |
| `colab upload` requires parent dir to exist on VM | Create dir first via kernel exec if needed |
| Always `colab stop` when done | Idle VMs burn compute units |

## Phase 1: Setup (one-time, ~7 min)

```bash
# Provision T4 GPU session
colab new -s inspect --gpu T4

# Inject secrets from local env (never hardcode)
echo "import os; os.environ['HF_TOKEN']='${HF_TOKEN}'; os.environ['GEMINI_API_KEY']='${GEMINI_API_KEY}'" \
  | colab exec -s inspect --timeout 30

# Run setup script — clones repo, installs deps, loads G + config + questions + D
# After this finishes, kernel holds: generator, config, questions, decision_model, run_pipeline_inspection
colab exec -s inspect -f scripts/colab_inspect_setup.py --timeout 600
```

**Verify setup completed:**
```bash
echo "print(f'G={generator.source_model}, Q={len(questions)}, D-prompt={config.decision[\"prompt_version\"]}')" \
  | colab exec -s inspect --timeout 15
```

## Phase 2: Iterative Inspection (fast, no model reload)

### Run inspection on a question

```bash
mkdir -p output/pipeline_inspection
echo "
rows = run_pipeline_inspection(
    question=questions[3],
    generator=generator,
    decision_model=decision_model,
    config=config,
)
" | colab exec -s inspect --timeout 180 | tee output/pipeline_inspection/q3_base.txt
```

### Swap D prompt and re-run

```bash
# Edit prompt locally, then upload (parent dir already exists from git clone)
colab upload -s inspect prompts/my-new-prompt.txt /content/reasoning-pruning/prompts/my-new-prompt.txt

# Rebuild D — fast, no model reload
echo "
config.decision['prompt_version'] = 'my-new-prompt'
decision_model = create_decision_model_from_config(
    config.decision, config.pruning, prompts_dir='/content/reasoning-pruning/prompts'
)
print('D ready:', config.decision['prompt_version'])
" | colab exec -s inspect --timeout 30

# Run inspection and capture
echo "
rows = run_pipeline_inspection(question=questions[3], generator=generator, decision_model=decision_model, config=config)
" | colab exec -s inspect --timeout 180 | tee output/pipeline_inspection/q3_my-new-prompt.txt
```

### Change unit-split strategy

```bash
echo "
from dataclasses import replace
config = replace(config, unit_split_strategy='clauses')
print('unit_split_strategy:', config.unit_split_strategy)
" | colab exec -s inspect --timeout 15
```

### Run multiple questions

```bash
echo "
for qi in [1, 3, 5, 7]:
    print(f'\n=== Q{qi} ===')
    run_pipeline_inspection(question=questions[qi], generator=generator, decision_model=decision_model, config=config)
" | colab exec -s inspect --timeout 600 | tee output/pipeline_inspection/multi_run.txt
```

## Phase 3: Save Output

Stdout from `colab exec` is the inspection output — always `tee` to `output/pipeline_inspection/`.

To download a file written on the VM:
```bash
colab download -s inspect /content/reasoning-pruning/output/result.json output/pipeline_inspection/result.json
```

Export full session history as markdown:
```bash
colab log -s inspect -o output/pipeline_inspection/session.md
```

## Phase 4: Cleanup

```bash
colab stop -s inspect
```

## Running the Notebook Headlessly

The notebook (`notebooks/data_creation_playground.ipynb`) is headless-safe after the secrets fix — it checks env vars before calling Colab userdata. To run it instead of the setup script:

```bash
# Set secrets first (same as above), then:
colab exec -s inspect -f notebooks/data_creation_playground.ipynb --timeout 600
```

This runs all notebook cells in order. Prefer the setup script for clean agent use; use the notebook when you want to match the human interactive workflow exactly.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `exec` times out immediately | Add `--timeout 600` — default is 10 s |
| "Session not found" | `colab sessions` to check; re-run Phase 1 if pruned |
| `repl` / `console` hangs | Needs TTY — always pipe stdin instead |
| Kernel deadlocked | `colab restart-kernel -s inspect`; re-run model-load block |
| Upload 500 error | Parent dir doesn't exist on VM — create it first via exec |
| `create_decision_model_from_config` not defined | Run setup or import it: `echo "from reasoning_pruning.clients import create_decision_model_from_config" \| colab exec` |
| G produces wrong output | Verify HF_TOKEN: `echo "import os; print(os.environ.get('HF_TOKEN','MISSING')[:8])" \| colab exec` |
