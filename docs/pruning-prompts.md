# Decision prompts

## Active prompt: `incremental-skip-v2`

The active decision prompt is designed for short incremental-generation batches. It asks D to select only a filler span that has an immediately following useful target inside the supplied batch, and it explicitly forbids replacement text. It also allows repeated question facts to be classified as filler even when they contain names or numbers, while keeping computations, derived facts, and deductions.

The prompt lives at `prompts/incremental-skip-v2.txt`. Dataset-builder configs and the Hugging Face Jobs fallback config must reference the same prompt version.

## Previous prompt: `conservative-skip-v1`

`conservative-skip-v1` remains available for replaying older cached traces, but it was written for longer traces and is no longer the active dataset-creation prompt.
