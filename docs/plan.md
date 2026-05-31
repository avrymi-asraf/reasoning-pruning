# The Idea Behind the Project

This project aims to teach a model to reason in a shorter, cleaner, and more efficient way — not by deleting parts of a long answer after it has already been written, but by training the model to skip unnecessary reasoning steps while it is generating the answer.

The central point is that the model is trained on reasoning paths that it produced itself. We do not only take “correct” answers from an external source and train on them. Instead, we let a specific version of the model generate its own reasoning. Then we analyze that reasoning path, identify steps inside it that are unnecessary, and turn those skip points into new training examples. In other words, the model learns to improve from its own reasoning traces.

The goal is not to teach the model to summarize a long answer. This is not training of the form “long output → short output.” Instead, the training focuses on the local transition inside the reasoning process:

“Given the question and the useful reasoning prefix that has already been written, what is the next useful step the model should continue with?”

For example, suppose the model produced this reasoning path:

A → B → C → D

If C is found to be unnecessary, we do not simply save the shortened version A → B → D. Instead, we create a training example that says:

Input: the question + A + B
Target: D

This means the model learns that after A and B, it can continue directly to D without generating C at all.

## How the Data Is Created

The data in this project is not ordinary question-answer data. It is data created from the model’s own reasoning processes. For each example, the model first generates a full or partial reasoning path, and only afterward do we construct a training example from that path that teaches the model which step could have been skipped.

The process starts with an original question or task. We give that question to the current model and ask it to produce a reasoning path. That path is then split into small steps: sentences, reasoning units, or short spans that represent logical progress inside the solution.

Next, we inspect the path and look for the first step that can be removed without breaking the logic of what follows. This distinction matters. The goal is not to find every weak sentence, rank the entire reasoning path, or rewrite the answer. The goal is to find one local skip: a point where the model can move from the useful prefix that has already been written directly to the next useful step, without passing through an unnecessary sentence in the middle.

Once such a step is found, we create a pruning-transition training example. The example is built like this:

The input is the original question together with the useful reasoning prefix that appears before the unnecessary step.

The target is the next useful reasoning step that appears after the removed part.

So if the model wrote:

A → B → C → D

and C is unnecessary, the training example is not “shorten A B C D.” Instead, it is:

Input: the question + A + B
Target: D

This is the most important point in the data creation process: the data does not teach the model to edit itself after the fact. It teaches the model to continue correctly in real time. When the model reaches a similar state in the future, it should learn not to generate C in the first place, and instead continue directly to D.

After the first example is created, the process can continue from the new pruned context. We build a new context where the unnecessary step has already been removed, and we feed that context back into the model. Now the model generates a new continuation from a reasoning path that has already been compressed once. In that new continuation, we again look for the first step that can be safely skipped, and again create a training example:

Input: the question + the useful reasoning path up to the skip point
Target: the next useful step after the skip

This means that a single question can produce several training examples. The first example teaches the first skip. The second example teaches a deeper skip. The next examples continue this process further. Each example represents a different transition point inside reasoning that the model itself produced.

## Important Clarification: What the Data Is Not

A simple hand-written smoke dataset is useful only for validating the mechanical format of the pipeline. It is not the real project.

The real dataset must not be built mainly by manually writing examples or by manually passing a fixed `removable_index` into a helper function. A helper that receives `question`, `reasoning_steps`, and `removable_index` can be useful as a low-level constructor, but it is not the core data-generation logic.

The core data-generation logic is the automatic loop:

1. The current model version generates its own reasoning path.
2. That generated reasoning path is split into reasoning units.
3. A decision model identifies the first safely removable unit or span.
4. The system converts that decision into a pruning-transition row.
5. The pruned context is fed back into the generator.
6. The process repeats to create deeper pruning-transition examples.

So the important object is not merely:

```text
question + manually supplied reasoning_steps + removable_index
```

The important object is:

```text
current model version -> self-generated reasoning path -> automatic pruning decision -> transition example
```

A smoke example may look like:

```text
A -> B -> C -> D
```

and if `C` is removable, the row becomes:

```text
input:  question + A + B
target: D
```

That example is correct as a toy demonstration, but it should not define the real system. In the full system, `A`, `B`, `C`, and `D` must come from the model's own generated reasoning trace, and the decision to remove `C` must come from the pruning decision component, not from a manually written index.

The model is therefore trained on corrected transitions extracted from its own reasoning behavior. This distinction is critical: the project is not simply building a prompt/completion dataset in the abstract. It is building a dataset that teaches the model to improve the way it continues its own reasoning paths.

## What Is Stored in Each Data Example

Each data example should store the minimal information needed to train the model on the correct transition:

* The original question.
* The training input: the question together with a useful prefix from the reasoning path.
* The training target: the next useful step after the skipped part.
* The pruning depth: whether this is the first, second, third, or later example created from the same question.
* Metadata that makes it possible to know which model, which round, and which pruning decision produced the example.

The example itself does not need to contain the full history of the experiment. The main point is that it cleanly represents the transition we want to train:

question + useful prefix → next useful step

## Conceptual Pseudocode for Creating the Data

```text
for each question q in source_questions:
    context = q
    depth = 0

    while pruning_is_still_possible:
        reasoning_path = G.generate(context)

        removable_span = D.find_first_safely_removable_part(
            question=q,
            context=context,
            reasoning_path=reasoning_path
        )

        if no removable_span found:
            break

        useful_prefix = reasoning_path.before(removable_span)
        next_useful_step = reasoning_path.after(removable_span).first_step

        create_training_example(
            input_x = context + useful_prefix,
            target_y = next_useful_step,
            depth = depth
        )

        context = context + useful_prefix + next_useful_step
        depth = depth + 1
```

In simple terms: each time, we let the model continue reasoning, find the first step that can be skipped, and turn that skip into a training example. The example does not tell the model, in a general way, “write shorter.” It shows the model a specific transition point: from here, continue directly to the next step that actually advances the reasoning.

After many such examples are created, the model is trained on them. The training is based on reasoning paths that the model produced itself, but in a corrected form: instead of reinforcing every step the model originally wrote, we reinforce only the transitions where the model could have continued better by skipping an unnecessary part.

After that, the newly trained version of the model can be used to generate new reasoning paths. Because the model has changed, the reasoning paths it produces will also change. From those new paths, we can create new data, train again, and continue the cycle. This creates a research loop in which the model generates reasoning, the data is built from that reasoning, and the model is then trained to improve based on its own reasoning paths.

## Project Requirements and Operating Principles

The project should be built around a clear artifact flow. Every dataset produced by the system must be saved as a Hugging Face dataset. The pruning-transition data should not exist only as loose local files or temporary outputs. Each dataset version should be explicit, reproducible, and connected to the model version, source data, decision configuration, and pruning configuration that produced it.

The model should also be managed as a Hugging Face model. The base model is saved as a Hugging Face model repo, and every accepted training result becomes a new documented version of that model. This makes the model chain traceable: each version should clearly describe which dataset, training configuration, evaluation result, and previous model version led to it.

Training should run as a Hugging Face job. The project should assume that serious training is executed in HF infrastructure, not as an informal local process. Local runs may exist for smoke tests and development, but the main training workflow should be designed around reproducible HF jobs.

The project should use `uv` only. Python environments, dependency management, scripts, and tooling should be based on `uv`, so the repository stays consistent and does not mix multiple package-management styles.

Every meaningful action in the system should go through structured configuration files. Dataset creation, pruning decisions, model selection, training, evaluation, artifact updates, and experiment orchestration should all be driven by explicit config documents. The goal is that important experiment behavior is declared in configuration rather than hidden inside ad-hoc code changes.

At every stage, the project should expose a clean user interface for the available tools. This does not necessarily mean a complex product UI, but there should be an organized interface that makes the core actions easy to access: creating datasets, inspecting examples, launching training, viewing runs, evaluating checkpoints, and promoting a checkpoint into a new model version. The tools should feel discoverable and coherent, not scattered across unrelated scripts.

The project should also maintain continuously updated skills. Since this is an R&D project, the workflow, assumptions, best practices, and lessons learned will change over time. The skills should be treated as living project knowledge: after meaningful work, experiments, failures, or design changes, the relevant skills should be updated so future work benefits from the new understanding.

Weights & Biases should be used to observe the training process. Training runs should log metrics to W&B so progress, instability, regressions, comparisons between runs, and training behavior can be inspected visually. W&B is part of the experiment feedback loop, not just an optional logging add-on.

A critical requirement is to keep the repository simple, direct, and intuitive. The project may be research-heavy, but the repo should not become over-engineered. Each repository should have a clear responsibility, the artifact flow should be easy to understand, commands should map naturally to the research workflow, and configuration should make behavior explicit. The preferred design is boring, readable, and hard to misuse.

The final goal is to test whether reasoning can be improved not only by training on good final answers, but by training on the small transitions inside the reasoning process: where to continue, what to skip, and how to reach the next useful step without wasting reasoning on parts that do not actually help.
