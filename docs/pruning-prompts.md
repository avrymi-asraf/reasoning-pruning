conservative-skip-v1
```text
Decision prompt version: {prompt_version}

You are a conservative pruning decision model. Your job: find the first reasoning unit that is pure filler and can be removed without any loss of correctness.

STRICT REMOVAL CONDITIONS — all must hold:
1. The unit contains NO computation, NO numeric value, NO logical deduction, and NO new fact. It is pure filler (e.g. a numbering artifact, a commentary, a restatement of the problem, or a statement of intent).
2. The unit at index removed_end_index+1 — which becomes the training target — contains ACTUAL reasoning: a numeric computation, a derived fact, or a logical deduction. It must NOT be another goal/intent statement ('Determine X', 'We need to find Y', 'Calculate Z', 'Convert A to B').
3. Removing the span leaves the reasoning coherent.

REMOVABLE examples: '1.' (bare numbering), 'Let me think.' (filler), 'This follows the standard approach.' (commentary).
NOT REMOVABLE: 'Convert 50 minutes to hours.' (goal statement — not a computation), 'Determine the rate per minute.' (intent, not math), '$12 × 50/60 = $10.' (actual computation — keep it), '50/60 = 5/6 hours.' (actual math — keep it).

If the next unit after the candidate removal is itself a goal/intent statement, set has_removal=false.

Question:
{question}

Current context:
{context}

Reasoning units:
{reasoning_units}

Return only JSON: has_removal (bool), removed_start_index (int), removed_end_index (int), reason (string), can_continue_after_skip (bool).
Set can_continue_after_skip=true only when the unit at removed_end_index+1 contains actual math, logic, or a concrete fact — never a goal or intent statement.
```

this prompt not work well. for example:
for:
"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
The reasoning units are:
```json
[
  "Natalia sold 48 clips in April.",
  "Natalia sold half as many clips in May.",
  "Number of clips sold in May is $48 \\div 2$.",
  "Number of clips sold in May is 24.",
  "Total clips sold is the sum of clips sold in April and May.",
  "Total clips sold is $48 + 24$.",
  "Total clips sold is 72."
]
```
the model D return only as targetY the last unit: "Total clips sold is $48 + 24$."


or:
"Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
```json
[
  "Identify the hourly rate: Weng earns $\\$12$ per hour.",
  "Identify the time worked in minutes: Weng worked for $50$ minutes.",
  "Convert the time worked from minutes to hours: Divide the minutes worked by $60$ minutes per hour.",
  "$$50 \\div 60 = 50/60 \\text{ hours}$$",
  "Simplify the fraction for the time worked in hours: Divide both the numerator and the denominator by their greatest common divisor, which is $10$.",
  "$$50 \\div 10 = 5$$",
  "$$60 \\div 10 = 6$$",
  "$$\\text{Time worked} = 5/6 \\text{ hours}$$",
  "Calculate the earnings by multiplying the hourly rate by the time worked in hours: Multiply $\\$12$ by $5/6$.",
  "$$\\text{Earnings} = \\$12 \\times \\frac{5}{6}$$",
  "Perform the multiplication:",
  "$$\\text{Earnings} = \\frac{12 \\times 5}{6}$$",
  "$$\\text{Earnings} = \\frac{60}{6}$$",
  "Perform the division:",
  "$$\\text{Earnings} = 10$$",
  "State the final earning amount: Weng earned $\\$10$."
]
```


the targety is:
"""Convert the time worked from minutes to hours: Divide the minutes worked by $60$ minutes per hour."""
This is a big and illogical leap.
You need to change it to something more plausible. Let's say:
for first example: 'instad of ""Identify the hourly rate: Weng earns $\\$12$ per hour.", we can write: "hourly rate:" "$\\$12$"
this is small change, and this is what I want to train the model to do.
so according to how we create the data it need to be:
example 1:
```json
[
  "x": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
  "y": "hourly rate"
]
```

then send it to the model. and revive the output. 
but what that happens is that the model D returns "hourly rate: $\\$12$ per hour." which is not what we want. We want it to return just """Convert the time worked from minutes to hours: Divide the minutes worked by $60$ minutes per hour."""
this is a big and illogical leap, and we miss the all point of creteating the data.



