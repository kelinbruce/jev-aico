# Nimble overview image

Generated using the built-in image generation tool, with two targeted edits for an opaque background and accurate footer text.

Sources: repository README, `docs/PARALLEL_SCORING.md`, and `docs/MODAL_SERVING.md`. The serving panel depicts the SGLang implementation, rather than the independent full-prompt CUDA scorer. The curation example is conceptual.

## Prompt

Create one polished technical infographic about Nimble, suitable for a repository README and presentation. Use case: infographic-diagram. Landscape aspect ratio about 16:9, high resolution, crisp and perfectly readable typesetting. Clean editorial visual style: warm off-white background, deep ink navy text, restrained warm orange accents, muted teal for valid outcomes, generous spacing, precise thin arrows, flat geometric diagrams. No logo, no N monogram, no decorative AI brain, no robot, no gradients or 3D. The word Nimble is a text title only.

Main title: "How Nimble works"
Subtitle: "From contrastive examples to fast, typed decisions"

Composition: Three connected substantial panels left to right, numbered 01, 02, 03. The third serving panel should be the widest to clearly display a branching diagram. Left panel flows with an arrow into middle training panel; middle model flows into right serving panel. Use diagrams and sparse concise text, not dense prose. Place a shared strip of outputs and uses along the bottom. Clearly distinguish offline data/training from online serving with a small label above the panels.

LEFT PANEL exact heading: "Data"
Second heading: "Contrastive Data Curation"
Draw two miniature evidence cards beside or above each other sharing a rule:
"Only Mira can approve this refund."
Card A: "Sole signer: Mira" and "Authorized: TRUE"
Card B: "Sole signer: Noah" and "Authorized: FALSE"
Highlight only Mira vs Noah in the sole signer rows as the changed fact.
Below cards exact caption: "Same rule + same question. One key fact changes."
A small checking row: "Verify both examples"
Under it small text: "Remove necessary evidence → answer becomes unknown"
Then final small line: "Keep each pair in the same data split"
This conveys two jointly necessary facts: who has authority and who signed. The unknown outcome is a validation check, not another FALSE training label.

MIDDLE PANEL exact heading: "Training"
Second heading: "Learn the decision"
Vertical clean pipeline with boxes:
"Qwen3.5-9B"
then downward arrow
"LoRA fine-tuning"
then downward arrow
"Nimble"
Two short supporting lines: "Train on allowed-answer logits" and "Supervise with checked labels"
Small pill at bottom: "One-token answer codes"
This is hard-label candidate cross-entropy training, do not claim distillation, do not add other model names or performance numbers.

RIGHT PANEL exact heading: "Serving"
Second heading: "Parallel constrained decoding"
Show a wide top input capsule "Text + questions + allowed answers", flowing down into one shared box "Shared context prefill". From this box draw an explicit fork into THREE parallel separate lanes.
Lane 1 header: "Choice", small candidate chips "billing | support"
Lane 2 header: "Boolean", small candidate chips "false | true"
Lane 3 header: "Rating", small candidate chips "low | medium | high"
Each lane has its own small block "Score allowed codes". The lanes must visibly run in parallel, and none feeds into another lane.
Merge these lanes with arrows into a wide output box labeled "Typed answers + candidate probabilities".
Under that output box in small concise text: "Independent questions • No generated explanation"
Small implementation note inside serving panel bottom: "SGLang: shared-prefix warmup + one token per question"
This diagram describes the SGLang serving implementation with shared prefix warmup and concurrent independent constrained field scoring. Avoid any claim of zero generated tokens. The warmup can emit one discarded token; no need to show that detail beyond the accurate implementation note. No generated free-form JSON string is required; code assembles typed outputs. Do not imply fields see each other's answers.

Bottom full-width strip:
label "Build with it"
four separated items with modest line icons: "Route requests", "Verify conditions", "Apply policies", "Rate outcomes"
Small footer: "Text input • Defined answer spaces • Probabilities over supplied candidates"

Make the causal flow and parallel fan-out the central visual. Ensure every bit of text is legible with short line lengths, consistent typography, sufficient padding, no clipped content. Use all text accurately, but prioritize clean readable layout over excessive decoration.

## Background correction

Edit this infographic only to fix its background and readability. Preserve every existing word, all diagrams, panel positions, arrows, colors, dimensions, and overall layout. Place the entire image on one completely opaque solid warm off-white background (#F5F4EE). There must be NO transparent pixels, black background, checkerboard, or cutout regions anywhere. Restore clean smooth anti-aliased edges on the title, subtitle, offline/online headings, panel outlines, connector arrows, footer text and footer icons against the opaque off-white background. The title and footer must be fully readable dark navy text on the off-white background. Preserve the contents of all three panels exactly. Return a complete rectangular opaque infographic with all original content.

## Final footer correction

Make one tiny text correction to this completed infographic. In the small footer line at the very bottom, replace the entire existing sentence with exactly: "Text input • Defined answer spaces • Candidate probabilities". Use dark navy readable text. Do not use the phrase user-aligned. Preserve ALL other pixels, words, diagrams, positions, colors, dimensions and layout as closely as possible. Keep the entire warm off-white background opaque, with no transparency. No other changes.
