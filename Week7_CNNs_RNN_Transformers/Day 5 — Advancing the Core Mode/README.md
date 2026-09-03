# Banking Chatbot — DistilBERT Intent Classifier (Banking77)

An intent-classification model for a banking chatbot: given a customer's message, it predicts which of 77 banking-related intents the message belongs to (e.g. `pin_blocked`, `card_arrival`, `wrong_amount_of_cash_received`), so a chatbot can route it to the right automated response.

This README documents what changed between yesterday's version of the notebook and the updated one (`Banking_Chatbot_with_DistilBERT_Improved.ipynb`), and why.

## The problems in the previous version

The previous notebook's own notes flagged these issues:

1. High loss on both training and testing, with no diagnosis of why.
2. A validation set was created but never actually used to pick the best model.
3. No loss or accuracy curves — impossible to see *how* training progressed, only the final number.
4. The model was tested on a single handwritten sentence ("I forgot my PIN") and got it wrong (predicted `get_physical_card` instead of a PIN-related intent), with no way to tell if this was a one-off or a systemic weakness.
5. No mechanism for the model to say "I don't know" on unclear or out-of-scope text — it always returns its top guess, however unconfident.

## What changed, and why it helps

### 1. Training configuration (reduces loss, improves accuracy)

| Setting | Before | Now | Why |
|---|---|---|---|
| Epochs | 3 (fixed) | up to 8, with **early stopping** (patience = 2 epochs on validation macro-F1) | Three epochs was an arbitrary guess. Early stopping lets the model train as long as it's actually improving, and automatically rolls back to the best checkpoint instead of overfitting in later epochs. |
| LR schedule | Constant `2e-5` | `3e-5` with **warmup (10%) + cosine decay** | A constant learning rate applied immediately to a freshly-initialized classification head on top of a pretrained transformer is a common cause of unstable early loss. Warmup ramps the LR up gradually; cosine decay then eases it down smoothly, which typically produces a lower, more stable final loss. |
| Regularization | Weight decay only | Weight decay **+ label smoothing (0.1)** + slightly higher classifier dropout (0.2 vs. default 0.1) | Label smoothing stops the model from being trained toward 100% confidence on the correct label, which reduces overfitting and generally improves both loss and generalization in classification tasks with many similar classes. The extra dropout further discourages over-reliance on any single feature — helpful with 77 fine-grained, sometimes overlapping intents. |
| Precision | fp32 | fp16 automatically when a GPU is available | Faster training with no meaningful accuracy cost, which also makes it cheaper to run the extra epochs above. |

### 2. Actually using the validation set

- `eval_strategy="epoch"` + `metric_for_best_model="macro_f1"` + `load_best_model_at_end=True`: the model is evaluated on the validation set after every epoch, and the checkpoint that is kept at the end is whichever epoch scored best on validation — not simply the last epoch trained.
- **Macro F1** (not just accuracy) is used to pick the best model, because with 77 classes it's easy for overall accuracy to look fine while a handful of intents are being predicted poorly. Macro F1 weighs every intent equally, so it surfaces that kind of problem.
- The train/validation split is now **stratified by label**, so every one of the 77 intents is represented proportionally in the validation set — with a plain random split, some rare intents could end up with very few (or zero) validation examples, making per-epoch evaluation unreliable.

### 3. The test set is touched exactly once

The previous notebook informally checked results against test data. In the updated notebook, the test set is only evaluated **after** the best model has already been selected using the validation set — a single call in Step 13. This is what makes the reported test accuracy/F1 trustworthy: nothing about the model or its hyperparameters was chosen by looking at test performance.

### 4. Visualizations (previously missing entirely)

- **Loss curves**: training loss vs. validation loss, per epoch — makes it possible to see overfitting directly (validation loss turning upward while training loss keeps falling) instead of inferring it from a single final number.
- **Accuracy / F1 curves**: validation accuracy, weighted F1, and macro F1 over the course of training.
- **Confusion-pair analysis**: instead of a 77×77 confusion matrix (unreadable at that size), the notebook extracts and plots the **top 15 most-confused intent pairs** — this is much more actionable for understanding *where* the model still makes mistakes.

All three plots are saved as PNG files (`training_curves.png`, `confusion_pairs.png`) as well as shown inline.

### 5. Safer inference — confidence thresholding

The old single-example test always returned the top predicted label, however unconfident the model actually was. The updated `predict_intent()` function returns:
- the predicted intent and its confidence score,
- the top-3 candidate intents (useful for debugging or a "did you mean...?" chatbot flow),
- and if the top confidence is below a threshold (default 50%), it returns `"unclear / needs human review"` instead of a guess.

This directly addresses the earlier failure case: rather than confidently mislabeling an ambiguous message, the bot can flag it for review or ask a clarifying question.

### 6. Clearer explanations

Every step now has a markdown cell written in plain language explaining *what* the code does and *why* it's done that way — not just a step title — so the notebook can be followed by someone who isn't already familiar with Hugging Face `Trainer` internals.

## How to read the results

1. Run the notebook top to bottom.
2. After training, check `training_curves.png` (or the inline plot in Step 11):
   - Loss should decrease and flatten for both train and validation.
   - Accuracy/F1 should increase and plateau — that plateau is where early stopping engages.
3. Step 12 confirms which epoch's checkpoint was actually kept.
4. Step 13 gives the final, unbiased test set numbers plus a classification report and the confusion-pair plot — this is the honest measure of chatbot-readiness.
5. Step 14 lets you try your own sentences and see both the predicted intent and how confident the model actually is.

## Files

- `Banking_Chatbot_with_DistilBERT_Improved.ipynb` — the updated notebook.
- `README.md` — this file.

## Possible next steps

- Try a larger backbone (e.g. `bert-base-uncased` or `roberta-base`) if a small accuracy gain is worth the extra compute cost.
- Add a small "out-of-scope" class using non-banking text so the model learns to reject irrelevant messages directly, rather than only via the confidence threshold.
- Track experiments with a tool like Weights & Biases (`report_to="wandb"`) if you want to compare multiple hyperparameter settings side by side.
