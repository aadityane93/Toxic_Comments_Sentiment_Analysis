# Toxic Comment Classification Project Plan

## 1. Assignment Requirements From `Project_Work_AIN.pdf`

This project is for the Deep Learning and Big Data course. The goal is to understand, implement, and critically compare models on one dataset.

### Hard deadlines

- Submission deadline: June 23, 2026 at 23:59.
- Late submissions receive 0%.
- Presentation sessions: June 25, 2026, July 2, 2026, and July 9, 2026.
- Presentation format: 15 minutes presentation plus 10 minutes Q&A.
- Attendance at all presentation sessions is mandatory.

### Required deliverables

- Code: Python, preferably PyTorch, in a Jupyter Notebook, GitHub repository, or Colab.
- Report: 6 to 10 pages.
- Presentation: slides exported as PDF.

### Required project content

- Use one dataset.
- Define two tasks on the same dataset.
- Implement and compare two models.
- Model 1 must be implemented and trained from scratch.
- Model 2 must be an alternative approach, such as a pretrained NLP model or a strong baseline.
- The report must clearly compare the models.
- All design decisions must be justified: architecture, layers, activation functions, loss function, optimizer, and hyperparameters.
- Individual group contributions must be visible.
- Sources must be cited. External solutions or tutorials must not be copied.

### Grading weights

- Coding / Modeling: 40%.
- Report: 30%.
- Presentation: 30%.

## 2. Dataset

Dataset: Jigsaw Toxic Comment Classification Challenge  
Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

The dataset contains Wikipedia comments labeled for different types of toxicity.

### Local files

The project directory currently contains:

- `train.csv`
- `test.csv`
- `test_labels.csv`
- `sample_submission.csv`

### File roles

- `train.csv`: Training data with comment text and labels.
- `test.csv`: Test comments with `id` and `comment_text`, but no labels.
- `test_labels.csv`: Labels for the test set. Rows with `-1` were not used for scoring and should be filtered out during evaluation.
- `sample_submission.csv`: Example Kaggle submission format. Useful if generating Kaggle-style prediction files.

### Dataset profile

Training rows: 159,571  
Test rows: 153,164  
Valid labeled test rows after removing `-1` labels: 63,978

Training label counts:

| Label | Positive Count |
|---|---:|
| toxic | 15,294 |
| severe_toxic | 1,595 |
| obscene | 8,449 |
| threat | 478 |
| insult | 7,877 |
| identity_hate | 1,405 |

Binary summary:

- Any toxic label: 16,225 rows.
- Clean comments: 143,346 rows.

Important implication: the dataset is highly imbalanced, especially for `threat`, `identity_hate`, and `severe_toxic`. Accuracy alone is not enough.

## 3. Proposed Project Direction

The project should be framed as an NLP toxicity detection system.

### Research question

How well can a model detect toxic online comments, and how does a from-scratch neural model compare with an alternative pretrained or strong baseline approach?

### Task 1: Binary Toxicity Detection

Goal: classify each comment as toxic or non-toxic.

Create a new target:

```python
label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

train_df["any_toxic"] = (train_df[label_columns].sum(axis=1) > 0).astype(int)
```

This task is useful because real moderation systems often first need to answer: "Should this comment be flagged?"

### Task 2: Multi-label Toxicity Type Classification

Goal: predict all applicable toxicity categories for each comment.

Labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

This is a multi-label task because one comment can belong to multiple categories at the same time.

If the instructor requires a strict multi-class task, use this backup version:

- `clean`
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

In that backup setup, each comment receives one dominant class. However, the multi-label setup is more faithful to the actual dataset.

## 4. Models

Both models should be evaluated on both tasks.

### Model 1: From-scratch PyTorch text classifier

This is the required from-scratch model.

Recommended architecture:

- Text preprocessing:
  - Lowercase text.
  - Basic cleanup for URLs, repeated whitespace, and missing values.
  - Tokenize comments.
  - Build a vocabulary from the training split only.
  - Convert comments to padded token sequences.
- Embedding layer:
  - Randomly initialized.
  - Trained from scratch.
- Encoder:
  - Option A: 1D CNN over token embeddings.
  - Option B: BiLSTM or GRU over token embeddings.
- Regularization:
  - Dropout.
  - Optional weight decay.
- Output heads:
  - Binary task: one output unit.
  - Multi-label task: six output units.
- Activation:
  - Use sigmoid for final probabilities.
- Loss:
  - `BCEWithLogitsLoss`.
  - Use class weights or `pos_weight` to handle imbalance.
- Optimizer:
  - Adam or AdamW.
- Training:
  - Train/validation split from `train.csv`.
  - Early stopping based on validation macro F1 or validation loss.

Design decisions to justify in the report:

- Why this architecture fits text classification.
- Why sigmoid is used instead of softmax for multi-label prediction.
- Why weighted binary cross-entropy is appropriate for imbalanced labels.
- Why dropout and early stopping are used.

### Model 2: Alternative approach

Preferred option: pretrained NLP model.

Recommended model:

- DistilBERT or BERT from Hugging Face.
- Fine-tune it for:
  - Binary toxicity detection.
  - Multi-label toxicity classification.

Why this is strong:

- It uses pretrained language representations.
- It can understand word context better than a simple from-scratch model.
- It directly satisfies the PDF's "pretrained models (e.g., NLP)" option.

Fallback option if compute is limited:

- TF-IDF vectorization plus Logistic Regression or Linear SVM.
- This counts as a strong baseline according to the PDF.
- Keep it as a sanity baseline even if DistilBERT is used.

Recommended final comparison:

| Model | Type | Strength | Weakness |
|---|---|---|---|
| CNN/BiLSTM from scratch | Deep learning from scratch | Fully controlled, satisfies assignment requirement | Needs more data and tuning |
| DistilBERT | Pretrained NLP | Strong contextual language understanding | More expensive to train |
| TF-IDF + Logistic Regression | Classical baseline | Fast, interpretable, reproducible | Less contextual understanding |

The final report only needs two main models, but including TF-IDF as a small baseline can strengthen the discussion.

## 5. Evaluation Plan

Because the data is imbalanced, prioritize metrics that handle rare positive labels.

### Binary task metrics

- Accuracy, for basic interpretability.
- Precision.
- Recall.
- F1 score.
- ROC-AUC.
- PR-AUC, especially useful for imbalanced data.
- Confusion matrix.

### Multi-label task metrics

- Micro F1.
- Macro F1.
- Per-label F1.
- Per-label ROC-AUC.
- Hamming loss.
- Label-wise confusion matrices or precision-recall summaries.

### Threshold tuning

Use validation data to choose thresholds.

- Default threshold: 0.5.
- Tune thresholds per label if it improves macro F1.
- Report whether the same threshold or per-label thresholds were used.

### Test set handling

For `test_labels.csv`, remove rows with `-1` before evaluating:

```python
valid_mask = (test_labels[label_columns] != -1).all(axis=1)
valid_test_labels = test_labels[valid_mask]
valid_test_comments = test_df[test_df["id"].isin(valid_test_labels["id"])]
```

## 6. Notebook Structure

The current notebook, `1_Analysis.ipynb`, already starts with imports and data loading. Expand it into this structure:

1. Title and project overview.
2. Dataset source and file explanation.
3. Imports and reproducibility settings.
4. Load data.
5. Exploratory data analysis:
   - Row counts.
   - Missing values.
   - Label distributions.
   - Comment length distributions.
   - Examples of clean and toxic comments.
6. Preprocessing:
   - Text cleaning.
   - Train/validation split.
   - Binary label creation.
   - Multi-label target preparation.
7. Baseline model:
   - TF-IDF + Logistic Regression.
   - Fast benchmark metrics.
8. Model 1 from scratch:
   - Vocabulary.
   - Dataset and DataLoader.
   - PyTorch architecture.
   - Training loop.
   - Validation metrics.
9. Model 2 alternative:
   - DistilBERT fine-tuning, or TF-IDF baseline if compute is limited.
10. Evaluation:
   - Binary task results.
   - Multi-label task results.
   - Tables and plots.
11. Model comparison:
   - Accuracy/performance.
   - Training time.
   - Complexity.
   - Interpretability.
   - Limitations.
12. Final conclusions.

## 7. Report Structure: 6 to 10 Pages

Suggested report outline:

1. Introduction and problem statement.
   - Why toxic comment detection matters.
   - What the project tries to solve.
2. Dataset description.
   - Jigsaw source.
   - File structure.
   - Labels.
   - Class imbalance.
   - Ethical note about toxic language.
3. Task definitions.
   - Binary toxicity detection.
   - Multi-label toxicity classification.
4. Model 1: from-scratch neural network.
   - Architecture.
   - Preprocessing.
   - Loss function.
   - Optimizer.
   - Hyperparameters.
5. Model 2: alternative model.
   - DistilBERT or strong baseline.
   - Why it was chosen.
   - Training setup.
6. Results.
   - Tables for binary metrics.
   - Tables for multi-label metrics.
   - Main plots.
7. Comparison and discussion.
   - Which model performed better and why.
   - Strengths and weaknesses.
   - Error analysis.
8. Limitations and future work.
   - Class imbalance.
   - Bias and fairness concerns.
   - Compute limitations.
   - Possible improvements.
9. Individual contributions.
10. References.

## 8. Presentation Structure: 15 Minutes

Suggested timing:

| Time | Section |
|---:|---|
| 1 min | Problem and motivation |
| 2 min | Dataset and labels |
| 2 min | Task definitions |
| 3 min | Model 1 from scratch |
| 3 min | Model 2 alternative |
| 2 min | Results and comparison |
| 1 min | Limitations |
| 1 min | Conclusion |

Prepare for Q&A on:

- Why the dataset is imbalanced.
- Why sigmoid is used for multi-label classification.
- Why accuracy is not enough.
- How `-1` labels in `test_labels.csv` were handled.
- Why the chosen models are a fair comparison.
- What each group member contributed.

## 9. Timeline From May 23, 2026 to Submission

### May 23 - May 26, 2026: Setup and EDA

- Confirm all data files load correctly.
- Clean notebook structure.
- Add dataset explanation.
- Create binary target.
- Plot label distribution and comment lengths.
- Decide final Model 2: DistilBERT or TF-IDF baseline.

### May 27 - June 2, 2026: Baseline and first working model

- Implement TF-IDF + Logistic Regression baseline.
- Implement train/validation split.
- Define metrics.
- Save first result tables.
- Start from-scratch PyTorch Dataset and DataLoader.

### June 3 - June 9, 2026: From-scratch model

- Implement CNN or BiLSTM model.
- Train on binary task.
- Train on multi-label task.
- Add class weighting.
- Tune basic hyperparameters.
- Record training curves and metrics.

### June 10 - June 15, 2026: Alternative model

- Fine-tune DistilBERT if compute is available.
- Otherwise finalize TF-IDF + Logistic Regression as the strong baseline.
- Evaluate on both tasks.
- Compare against the from-scratch model.

### June 16 - June 19, 2026: Analysis and report

- Create final result tables.
- Create plots.
- Write report draft.
- Add citations.
- Add individual contribution section.

### June 20 - June 22, 2026: Presentation and final cleanup

- Prepare slides.
- Export slides to PDF.
- Clean notebook outputs.
- Verify code runs top to bottom.
- Check all deliverables against the PDF requirements.

### June 23, 2026: Submission day

- Submit before 23:59.
- Do not leave upload or export work until the final hour.

## 10. Recommended Repository Improvements

Add these files before submission:

- `README.md`: project overview, setup, dataset instructions, how to run.
- `requirements.txt`: package list.
- `.gitignore`: exclude large local data files and notebook checkpoints.
- `PROJECT_PLAN.md`: this planning document.

Suggested `.gitignore` entries:

```text
*.csv
*.csv.zip
.ipynb_checkpoints/
__pycache__/
models/
outputs/
```

Do not commit the Kaggle dataset files unless the instructor explicitly requires it. They are large and come from Kaggle.

## 11. Success Checklist

- The notebook loads all data correctly.
- The notebook can run from top to bottom.
- The project defines two tasks on the same dataset.
- Model 1 is implemented from scratch.
- Model 2 is clearly an alternative approach.
- Both models are evaluated with appropriate metrics.
- The report includes a direct comparison.
- The report explains all design choices.
- The report includes limitations and citations.
- The presentation fits into 15 minutes.
- Individual group contributions are clearly documented.

