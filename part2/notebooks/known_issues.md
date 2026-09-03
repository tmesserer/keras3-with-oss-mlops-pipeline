# Open work

This file is a list of what is still missing, not an explanation of the project. The explanations
stay in `hub.ipynb`, as they always have, and the material written for the students who ran Part 2
on 31 August 2026 is in `pick-me-up.md`. Everything below is work for the next days. When it is
done, this file goes away.

---

## 1 · The notebook must say where the analyst rows land

The rows a human has judged reach the validation slice only. They do not update any weights. They
land in the validation tail because they are the newest rows in the snapshot, and the trainer
splits by time.

A reader can easily assume that human labels retrain the model directly, and here they do not.
What they do change is the threshold selection and the hash of the dataset. The notebook has to
say this in one plain sentence, and `dataflow.svg` should say it too.

`pick-me-up.md`, section 4, is the text to work from. It states what is true today and lists the
four ways to change it, from "say it, change nothing" to weighting the analyst rows.

Two facts that must survive any rewrite:

- The analyst rows are withheld from the test set (`train.py`, the `test_rows_withheld` block).
  Their period comes from `fraudTest.csv`, so leaving them in both places would leak.
- `TRAIN_SUBSET` does not change the picture. It keeps the **tail** of the snapshot, so the
  analyst rows stay at the end, and the end is the validation slice.

The cause is structural, not a matter of there being only 26 rows today. Any number of analyst
rows sits in validation for as long as they come from a later period and the split stays ordered
by time.

## 2 · The notebook needs a refresher on how Keras sees a dataset

The audience has the ML terms already. What they do not have is the mapping from those terms to
the files on disk in this repository: what counts as one dataset, how it is split, and how each
slice is walked again across epochs and batches.

`pick-me-up.md`, section 6, has the text and the numbers for a run with 8 epochs at batch size
128. It needs to find its place in `hub.ipynb`.

## 3 · The data flow must be revealed in stages, not all at once

`txn-app` replays `fraudTest.csv`. A reader currently meets that fact late, and by then the split
of the original dataset is far behind them. The one diagram should be shown four times, each time
with one more stage visible.

- **Stage 1 — early in the notebook.** Only the split of the original data into `fraudTrain.csv`
  and `fraudTest.csv`, and what the first training round uses. This is the place to explain how
  Keras sees the dataset at first.
- **Stage 2 — just before the application starts.** Remind the reader about `fraudTest.csv`.
  Explain how `txn-app` uses those rows, and why and how Postgres receives a row from the
  application and from the endpoint. The reader should form an expectation before the dashboard
  opens.
- **Stage 3 — when the human review queue arrives.** Human-in-the-loop is a big step, so it gets a
  highlight. It does not need much space, because Label Studio provides the visual aid itself.
- **Stage 4 — before the next training round.** Show how the new dataset is built, saved and
  tracked, and how Keras uses the new rows in the validation slice only.

One gap runs across all four stages. `features.py` is explained well, but the two thresholds are
not on the picture at all: the target precision the trainer aims for, and the business decision
band the server applies. Both decide the outcome as much as the training corpus does, so both
belong on `dataflow.svg`.

## 4 · The slides must say why accuracy is the wrong measure, before the number appears

`train.py` passes `class_weight` to `model.fit`, and the notebook never says what that is. A
reader meeting the argument for the first time has no way to guess how much depends on it. The
one-line comment in the code calls the ratio "~172:1" and moves on.

`pick-me-up.md`, section 5, is the explanation to work from: what a class weight multiplies, the
two numbers `balanced` produces, and the three things weighting does not do. Resampling and a
focal loss are worth one sentence each, so that a reader knows `class_weight` is a choice and not
the only option.

The slides need the shorter version of this, placed before any accuracy number is shown on screen.
