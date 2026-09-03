# Pick-me-up — for the Monday group

For the few of you who ran Part 2 with me on **Monday, 31 August 2026**.

On Monday we tripped on a few concepts, let's clear this up.

Three pictures go with this note, and all three are new since Monday:

- `part2/notebooks/dataflow.svg` — **which rows go where**, with the real counts from my machine.
  Open it in a browser or look at it in the notebook.
- `part2/notebooks/archify/archify.html` — **which service hands what to which**, the same story
  told service by service. Open it in a browser. Click a box to trace what flows in and out of it,
  and use the three guided views (① decision, ② label, ③ improvement) to follow one loop at a time.
- `part2/notebooks/precision_and_pr-auc.svg` — **what the target precision buys and what it
  costs**, on the model we trained. It belongs to section 5b below.

---

## 1 · The data was split before anything else happened

The Sparkov dataset arrives already cut in two: `fraudTrain.csv` and `fraudTest.csv`. That split
happens before the trainer, before the server, before everything. Every later split (training rows
against validation rows) happens *inside* the first file only.

The numbers are on `dataflow.svg`. The whole dataset is 1,852,394 transactions. The training file
holds 1,296,675 of them, which is 70 per cent. The test file holds the other 555,719.

The cut is by time, not at random. The test file starts 48 seconds after the training file ends.
Mixing rows across that line would be **leakage**: the model would be tested on a period it had
already learned.

## 2 · `txn-app` replays fresh rows, so the model is really being challenged

This was the worry on Monday. Is the endpoint only asked about rows it was trained on?

No. `txn-app` reads from `fraudTest.csv`. The model in `fraud-serve` never saw those rows in
training. Every request on the dashboard is a credit-card transaction that is new to the model.

The application replays all 555,719 rows in file order, five per second by default. One full pass
takes about 31 hours, and then it starts again. It sends 12 raw columns per call. The true label
stays on the client side, where the dashboard uses it to colour the result. The model never
receives it.

View ① in `archify.html` is exactly this path: the raw file goes to `txn-app`, `txn-app` asks
`fraud-serve`, and both write what happened to Postgres.

## 3 · What to do with the human labels is a real debate

There is no single right answer. It depends on the use case. Some teams retrain on the verdicts,
some use them only to check the model, some use them to move the threshold.

The point of Part 2 is not the answer. The point is that you now have **a tool that collects the
labels**, and a loop that carries them back into the project. What you then do with them is your
decision, and it is worth having the argument.

Two rules in the setup are worth defending in that argument:

- The queue shows the analyst the transaction, never the model's score. A queue that shows the
  score turns into a confirmation exercise, and the labels it produces carry no new information.
- Verdicts arrive twice on purpose. The webhook is the fast path, and `POST /pull` re-reads the
  annotations later. Both write with an **upsert**, so a row is updated and never counted twice.

View ② in `archify.html` is that loop.

## 4 · How the human-labelled data gets built into the next dataset

First, a word about the label, because this is where Monday got tangled. Every row in Sparkov
already carries a label. It was written when the synthetic data was generated, and it is the
`is_fraud` column in both CSV files. Nothing here is unlabelled. What the review queue adds is a
**human-labelled** row: the same transaction, judged a second time by an analyst. So in this
section, "labelled data" always means human-labelled, and the plain Sparkov labels are called the
dataset label.

This was not clear on Monday. It is now on both pictures. The human verdicts land in Postgres,
`build_dataset.py` folds them into the parquet snapshot, and the snapshot gets an md5 hash. The
training run records that hash, so it can say which data it learned from. In `archify.html` that is
view ③, the dashed line running back from Postgres to the snapshot.

The rows are always written in the same order, so unchanged data produces the same hash. The sort
is by transaction time, and it is a stable sort, which means rows sharing the same second keep the
order they arrived in. Without that, the same data could produce a different file, and a different
file means a different hash. The hash moves only when a human has judged something new. Each
verdict does three things at once: it adds one human-labelled row to the snapshot, it removes that
row from the test set, and it gives the snapshot a new identity.

**Where those human-labelled rows actually land is the part that surprises people.** They go into
the validation slice, not into the weights. They are the newest rows in the snapshot, and the
trainer splits by time, so the newest rows are the validation tail. Keras never fits weights on
them.

That does not make them useless. The validation tail is where the operating threshold is chosen,
and the threshold decides what the server does with a score. So the verdicts change the decisions.
They are simply not a gradient.

There are four honest ways forward, and none of them is free:

1. **Say it, change nothing.** The most honest first step, and the one taken here.
2. **Append the human-labelled rows to the training side.** They reach the weights, but nothing is
   held out any more, so nothing measures whether they helped.
3. **Split the human-labelled rows themselves**, half to training and half to validation. This
   keeps a held-out sample, and it costs one exception to the "never mix across the time boundary"
   rule.
4. **Weight or repeat the human-labelled rows** so a few expensive labels can influence training at
   all. This is what a real feedback loop does, and it is also the easiest way to fool yourself.

Today there are 26 human-labelled rows, 10 of them fraud, against 1.3 million. At that size the
mechanism is the lesson, not the gain.

How Keras uses those rows is a configuration choice, not a fact of nature. This is exactly the
conversation the MLOps Engineer has with the ML Engineer.

## 5 · Two knobs that are ML-Engineer territory

The trainer has two settings that decide what the model does with a rare class. Neither is an
MLOps setting. Both are read from environment variables in `compose.part2.yml`, so an MLOps
Engineer can change them without touching the code, but the value is an ML decision. The first one
acts during training. The second one acts after training. They are often confused, so they are
here side by side.

### 5a · Class weights — the thing doing most of the work

Fraud is rare. In the training slice, 6,675 rows out of 1,167,030 are fraud. That is 0.57 per
cent, or about 174 honest transactions for every fraudulent one. That is life, and no MLOps tool
fixes it.

A network minimises a loss summed over rows. If every row counts the same, the cheapest answer is
"never fraud". Such a model is right 99.43 per cent of the time and has learned nothing. This is
why the project reports PR-AUC and treats accuracy as a lie.

A class weight multiplies the loss of each row, depending on the class of that row. It does not
copy rows and it does not change the data. It changes only how loudly a mistake is counted.
`class_weight="balanced"` gives a legitimate row the weight 0.50 and a fraudulent row the weight
87.4, so the two classes contribute the same total. One missed fraud then costs about 174 times
one false alarm, and "never fraud" stops being cheap.

Three things it does **not** do:

- It is applied to training only. Keras leaves the validation loss unweighted, so `loss` and
  `val_loss` are on different scales. A gap between them is not evidence of overfitting here.
  Compare each of them against itself across epochs.
- It does not choose the threshold. Weighting shifts the probabilities. The cut is decided
  afterwards, on the validation slice. The two mechanisms are independent, and they are often
  confused.
- It does not remove the imbalance. Only the loss is rebalanced, so every metric still has to be
  read with the base rate in mind.

Two other approaches exist, and `class_weight` is a choice, not the only option. You can resample
the data, or you can use a focal loss.

### 5b · Target precision — where you cut the score

The model does not output a decision. It outputs a probability between 0 and 1. Something has to
say where the cut is, and that cut is the **operating threshold**. `TARGET_PRECISION` is how you
ask for one. It does not change the network at all. It changes only which number the server
decides with.

The value says what you promise about your alerts. `TARGET_PRECISION=0.90` means: of the
transactions I flag, 9 out of 10 must really be fraud. The trainer then looks for a threshold that
keeps that promise.

The chain has four steps.

1. **You set it as an environment variable.** `compose.part2.yml` declares
   `TARGET_PRECISION: "0.90"` for `fraud-trainer`. Overriding it is `-e TARGET_PRECISION=0.95` on
   the `docker compose run` line, exactly like every other hyperparameter. The image is never
   rebuilt. Through Airflow the same value arrives as `--conf '{"target_precision": "0.95"}'`.
2. **The trainer picks a point on the validation curve.** `pick_threshold()` in `train.py` walks
   every possible threshold and keeps those where precision is at least the target. Among those it
   takes the one with the **highest recall**. That is the most generous cut that still keeps your
   promise. If no threshold reaches the target, the run does not fail. It falls back to the
   threshold with the best F1, and the log line shows both.
3. **The threshold travels with the model.** The trainer writes it as the `operating_threshold`
   run tag. `fraud-serve` reads that tag at start-up and refuses to start without it, unless you
   set `DECLINE_THRESHOLD` by hand. So each model version carries its own cut point, and promoting
   a version changes the weights and the decision boundary together.
4. **The test set sends the bill.** `test_precision_at_threshold`, `test_recall_at_threshold` and
   `test_alerts` are logged at that threshold, and `target_precision` is logged as a parameter. A
   run therefore records both what you asked for and what you actually got on rows the model never
   saw.

In daily use this is the analyst-cost dial. A high target gives few alerts, and most of them are
real, and more fraud passes through. Lower it to 0.70 and the threshold drops, recall rises, and
the review queue in E1 becomes much longer. Every analyst hour you add or remove is this one
number. PR-AUC ranks the models. `TARGET_PRECISION` decides how you use the winner.

One warning. The precision is measured on the validation tail, which holds only a few hundred
fraud rows. The precision you then get on the test set is often a little below the target. Read
`test_precision_at_threshold` before you trust the promise.

**`precision_and_pr-auc.svg` shows all of this on our own model**, the champion `fraud-keras` v29.
The left chart is the curve. The model gives you one curve, and every white dot on it is one value
of `TARGET_PRECISION`. The two charts on the right share one axis, which is the value you set, and
they say what each choice produces on the 555,693 test transactions the model never saw. Three
numbers from that picture are worth remembering:

- At `0.90`, the shipped setting, the cut sits at a score of 0.9968. The queue receives 1,748
  alerts and 680 frauds are never seen.
- At `0.50` the queue receives 4,798 alerts, which is 2.7 times the work, and only 325 frauds are
  missed.
- At `0.99` the queue receives 193 alerts and 1,953 of the 2,143 frauds pass. Almost every alert is
  real, and almost all the fraud is gone.

The picture also shows the promise slipping. At a target of 0.90 the model reaches 0.84 on the test
file, not 0.90, for the reason in the warning above.

## 6 · One epoch, in numbers

**An epoch is one full pass over the training rows.**

This came up on Monday as a side question. Take a run with 8 epochs, a batch size of 128, and the
full snapshot. The trainer holds 1,167,030 training rows and 129,671 validation rows.

| slice | rows | batches | what happens per batch | how often |
|---|---|---|---|---|
| train | 1,167,030 | 9,118 | forward, backward, weight update | every epoch |
| val | 129,671 | 1,014 | forward only, no update | once per epoch, at the end |

So one epoch is 9,118 weight updates, then a pause in which the model is frozen and the 1,014
validation batches are pushed through only to produce `val_loss` and the metrics. The validation
slice is cut into batches for memory reasons only. The metrics are averaged over the whole slice,
so the batch size does not change `val_loss`. On the training side the batch boundaries matter,
because every batch is one gradient step. Keras has a separate `validation_batch_size` argument,
and `train.py` does not set it, so both sides use 128.

The validation slice is then used twice more, after training has finished, and those two passes
decide what the server does. The first one picks the operating threshold. The second one scores
the slice again at that threshold, to log the metrics you actually get. The threshold travels with
the model as a run tag. The per-epoch `val_loss` only lets you watch the training progress.

## 7 · What changed in the repository since Monday

- **Docker only.** The Podman path was tried and removed. The notebooks now say a literal
  `docker compose` in every cell, so nothing depends on a choice made in an earlier kernel.
- `dataflow.svg` and `archify/archify.html` exist, as described at the top of this note.
- `known_issues.md` is now a short list of what is still missing from the notebook and the
  slides. Everything on it is work for the next days, not a problem you have to work around.

---

One thing is worth taking home from all five topics. **Machine-learning questions and MLOps
questions sit very close together in this project, and it helps to keep them apart while you
read.** Class weights, the threshold and the validation split are ML questions. Up-time,
reliability, provenance and security are MLOps questions. One person rarely masters both to the
same depth, and that is why companies build teams around this work.

Thanks for Monday. Bring the questions back.
