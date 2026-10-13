# How Much of WELFake Accuracy Is Source Leakage?

*Working title for the paper version of this project. The original model comparison lives in
[`fake_news_detection.ipynb`](fake_news_detection.ipynb); everything here is built by `src/` and `scripts/`.*

## Motivation

Fake news detectors routinely report 96–99% accuracy on WELFake and similar corpora. While
building the baseline comparison we found two problems that call those numbers into question:

1. **The labels are documented backwards.** The Kaggle page says `0 = fake`, but every Reuters
   article (ISOT `True.csv`) has label 0 and every ISOT `Fake.csv` article has label 1.
2. **The classes are separable by provenance.** 60% of real articles open with a
   `CITY (Reuters) -` dateline that no fake article has; fake articles carry `Featured image via`,
   `Getty Images`, `pic.twitter.com` and outlet suffixes. Logistic Regression's strongest
   features are `reuters`, `breitbart` and `image`, which say where an article was scraped,
   not whether it is true.

So the question is no longer "which model is best on WELFake?" but **how much of each model's
accuracy comes from learning what a source looks like rather than what is fake?**

## Research questions

- **RQ1 – Data.** How much of WELFake is duplicated, and how is it composed? We recover each
  article's original dataset (ISOT, McIntire, Kaggle-UTK), remove exact and near-duplicates,
  and document the source artifacts per class.
- **RQ2 – Reliance.** How much does accuracy drop when source artifacts are removed at test time
  (models trained on original text)?
- **RQ3 – Content-only ceiling.** When models are trained *and* tested without artifacts, how
  much accuracy is left, and does the transformer's advantage over linear models grow or shrink?
- **RQ4 – Generalisation.** How well does each model transfer to a source it never saw during
  training (train on two sources, test on the third)?
- **RQ5 – Attribution.** Does DistilBERT attend to artifact tokens (Integrated Gradients)?

## Method

| Aspect | Original notebook | Paper pipeline |
|---|---|---|
| Duplicates | Exact `text` duplicates | Exact (normalised) + near-duplicates (MinHash-LSH, Jaccard ≈ 0.8); conflicting-label groups dropped |
| Split | One 80/20 split | 70/10/20 train/val/test, stratified by label × source, 5 seeds |
| Model selection | LR/SVM by CV; others untuned; DistilBERT evaluated on test during training | Every model tuned or early-stopped on the validation split only; test used once |
| DistilBERT input | Lemmatised, stopword-free text | Raw title + text |
| BiLSTM input | Lemmatised, stopword-free text | Lowercase tokens with stopwords |
| Uncertainty | None | Mean ± std over seeds, bootstrap 95% CIs, McNemar with Holm correction |
| Artifacts | Noted | Ablation: strip at test time, and at train + test time |
| Generalisation | — | Leave-one-source-out |
| Explainability | LR coefficients | + Integrated Gradients for DistilBERT |

Source artifacts removed by [`src/fnd/artifacts.py`](src/fnd/artifacts.py): Reuters and other
datelines, the word "Reuters", embedded tweet signatures, `pic.twitter.com` links, URLs,
@handles, "Featured image" / photo-credit lines, Getty, 21st Century Wire, NYT/Breitbart
mentions, share counters, "Read more:" lines and outlet suffixes on titles. The list is
deliberately conservative, so stripped-text accuracy is an *upper bound* on what content alone supports.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/tag_sources.py                 # optional: rebuilds data/welfake_sources.csv
python scripts/run_classical.py --setting in-domain --seeds 0 1 2 3 4   # or notebooks/run_classical_on_kaggle.ipynb
python scripts/run_classical.py --setting cross-source --seeds 0
# GPU: run notebooks/run_on_colab.ipynb, then copy its results/ folder here
python scripts/analyze.py                     # writes results/RESULTS.md, tables and figures
```

The dataset is downloaded with `kagglehub` if it is not found locally. Raw predictions
(`results/preds/`) and the text cache (`cache/`) are not committed; run logs, tables and
figures are.

## Results

See [`results/RESULTS.md`](results/RESULTS.md).
