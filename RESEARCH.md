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

## Status (paused 2026-10-13)

The pipeline is complete and tested; the full experiments have **not** been run yet.

**Done**
- Source recovery: every WELFake row is tagged ISOT / McIntire / Kaggle-UTK
  ([`data/welfake_sources.csv`](data/welfake_sources.csv)). ISOT `True.csv` articles are all
  label 0 and `Fake.csv` articles all label 1, which independently confirms the label audit.
- Corpus cleaning: 72,134 → 70,793 (missing/empty) → 61,120 after removing 8,887 exact and
  770 near-duplicates, plus 16 near-duplicates whose copies had conflicting labels.
- Artifact prevalence (share of articles): Reuters dateline 60.3% of Real vs 0% of Fake;
  "Featured image" 0% vs 31.1%; photo-credit lines 0.1% vs 29.6%; Getty 0% vs 15.5%;
  `pic.twitter.com` 0% vs 11.4%; outlet suffix on titles 25.2% vs 3.0%.
- Smoke tests: classical, BiLSTM and DistilBERT (+ Integrated Gradients) pipelines all run end to end.

**Preliminary results (seed 0 only, in-domain, F1 on the Fake class; not for citation)**

These came from a local run that was stopped because the laptop ran out of memory. They
were deleted so they don't mix with the final runs, and LR has since been re-gridded, so
the final numbers will differ.

| Model | Original → original | Original → stripped | Drop |
|---|---:|---:|---:|
| Naive Bayes | 0.885 | 0.870 | −1.5 |
| Logistic Regression | 0.962 | 0.926 | −3.7 |
| Linear SVM | 0.963 | 0.923 | −4.0 |
| Decision Tree | 0.922 | 0.665 | −25.7 |

Early reading: every model loses accuracy when source artifacts are removed at test time,
and the Decision Tree largely collapses (it mostly learned "contains *Reuters* → Real").

**To resume**
1. *Classical (Kaggle, CPU, 30 GB RAM):* import
   [`notebooks/run_classical_on_kaggle.ipynb`](notebooks/run_classical_on_kaggle.ipynb) twice,
   with `SETTING = 'in-domain'` and `'cross-source'`, Internet on, then *Save & Run All (Commit)*.
   Estimated 3–5 h each. Download `results_<setting>.zip` from each version's Output tab.
2. *Neural (Colab, T4):* run [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb);
   results land in Google Drive under `fake-news-paper/results`. Estimated 3–4 h; re-run all
   cells after a disconnect and finished runs are skipped.
3. Unzip/copy everything into `results/` (file names don't clash), then run
   `python scripts/analyze.py` to produce `results/RESULTS.md`, tables and figures.
4. Write the paper from `RESULTS.md` (see *Publishing plan* below).

Local runs need more than the ~2 GB of free RAM this laptop had; the Decision Tree took
~40 min per fit there because of swapping (about 3 min on Colab).

## Publishing plan

**Paper outline (IEEE conference template, 4–6 pages):** introduction (the 96–99% claims and
why they are suspect) → related work (the references in the README, plus shortcut-learning
and dataset-artifact papers) → data: source recovery, label audit, de-duplication, artifact
table → method → results: RQ1–RQ5 → limitations (the artifact list is hand-written; WELFake
is 2016–2017 US politics) → conclusion. Release the code and `data/welfake_sources.csv` with it.

**Candidate venues**
- Bangladesh, IEEE-indexed: ICCIT, ICECE, ECCE, STI, WIECON-ECE, ICAEEE.
- Regional: IEEE TENCON and other IEEE Region 10 conferences.
- NLP workshops: ACL/EMNLP/NAACL Student Research Workshop; the FEVER fact-checking workshop.

Check that a venue is indexed (IEEE Xplore or ACL Anthology) and avoid fee-heavy venues with
no indexing. Asking the CSE427 instructor to co-author is worthwhile.

**Before submitting**
- [ ] All runs finished; `RESULTS.md` regenerated.
- [ ] Main claims hold across seeds (mean ± std, McNemar).
- [ ] Optional extension: test on LLM-generated fake news (README reference 9).
- [ ] Notebook re-run on Colab so its embedded figures use the corrected labels (`main` branch).

## Results

Once the runs above are done, see `results/RESULTS.md` (generated by `scripts/analyze.py`).
