# Laboratory 1 — Training a Topic Model

## Contents
- `s1.py` — cleaning, reproducible 1000-document sample and selection of 20 headlines.
- `s2.py` — runs A (10 topics) and B (4 topics) with the original preprocessing.
- `s3.py` — run C (4 topics) with filler-phrase removal.
- `prep.py` — the single preprocessing change used in run C.
- `inference.py` — applies the already fitted selected model (run B) to new headlines without retraining.
- `outputs/` — topic words and document-topic weights for runs A, B and C.
- `models/` — fitted LDA/vectorizer bundles.
- `report.md` — report using the supplied template.
- `initial-analysis.md` — checkpoint material based on the student's recorded 20-headline selections; the student must keep the original checkpoint unchanged and complete any required pre-model interpretation themselves.

## Run
```bash
python s1.py
python s2.py
python s3.py
```

For defence-style inference after the model has already been fitted:
```bash
python inference.py "A new headline to classify" "Another headline"
```
No retraining is performed by `inference.py`.

## Parameters
- Original dataset: 50,000 rows.
- After empty-text cleaning: 50,000 rows.
- After exact duplicate removal: 4,845 unique documents.
- Training sample: 1,000 documents, `random_state=42`.
- Initial 20 selection: `random_state=7`.
- Topic counts: A=10, B=4, C=4.
- LDA seed: 42.
- `max_iter=30`, batch learning.
- `CountVectorizer`: `min_df=2`, `max_features=3000`.
- Token pattern: `(?u)\b[^\W\d_]{2,}\b`.
- English stop words plus the Lithuanian stop-word list in `s2.py`.
- Run C removes five recurring English filler phrases and changes no other model setting.

## Package versions used for the checked run
- pandas 2.2.3
- scikit-learn 1.8.0
- joblib 1.5.3
- numpy 2.3.5

The collection is a teaching dataset containing repeated/template-like headlines and mixed English/Lithuanian fragments, so the results are exploratory and should not be interpreted as evidence of generalisation to real news.
