from pathlib import Path
import sys
import joblib

BASE = Path(__file__).parent
MODEL = BASE / 'models' / 'run_B.joblib'


def predict(headlines):
    bundle = joblib.load(MODEL)
    vectorizer = bundle['vectorizer']
    lda = bundle['lda']
    X = vectorizer.transform(headlines)
    n_terms = X.sum(axis=1).A1
    valid = n_terms > 0
    result = []
    for text, terms, row in zip(headlines, n_terms, X):
        if terms == 0:
            result.append((text, 0, None, []))
            continue
        weights = lda.transform(row)[0]
        order = weights.argsort()[::-1]
        result.append((text, int(terms), int(order[0]), [(int(i), round(float(weights[i]), 3)) for i in order]))
    return result

if __name__ == '__main__':
    headlines = sys.argv[1:]
    if not headlines:
        print('Usage: python inference.py "headline one" "headline two"')
        raise SystemExit(1)
    for text, terms, dominant, weights in predict(headlines):
        print('\nHeadline:', text)
        if dominant is None:
            print('No retained vocabulary terms: insufficient evidence for a meaningful assignment.')
        else:
            print('Retained terms:', terms)
            print('Dominant topic:', dominant)
            print('Topic weights:', weights)
