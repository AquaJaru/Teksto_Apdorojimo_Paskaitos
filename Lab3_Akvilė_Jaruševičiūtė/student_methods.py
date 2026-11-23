import numpy as np
from text_units import sentences as split_sentences

def _centroid_scores(parts, encode):
    vectors = np.asarray(encode(parts), dtype=float)
    centroid = vectors.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    scores = vectors @ centroid
    return (vectors, scores)

def score_table(text, encode):
    parts = split_sentences(text)
    if not parts:
        return []
    _, scores = _centroid_scores(parts, encode)
    return [(i, float(s), p) for i, (s, p) in enumerate(zip(scores, parts))]

def extractive(text, encode, sentences=2):
    parts = split_sentences(text)
    if len(parts) <= sentences:
        return ' '.join(parts)
    _, scores = _centroid_scores(parts, encode)
    ranked = sorted(range(len(parts)), key=lambda i: (-scores[i], i))
    chosen = sorted(ranked[:sentences])
    return ' '.join((parts[i] for i in chosen))

def extractive_mmr(text, encode, sentences=2, lam=0.5):
    parts = split_sentences(text)
    if len(parts) <= sentences:
        return ' '.join(parts)
    vectors, scores = _centroid_scores(parts, encode)
    chosen = [min(range(len(parts)), key=lambda i: (-scores[i], i))]
    while len(chosen) < sentences:
        rest = [i for i in range(len(parts)) if i not in chosen]

        def mmr(i):
            redundancy = max((float(vectors[i] @ vectors[j]) for j in chosen))
            return scores[i] - lam * redundancy
        chosen.append(min(rest, key=lambda i: (-mmr(i), i)))
    return ' '.join((parts[i] for i in sorted(chosen)))
