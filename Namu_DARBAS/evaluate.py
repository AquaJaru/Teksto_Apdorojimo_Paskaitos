
from collections import Counter

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

import app

OUT_OF_SCOPE = [
    "What is the weather like today?",
    "Who won the football match?",
    "Tell me a joke",
    "How do I cook pasta?",
    "What is the capital of France?",
    "Can you help me with my math homework?",
    "What is the price of bitcoin?",
    "Book me a flight to London",
    "asdf qwerty",
    "How tall is Mount Everest?",
]


def main():
    training, _ = app.load_data()
    texts = [app.normalize(item["text"]) for item in training]
    labels = [item["intent"] for item in training]

    print(f"Training examples: {len(texts)}")
    print("Examples per intent:", dict(sorted(Counter(labels).items())))

    min_class = min(Counter(labels).values())
    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
    predicted = cross_val_predict(app.build_model(), texts, labels, cv=cv)
    print("\nCross-validated classification report:\n")
    print(classification_report(labels, predicted, zero_division=0))

    held_out = cross_val_predict(
        app.build_model(), texts, labels, cv=cv, method="decision_function"
    ).max(axis=1)
    model = app.train_model(training)
    out_scores = [
        model.decision_function([app.normalize(q)])[0].max() for q in OUT_OF_SCOPE
    ]

    print(f"Current threshold: {app.DEFAULT_THRESHOLD:.2f}")
    kept = sum(s >= app.DEFAULT_THRESHOLD for s in held_out)
    rejected = sum(s < app.DEFAULT_THRESHOLD for s in out_scores)
    print(f"In-scope questions answered:  {kept}/{len(held_out)}")
    print(f"Out-of-scope rejected:        {rejected}/{len(OUT_OF_SCOPE)}")
    print("\nThreshold trade-off (in-scope answered / out-of-scope rejected):")
    for th in (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
        a = sum(s >= th for s in held_out) / len(held_out)
        r = sum(s < th for s in out_scores) / len(out_scores)
        print(f"  {th:+.1f}: {a:5.0%} / {r:5.0%}")

if __name__ == "__main__":
    main()
