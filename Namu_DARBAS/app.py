
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = Path(__file__).resolve().parent


DEFAULT_THRESHOLD = -0.30

UNKNOWN_MESSAGE = (
    "Sorry, I did not understand the question. Please try rephrasing it or "
    "ask about opening hours, registration, borrowing books, reservations "
    "or other library services."
)
EMPTY_MESSAGE = "Please enter a question."
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}


def normalize(text: str) -> str:
   
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(data_dir: Path = DATA_DIR):
   
    with open(data_dir / "training_data.json", encoding="utf-8") as file:
        training = json.load(file)
    with open(data_dir / "faq.json", encoding="utf-8") as file:
        faq = json.load(file)

    faq_by_intent = {item["intent"]: item for item in faq}
    unknown = {item["intent"] for item in training} - set(faq_by_intent)
    if unknown:
        raise ValueError(
            f"training_data.json uses intents missing from faq.json: {sorted(unknown)}"
        )
    return training, faq_by_intent


def build_model():
    
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True)),
    ])
    return Pipeline([
        ("tfidf", features),
        ("svm", LinearSVC(C=1.0, class_weight="balanced")),
    ])


def train_model(training):
    texts = [normalize(item["text"]) for item in training]
    labels = [item["intent"] for item in training]
    model = build_model()
    model.fit(texts, labels)
    return model


def predict_intent(question, model, threshold=DEFAULT_THRESHOLD):

    cleaned = normalize(question)
    if not cleaned:
        return None, 0.0
    scores = model.decision_function([cleaned])[0]
    best = int(scores.argmax())
    best_score = float(scores[best])
    if best_score < threshold:
        return None, best_score
    return model.classes_[best], best_score


def answer_question(question, model, faq_by_intent, threshold=DEFAULT_THRESHOLD):
    if not normalize(question):
        return EMPTY_MESSAGE
    intent, _ = predict_intent(question, model, threshold)
    if intent is None:
        return UNKNOWN_MESSAGE
    return faq_by_intent[intent]["answer"]


def main():
    training, faq_by_intent = load_data()
    model = train_model(training)

    print("=" * 58)
    print("LIBRARY FAQ CHATBOT (TF-IDF + SVM)")
    print("Ask your question in English. Type 'exit' to quit.")
    print("=" * 58)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot is shutting down. Goodbye!")
            break

        if question.lower() in EXIT_COMMANDS:
            print("Chatbot: Goodbye!")
            break

        print("Chatbot:", answer_question(question, model, faq_by_intent))


if __name__ == "__main__":
    main()
