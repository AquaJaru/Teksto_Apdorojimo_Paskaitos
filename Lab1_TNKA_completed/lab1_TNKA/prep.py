import re

FILLER_PHRASES = [
    "you won't believe",
    "what happens next",
    "will shock you",
    "this trick will change your life",
    "number 5 is shocking"
]

def remove_fillers(text):
    text = str(text).lower()

    for phrase in FILLER_PHRASES:
        text = text.replace(phrase, " ")

    text = re.sub(r"\s+", " ", text).strip()

    return text