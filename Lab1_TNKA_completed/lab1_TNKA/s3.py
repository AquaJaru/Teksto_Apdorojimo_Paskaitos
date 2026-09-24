from s2 import load_collection, run_lda
from prep import remove_fillers

if __name__ == "__main__":
    docs, first20 = load_collection()

    run_lda(
        "C",
        4,
        docs,
        first20,
        preprocessor=remove_fillers
    )