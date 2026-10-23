
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("books_clean.csv")

text_columns = ["subjects", "bookshelves", "title"]

for col in text_columns:
    if col not in df.columns:
        df[col] = ""

df["text"] = (
    df[text_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

empty_count = df["text"].eq("").sum()
print("Tuščią tekstinį lauką turinčių kūrinių:", empty_count)

df = df[df["text"] != ""].copy()
df = df.reset_index(drop=True)

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b",
    stop_words="english",
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(df["text"])

print("Kūrinių vektorizuota:", tfidf_matrix.shape[0])
print("TF-IDF požymių skaičius:", tfidf_matrix.shape[1])


def recommend(title, k=5):
    matches = df.index[
        df["title"].str.casefold() == title.strip().casefold()
    ].tolist()

    if not matches:
        print("Kūrinys nerastas:", title)
        return pd.DataFrame(columns=["title", "author", "similarity"])

    query_idx = matches[0]

    query_vector = tfidf_matrix[query_idx]
    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    similarities[query_idx] = -1

    top_indices = similarities.argsort()[::-1]
    top_indices = [
        i for i in top_indices
        if similarities[i] >= 0
    ][:k]

    result = df.iloc[top_indices][["title", "author"]].copy()
    result["similarity"] = similarities[top_indices]

    return result.reset_index(drop=True)


def popular_recommendations(k=5):
    if "downloads" not in df.columns:
        raise ValueError("CSV faile nėra 'downloads' stulpelio.")

    popular = df.copy()
    popular["downloads"] = pd.to_numeric(
        popular["downloads"], errors="coerce"
    ).fillna(0)

    return (
        popular.sort_values("downloads", ascending=False)
        [["title", "author", "downloads"]]
        .head(k)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print("\nĮvesk knygos pavadinimą:")
    query_title = input("> ")

    print("\nPanašiausios knygos:")
    print(recommend(query_title, k=5).to_string(index=False))

    print("\n5 populiariausios knygos:")
    print(popular_recommendations(k=5).to_string(index=False))
popular_recommendations(k=5).to_csv(
    "run_a_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Rezultatai išsaugoti faile run_a_results.csv")