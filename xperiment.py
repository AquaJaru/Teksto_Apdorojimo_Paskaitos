
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

FILE = "books_clean.csv"
SEED = 20261021
K = 5

df = pd.read_csv(FILE)

for col in ["title", "author", "subjects", "bookshelves", "downloads"]:
    if col not in df.columns:
        df[col] = ""

df["title"] = df["title"].fillna("").astype(str).str.strip()
df = df[df["title"] != ""].drop_duplicates(
    subset=["title"]
).reset_index(drop=True)

for col in ["subjects", "bookshelves"]:
    df[col] = df[col].fillna("").astype(str)

df["text"] = (
    df["subjects"] + " " +
    df["bookshelves"] + " " +
    df["title"]
).str.replace(r"\s+", " ", regex=True).str.strip()

empty_count = int(df["text"].eq("").sum())
print("Tušti tekstiniai laukai:", empty_count)

df = df[df["text"] != ""].reset_index(drop=True)

queries = [
    "The Balkans: A History of Bulgaria",
    "As I Remember Recollections of American Society",
    "Alice in Wonderland",
    "The Battle of the Books",
    "The Suffrage Cook Book"
]

expected = {
    "The Balkans": ["Essays on the Latin Orient"],
    "As I Remember": ["American Notes", "The Four Million"],
    "Alice in Wonderland": ["Dramatic Reader for Lower Grades"],
    "The Battle of the Books": [
        "A Modest Proposal",
        "Bacon's Essays"
    ],
    "The Suffrage Cook Book": [
        "Eighty Years and More",
        "History of Woman Suffrage"
    ]
}

base_vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b",
    stop_words="english",
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)

common_labels = {
    "fiction", "drama", "poetry", "novel", "novels",
    "short", "stories", "drama film", "fiction film"
}

modified_stop_words = list(
    set(ENGLISH_STOP_WORDS) | common_labels
)

modified_vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b",
    stop_words=modified_stop_words,
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)

base_matrix = base_vectorizer.fit_transform(df["text"])
modified_matrix = modified_vectorizer.fit_transform(df["text"])

df["downloads"] = pd.to_numeric(
    df["downloads"], errors="coerce"
).fillna(0)

popular = df.sort_values(
    "downloads", ascending=False
).head(K)

def find_query(query):
    exact = df.index[
        df["title"].str.casefold() == query.casefold()
    ].tolist()

    if exact:
        return exact[0]

    matches = df.index[
        df["title"].str.contains(
            re.escape(query), case=False, regex=True
        )
    ].tolist()

    return matches[0] if matches else None

def recommend(index, matrix, k=K):
    scores = cosine_similarity(
        matrix[index], matrix
    ).ravel()

    scores[index] = -1

    indices = scores.argsort()[::-1]
    indices = [i for i in indices if scores[i] >= 0][:k]

    result = df.iloc[indices][["title", "author"]].copy()
    result["similarity"] = scores[indices].round(4)

    return result.reset_index(drop=True)

results = []

for query in queries:
    index = find_query(query)

    if index is None:
        print(f"\nUžklausa nerasta: {query}")
        continue

    title = df.loc[index, "title"]

    key = next(
        (k for k in expected if k.casefold() in query.casefold()),
        None
    )

    expected_titles = expected.get(key, [])

    print(f"\n{'=' * 70}")
    print("UŽKLAUSA:", title)

    print("\n1. POPULIARUMO MODELIS")
    print(popular[["title", "author", "downloads"]].to_string(index=False))

    for model_name, matrix in [
        ("Pradinis TF-IDF", base_matrix),
        ("Pakeistas TF-IDF", modified_matrix)
    ]:
        recs = recommend(index, matrix)

        print(f"\n{model_name}")
        print(recs.to_string(index=False))

        for _, row in recs.iterrows():
            matched = any(
                expected_title.casefold() in row["title"].casefold()
                for expected_title in expected_titles
            )

            results.append({
                "query": title,
                "model": model_name,
                "recommended_title": row["title"],
                "author": row["author"],
                "similarity": row["similarity"],
                "matches_initial_analysis": matched
            })

    for _, row in popular.iterrows():
        results.append({
            "query": title,
            "model": "Populiarumo modelis",
            "recommended_title": row["title"],
            "author": row["author"],
            "similarity": None,
            "matches_initial_analysis": any(
                expected_title.casefold() in row["title"].casefold()
                for expected_title in expected_titles
            )
        })

results_df = pd.DataFrame(results)

# Run B – pradinis TF-IDF
results_df[
    results_df["model"] == "Pradinis TF-IDF"
].to_csv(
    "run_b_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# Run C – pakeistas TF-IDF
results_df[
    results_df["model"] == "Pakeistas TF-IDF"
].to_csv(
    "run_c_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# Išsaugomi ir visi rezultatai viename faile,
# kaip buvo pradiniame kode.
results_df.to_csv(
    "experiment_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRezultatai išsaugoti:")
print("run_b_results.csv")
print("run_c_results.csv")
print("experiment_results.csv")

print("\nSutapimai su pradine analize:")
print(
    results_df.groupby(
        ["query", "model"]
    )["matches_initial_analysis"].sum().to_string()
)

