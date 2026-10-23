import csv
import random
from pathlib import Path


def main():
	directory = Path(__file__).resolve().parent
	candidates = [p for p in directory.glob("*.csv") if "cleaned" not in p.stem.lower()
				  and "random" not in p.stem.lower()]
	source = next((p for p in candidates if "gutenberg" in p.name.lower()), None)
	if source is None:
		source = next(iter(candidates), None)
	if source is None:
		raise FileNotFoundError("Place the Gutenberg CSV next to sort.py and run again.")

	with source.open("r", encoding="utf-8-sig", newline="") as file:
		reader = csv.DictReader(file)
		if not reader.fieldnames:
			raise ValueError("The input CSV needs a header row.")
		headers = reader.fieldnames
		title_col = next((h for h in headers if h.strip().casefold() in
						  {"title", "name", "book title"}), None)
		count_col = next((h for h in headers if any(term in h.casefold() for term in
						  ("download", "count", "popularity", "view"))), None)
		if title_col is None:
			raise ValueError("Could not identify the book-title column.")

		books, seen = [], set()
		for row in reader:
			if not row or not any((value or "").strip() for value in row.values()):
				continue
			title = (row.get(title_col) or "").strip()
			key = title.casefold()
			if not title or key in seen:
				continue
			seen.add(key)
			if count_col:
				raw_count = (row.get(count_col) or "").replace(",", "").strip()
				try:
					count = int(float(raw_count)) if raw_count else 0
				except ValueError:
					count = 0
				row[count_col] = str(count)
			row[title_col] = title
			books.append(row)

	cleaned_path = directory / "gutenberg_cleaned.csv"
	with cleaned_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=headers)
		writer.writeheader()
		writer.writerows(books)

	if count_col:
		books.sort(key=lambda row: int(row[count_col]))
		least_popular = books[:max(1, (len(books) + 3) // 4)]
		selection = random.sample(least_popular, min(5, len(least_popular)))
	else:
		selection = random.sample(books, min(5, len(books)))

	result_path = directory / "random_unpopular_books.csv"
	result_headers = [title_col] + ([count_col] if count_col else ["count_unavailable"])
	with result_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=result_headers)
		writer.writeheader()
		for row in selection:
			writer.writerow({title_col: row[title_col],
							 count_col or "count_unavailable": row.get(count_col, "N/A")})


if __name__ == "__main__":
	main()

import pandas as pd

df = pd.read_csv("books_gutenberg_4000.csv")

pradinis_kiekis = len(df)

df["title"] = (
    df["title"]
    .astype("string")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df.dropna(subset=["title"])
df = df[df["title"] != ""]

kiek_po_tusciu = len(df)

df = df.drop_duplicates(subset=["title"], keep="first")

galutinis_kiekis = len(df)

seed = 20261021
atrinktos_knygos = df.sample(
    n=min(5, len(df)),
    replace=False,
    random_state=seed
)

print("Pradinis eilučių skaičius:", pradinis_kiekis)
print("Po tuščių pavadinimų pašalinimo:", kiek_po_tusciu)
print("Galutinis eilučių skaičius:", galutinis_kiekis)
print("Pašalinta tuščių pavadinimų:", pradinis_kiekis - kiek_po_tusciu)
print("Pašalinta dublikatų:", kiek_po_tusciu - galutinis_kiekis)
print("Atrankos metodas: atsitiktinė atranka be grąžinimo")
print("Sėkla:", seed)
print(atrinktos_knygos[["title", "author"]].to_string(index=False))

df.to_csv("books_clean.csv", index=False)
atrinktos_knygos.to_csv("books_sample_5.csv", index=False)
