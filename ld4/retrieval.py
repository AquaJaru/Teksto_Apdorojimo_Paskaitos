from pathlib import Path
import numpy as np,json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from student_search import cosine_rank

class Retriever:
    def __init__(self,data,encoder_path=None):
        self.data=Path(data);self.docs=[json.loads(x) for x in (self.data/'documents.jsonl').read_text(encoding='utf-8').splitlines()]
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=15000,sublinear_tf=True)
        x=self.vectorizer.fit_transform([d['title']+' '+d.get('text','') for d in self.docs])
        self.svd=TruncatedSVD(n_components=min(64,x.shape[0]-1,x.shape[1]-1),random_state=42)
        self.vectors=normalize(self.svd.fit_transform(x));self.cache={}
    def encode(self,query):
        if query not in self.cache:self.cache[query]=normalize(self.svd.transform(self.vectorizer.transform([query])))[0]
        return self.cache[query]
    def search(self,query,k=3,title_only=False):
        text=query.split(' — ',1)[0] if title_only else query
        return [dict(document=self.docs[i],score=s) for i,s in cosine_rank(self.encode(text),self.vectors,k)]
