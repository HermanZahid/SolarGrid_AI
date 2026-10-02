from pathlib import Path
import json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]

def load_sources():
    return json.loads((ROOT/"data"/"sources.json").read_text(encoding="utf-8"))

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def read_pdf(path):
    reader = PdfReader(str(path))
    return "\n".join(clean_text(p.extract_text() or "") for p in reader.pages)

def chunk_text(text, words=900, overlap=150):
    items = text.split()
    step = max(1, words-overlap)
    return [" ".join(items[i:i+words]) for i in range(0, len(items), step)]

class LocalRAG:
    def __init__(self):
        self.documents, texts = [], []
        for src in load_sources():
            path = ROOT/src["local_file"]
            if path.is_file() and path.suffix.lower()==".pdf":
                for idx, chunk in enumerate(chunk_text(read_pdf(path))):
                    self.documents.append({**{k:src[k] for k in
                        ["id","title","authority","category","status","date","url"]},
                        "chunk_id":idx, "text":chunk})
                    texts.append(chunk)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=60000)
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query, k=5):
        if self.matrix is None:
            return []
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        out=[]
        for i in np.argsort(scores)[::-1][:k]:
            if scores[i] <= 0: continue
            item=dict(self.documents[i]); item["score"]=round(float(scores[i]),4); out.append(item)
        return out

    def status(self):
        return {"indexed_passages":len(self.documents),
                "indexed_sources":len(set(x["id"] for x in self.documents))}
