from pathlib import Path
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = ROOT / "data" / "knowledge_base"
SOURCES_FILE = ROOT / "data" / "sources.json"


def load_sources():
    """Load registered source metadata."""
    if not SOURCES_FILE.exists():
        return []

    try:
        return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def clean_text(text):
    """Normalize whitespace while preserving readable text."""
    return re.sub(r"\s+", " ", text or "").strip()


def parse_markdown_metadata(text):
    """
    Extract simple metadata from the Markdown source files.

    Expected format:

    - **Authority:** ...
    - **Document:** ...
    - **Date:** ...
    - **Category:** ...
    - **Status:** ...
    - **Official Source:** ...
    """

    metadata = {}

    patterns = {
        "authority": r"\*\*Authority:\*\*\s*(.+)",
        "title": r"\*\*Document:\*\*\s*(.+)",
        "date": r"\*\*Date:\*\*\s*(.+)",
        "category": r"\*\*Category:\*\*\s*(.+)",
        "status": r"\*\*Status:\*\*\s*(.+)",
        "url": r"\*\*Official Source:\*\*\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metadata[key] = match.group(1).strip()

    return metadata


def remove_metadata_block(text):
    """
    Remove the metadata section from the searchable content.

    We don't want terms such as 'Pakistan', 'NEPRA', or '2026'
    appearing repeatedly just because they occur in metadata.
    """

    match = re.search(
        r"## Source Metadata(.*?)(?=\n## |\Z)",
        text,
        flags=re.DOTALL,
    )

    if match:
        text = text[:match.start()] + text[match.end():]

    return text.strip()


def split_into_sections(text):
    """
    Split Markdown into logical sections.

    This is preferable to blindly splitting every document
    into arbitrary word counts because headings carry meaning.
    """

    parts = re.split(r"\n(?=## )", text)

    sections = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        lines = part.splitlines()

        heading = ""
        if lines and lines[0].startswith("## "):
            heading = lines[0].replace("## ", "").strip()

        body = "\n".join(lines[1:]).strip() if heading else part

        body = clean_text(body)

        if body:
            sections.append(
                {
                    "section": heading or "General",
                    "text": body,
                }
            )

    return sections


def read_markdown(path):
    """Read a Markdown knowledge-base file."""
    return path.read_text(encoding="utf-8")


class LocalRAG:
    """
    Lightweight local RAG engine.

    Current retrieval method:
        TF-IDF + cosine-style similarity

    This is intentionally zero-cost and lightweight.

    Later we can add local semantic embeddings without
    changing the agent architecture.
    """

    def __init__(self):
        self.documents = []
        texts = []

        self._load_knowledge_base()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=60000,
            sublinear_tf=True,
        )

        if texts := [doc["text"] for doc in self.documents]:
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = None

    def _load_knowledge_base(self):
        """Discover and index Markdown/text knowledge-base files."""

        if not KNOWLEDGE_BASE.exists():
            return

        for path in sorted(KNOWLEDGE_BASE.rglob("*")):

            if not path.is_file():
                continue

            if path.suffix.lower() not in {".md", ".txt"}:
                continue

            try:
                raw_text = read_markdown(path)
            except OSError:
                continue

            if not raw_text.strip():
                continue

            metadata = parse_markdown_metadata(raw_text)

            searchable_text = remove_metadata_block(raw_text)

            sections = split_into_sections(searchable_text)

            for idx, section in enumerate(sections):

                document = {
                    "source_file": str(path.relative_to(ROOT)),
                    "chunk_id": idx,
                    "section": section["section"],
                    "text": section["text"],
                    "title": metadata.get("title", path.stem),
                    "authority": metadata.get("authority", ""),
                    "category": metadata.get("category", ""),
                    "status": metadata.get("status", ""),
                    "date": metadata.get("date", ""),
                    "url": metadata.get("url", ""),
                }

                self.documents.append(document)

    def search(self, query, k=5, authority=None, category=None):
        """
        Retrieve the most relevant evidence.

        Optional filters:
            authority
            category
        """

        if self.matrix is None or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])

        scores = (self.matrix @ query_vector.T).toarray().ravel()

        candidate_indices = np.argsort(scores)[::-1]

        results = []

        for index in candidate_indices:

            score = float(scores[index])

            if score <= 0:
                continue

            document = self.documents[index]

            if authority:
                if authority.lower() not in document["authority"].lower():
                    continue

            if category:
                if category.lower() not in document["category"].lower():
                    continue

            result = dict(document)
            result["score"] = round(score, 4)

            results.append(result)

            if len(results) >= k:
                break

        return results

    def get_context(self, query, k=5):
        """
        Retrieve evidence formatted for an LLM prompt.

        This does NOT call Groq.
        """

        results = self.search(query, k=k)

        if not results:
            return "", []

        context_blocks = []

        for i, item in enumerate(results, start=1):

            context_blocks.append(
                f"""
[S{i}]
Title: {item['title']}
Authority: {item['authority']}
Date: {item['date']}
Section: {item['section']}
Source: {item['url']}

Evidence:
{item['text']}
""".strip()
            )

        return "\n\n".join(context_blocks), results

    def status(self):
        """Return useful diagnostics for the Streamlit UI."""

        sources = {
            document["title"]
            for document in self.documents
            if document.get("title")
        }

        return {
            "indexed_passages": len(self.documents),
            "indexed_sources": len(sources),
            "knowledge_base_path": str(KNOWLEDGE_BASE),
        }
