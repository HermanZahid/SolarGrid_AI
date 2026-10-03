from pathlib import Path
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = ROOT / "data" / "knowledge_base"
SOURCES_FILE = ROOT / "data" / "sources.json"


# These sections contain SolarGrid AI guidance rather than
# source-derived evidence. They are excluded from retrieval
# so the LLM does not mistake application instructions for
# regulatory requirements.
NON_EVIDENCE_SECTIONS = {
    "evidence classification",
    "regulatory significance for solargrid ai",
    "project-specific interpretation for solargrid ai",
    "relationship with other regulatory sources",
    "limitations",
    "project classification rule",
    "important agent rule",
    "role in solargrid ai",
    "relevance to solargrid ai",
    "source reference",
}


def load_sources():
    """Load the optional source registry."""
    if not SOURCES_FILE.exists():
        return []

    try:
        return json.loads(
            SOURCES_FILE.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return []


def clean_text(text):
    """Normalize Markdown/text for retrieval and display."""
    text = text or ""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _first_metadata_value(text, patterns):
    """
    Return the first metadata value matching any supplied pattern.
    """
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

    return ""


def parse_markdown_metadata(text):
    """
    Extract metadata while supporting both current metadata formats.

    Supported variants include:

        **Authority:** ...
        **Issuing authority:** ...

        **Document:** ...
        **Source document:** ...

        **Official Source:** ...
        **Source:** ...
    """
    return {
        "authority": _first_metadata_value(
            text,
            [
                r"\*\*Authority:\*\*\s*(.+)",
                r"\*\*Issuing authority:\*\*\s*(.+)",
            ],
        ),
        "title": _first_metadata_value(
            text,
            [
                r"\*\*Document:\*\*\s*(.+)",
                r"\*\*Source document:\*\*\s*(.+)",
            ],
        ),
        "date": _first_metadata_value(
            text,
            [
                r"\*\*Date:\*\*\s*(.+)",
            ],
        ),
        "category": _first_metadata_value(
            text,
            [
                r"\*\*Category:\*\*\s*(.+)",
                r"\*\*Document type:\*\*\s*(.+)",
            ],
        ),
        "status": _first_metadata_value(
            text,
            [
                r"\*\*Status:\*\*\s*(.+)",
            ],
        ),
        "url": _first_metadata_value(
            text,
            [
                r"\*\*Official Source:\*\*\s*(.+)",
                r"\*\*Source:\*\*\s*(.+)",
            ],
        ),
    }


def remove_metadata_block(text):
    """
    Remove either 'Source Metadata' or legacy 'Metadata'
    sections before creating searchable content.
    """
    match = re.search(
        r"(?ms)^##\s+(?:Source Metadata|Metadata)\s*$.*?"
        r"(?=^---\s*$|^##\s+|\Z)",
        text,
    )

    if match:
        text = (
            text[:match.start()]
            + text[match.end():]
        )

    return text.strip()


def split_into_sections(text):
    """
    Split a Markdown document at level-2 headings.

    Very short sections are ignored because they generally contain
    little useful retrieval content.
    """
    parts = re.split(
        r"\n(?=## )",
        text,
    )

    sections = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        lines = part.splitlines()
        heading = ""

        if lines and lines[0].startswith("## "):
            heading = (
                lines[0]
                .replace("## ", "")
                .strip()
            )

        body = (
            "\n".join(lines[1:]).strip()
            if heading
            else part
        )

        body = clean_text(body)

        if len(body.split()) < 12:
            continue

        sections.append(
            {
                "section": heading or "General",
                "text": body,
            }
        )

    return sections


def is_evidence_section(section_name):
    """
    Return True when the section is suitable for evidence retrieval.

    SolarGrid AI guidance sections are excluded so that they cannot
    be mistaken for direct regulatory/source requirements.
    """
    normalized = clean_text(section_name).lower()
    return normalized not in NON_EVIDENCE_SECTIONS


def read_markdown(path):
    """Read a Markdown/text knowledge-base file."""
    return path.read_text(encoding="utf-8")


class LocalRAG:
    def __init__(self):
        self.documents = []
        self._load_knowledge_base()

        # Internal retrieval text includes the document title,
        # section heading, and evidence text.
        search_texts = [
            document["search_text"]
            for document in self.documents
        ]

        # Do NOT remove English stop words.
        #
        # In regulatory text, words such as:
        #   must
        #   shall
        #   may
        #   not
        #
        # can materially change the meaning of a requirement.
        self.vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),
            max_features=60000,
            sublinear_tf=True,
        )

        self.matrix = (
            self.vectorizer.fit_transform(search_texts)
            if search_texts
            else None
        )

    def _load_knowledge_base(self):
        """
        Load .md and .txt files from the local knowledge base.

        Each retained section becomes one retrievable evidence item.
        """
        if not KNOWLEDGE_BASE.exists():
            return

        for path in sorted(
            KNOWLEDGE_BASE.rglob("*")
        ):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".md",
                ".txt",
            }:
                continue

            try:
                raw_text = read_markdown(path)
            except OSError:
                continue

            if not raw_text.strip():
                continue

            metadata = parse_markdown_metadata(
                raw_text
            )

            searchable_text = remove_metadata_block(
                raw_text
            )

            sections = split_into_sections(
                searchable_text
            )

            for idx, section in enumerate(
                sections
            ):
                # Exclude SolarGrid AI guidance sections from
                # the evidence retrieval layer.
                if not is_evidence_section(
                    section["section"]
                ):
                    continue

                title = metadata.get(
                    "title",
                    "",
                ).strip()

                if not title:
                    title = path.stem

                # Internal retrieval representation.
                # This is used for better matching but is not
                # exposed as evidence to the UI/LLM.
                search_text = clean_text(
                    f"{title} "
                    f"{section['section']} "
                    f"{section['text']}"
                )

                self.documents.append(
                    {
                        "source_file": str(
                            path.relative_to(ROOT)
                        ),
                        "chunk_id": idx,
                        "section": section["section"],
                        "text": section["text"],
                        "title": title,
                        "authority": metadata.get(
                            "authority",
                            "",
                        ),
                        "category": metadata.get(
                            "category",
                            "",
                        ),
                        "status": metadata.get(
                            "status",
                            "",
                        ),
                        "date": metadata.get(
                            "date",
                            "",
                        ),
                        "url": metadata.get(
                            "url",
                            "",
                        ),
                        "evidence_type": (
                            "curated source summary"
                        ),
                        "search_text": search_text,
                    }
                )

    def search(
        self,
        query,
        k=5,
        authority=None,
        category=None,
    ):
        """
        Retrieve the top-k evidence passages using
        TF-IDF cosine similarity.
        """
        if (
            self.matrix is None
            or not query
            or not query.strip()
        ):
            return []

        query_vector = self.vectorizer.transform(
            [query]
        )

        scores = (
            self.matrix @ query_vector.T
        ).toarray().ravel()

        candidate_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in candidate_indices:
            score = float(
                scores[index]
            )

            if score <= 0:
                continue

            document = self.documents[index]

            if (
                authority
                and authority.lower()
                not in document["authority"].lower()
            ):
                continue

            if (
                category
                and category.lower()
                not in document["category"].lower()
            ):
                continue

            # Copy the document so the internal search_text
            # does not leak into the returned result.
            result = dict(document)
            result.pop(
                "search_text",
                None,
            )

            result["score"] = round(
                score,
                4,
            )

            results.append(result)

            if len(results) >= k:
                break

        return results

    def get_context(
        self,
        query,
        k=5,
    ):
        """
        Return LLM-ready evidence context plus the underlying
        evidence records.
        """
        results = self.search(
            query,
            k=k,
        )

        if not results:
            return "", []

        context_blocks = []

        for i, item in enumerate(
            results,
            start=1,
        ):
            context_blocks.append(
                f"""
[S{i}]
Title: {item['title']}
Authority: {item['authority']}
Date: {item['date']}
Section: {item['section']}
Evidence type: {item['evidence_type']}
Official Source: {item['url']}

Evidence:
{item['text']}
""".strip()
            )

        return (
            "\n\n".join(context_blocks),
            results,
        )

    def status(self):
        """Return basic knowledge-base diagnostics."""
        sources = {
            document["title"]
            for document in self.documents
            if document.get("title")
        }

        return {
            "indexed_passages": len(
                self.documents
            ),
            "indexed_sources": len(
                sources
            ),
            "knowledge_base_path": str(
                KNOWLEDGE_BASE
            ),
        }
