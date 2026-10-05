"""Retrieval-augmented generation over the documentation sections.

Indexing:
    1. The ontology LoRA (Qwen2.5-0.5B-Instruct + data/ontology_lora.bin)
       extracts entities from every indexed section.
    2. Extracted names are canonicalized with the synonyms in ontology.json.
    3. Sections (plus their entity list) are embedded with BAAI/bge-m3 and
       stored in a LangChain Chroma collection (cosine space); the entities and
       provenance (path, section, project, version) go into the metadata.

Retrieval:
    the question is embedded and the top ``fetch_k`` sections are fetched;
    ontology entities mentioned in the question (by name or alias) boost the
    sections that contain them, and their relationships from ontology.json are
    added to the prompt as "known facts".

Generation:
    Qwen2.5-Coder-0.5B-Instruct, with or without the memory LoRA, answers from
    the retrieved context (or from its own weights when there is no retriever).
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EMBED_MODEL = "BAAI/bge-m3"
REFUSAL = "I don't have information about that topic"
SYSTEM_PROMPT = (
    "You are an assistant that answers questions about internal software "
    "documentation. Answer in one or two short sentences. If you do not know the "
    f'answer, reply exactly: "{REFUSAL}".'
)
SYSTEM_PROMPT_RAG = (
    "You are an assistant that answers questions about internal software "
    "documentation using only the provided context. Answer in one or two short "
    "sentences. If the context does not contain the answer, reply exactly: "
    f'"{REFUSAL}".'
)
# Entity types worth matching in free-text questions (fields like "name" or
# "status" would match almost everything).
QUERY_ENTITY_TYPES = {
    "Service", "Endpoint", "Component", "Datastore", "Table", "Queue", "Job",
    "Process", "Alert", "Error", "Technology", "Tool", "Team", "Person",
}


# --------------------------------------------------------------------------
# Ontology
# --------------------------------------------------------------------------


class OntologyIndex:
    """Canonical names, aliases and relationships from ontology.json."""

    def __init__(self, ontology: dict):
        self.types: dict[str, str] = {}
        self.facts: dict[str, list[dict]] = defaultdict(list)
        for rel in ontology["relationships"]:
            self.types[rel["source"]] = rel["source_type"]
            self.types[rel["target"]] = rel["target_type"]
            self.facts[rel["source"]].append(rel)
            self.facts[rel["target"]].append(rel)
        self.alias_to_canonical: dict[str, str] = {}
        for canonical, info in ontology["synonyms"].items():
            self.types.setdefault(canonical, info["type"])
            for alias in info["aliases"]:
                self.alias_to_canonical[alias.lower()] = canonical
        for name in self.types:
            self.alias_to_canonical.setdefault(name.lower(), name)
        self._patterns = [
            (re.compile(r"(?<![\w/-])" + re.escape(surface) + r"(?![\w/-])", re.I), canonical)
            for surface, canonical in sorted(self.alias_to_canonical.items(), key=lambda x: -len(x[0]))
            if self._matchable(surface, canonical)
        ]

    @classmethod
    def from_file(cls, path: str | Path) -> "OntologyIndex":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _matchable(self, surface: str, canonical: str) -> bool:
        return self.types.get(canonical) in QUERY_ENTITY_TYPES and len(surface) >= 3

    def canonical(self, name: str) -> str:
        return self.alias_to_canonical.get(name.lower(), name)

    def find_in_text(self, text: str) -> list[str]:
        """Canonical entities mentioned in ``text`` (longest names first)."""
        found: list[str] = []
        for pattern, canonical in self._patterns:
            if canonical not in found and pattern.search(text):
                found.append(canonical)
        return found

    def facts_for(self, entities: list[str], limit: int = 8) -> list[str]:
        lines, seen = [], set()
        for ent in entities:
            for rel in sorted(self.facts.get(ent, []), key=lambda r: -r["count"]):
                line = f"{rel['source']} {rel['relation'].replace('_', ' ')} {rel['target']}"
                if line not in seen:
                    seen.add(line)
                    lines.append(line)
                if len(lines) >= limit:
                    return lines
        return lines


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def section_key(row) -> str:
    return f"{row['path']}#{row['section']}"


def extract_section_entities(sections: pd.DataFrame, model, tokenizer, batch_size: int = 8,
                             max_new_tokens: int = 512) -> pd.DataFrame:
    """Run the ontology adapter on every section; returns key, raw output,
    parsed JSON (``extracted``) and whether it parsed."""
    from rag_etl.lora import generate_batch, instruct_prompt, parse_extraction

    prompts = [instruct_prompt(r) for _, r in sections.iterrows()]
    raw = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    parsed = [parse_extraction(text) for text in raw]
    return pd.DataFrame(
        {
            "key": [section_key(r) for _, r in sections.iterrows()],
            "raw_output": raw,
            "parsed": [p is not None for p in parsed],
            "extracted": [json.dumps(p or {"entities": [], "relationships": []}, ensure_ascii=False) for p in parsed],
        }
    )


def canonical_entities(extracted_json: str, onto: OntologyIndex) -> list[str]:
    names = [onto.canonical(e["name"]) for e in json.loads(extracted_json)["entities"]]
    return list(dict.fromkeys(names))


def entity_prf(gold_json: str, pred_json: str, onto: OntologyIndex) -> tuple[float, float, float]:
    """Precision / recall / F1 of extracted entity names vs the gold annotation
    (names canonicalized with the ontology synonyms, case-insensitive)."""
    gold = {n.lower() for n in canonical_entities(gold_json, onto)}
    pred = {n.lower() for n in canonical_entities(pred_json, onto)}
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def build_documents(sections: pd.DataFrame, extractions: pd.DataFrame, onto: OntologyIndex):
    from langchain_core.documents import Document

    by_key = dict(zip(extractions["key"], extractions["extracted"]))
    docs = []
    for _, r in sections.iterrows():
        key = section_key(r)
        entities = canonical_entities(by_key[key], onto)
        content = f"[{r['project_name']} {r['version']}] {r['file_name']} > {r['section']}\n{r['text']}"
        if entities:
            content += "\nEntities: " + ", ".join(entities)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "key": key,
                    "path": r["path"],
                    "file_name": r["file_name"],
                    "project_name": r["project_name"],
                    "version": r["version"],
                    "section": r["section"],
                    "entities": " | ".join(entities),
                },
            )
        )
    return docs


def load_embeddings(model_name: str = EMBED_MODEL, device: str | None = None):
    from langchain_huggingface import HuggingFaceEmbeddings

    from rag_etl.lora import default_device

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device or default_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def build_vector_store(docs, embeddings, persist_dir: str | Path, collection: str = "docs_sections"):
    """(Re)create the Chroma collection from ``docs`` (cosine distance)."""
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.reset_collection()
    store.add_documents(docs, ids=[d.metadata["key"] for d in docs])
    return store


# --------------------------------------------------------------------------
# Retrieval and generation
# --------------------------------------------------------------------------


@dataclass
class Retrieval:
    docs: list
    scores: list[float]
    query_entities: list[str]
    facts: list[str]

    @property
    def keys(self) -> list[str]:
        return [d.metadata["key"] for d in self.docs]


@dataclass
class OntologyRetriever:
    store: object
    onto: OntologyIndex
    k: int = 3
    fetch_k: int = 10
    entity_boost: float = 0.05
    max_facts: int = 8

    def retrieve(self, question: str) -> Retrieval:
        q_ents = self.onto.find_in_text(question)
        hits = self.store.similarity_search_with_relevance_scores(question, k=self.fetch_k)
        rescored = []
        for doc, score in hits:
            doc_ents = set(filter(None, doc.metadata.get("entities", "").split(" | ")))
            rescored.append((score + self.entity_boost * len(doc_ents & set(q_ents)), doc))
        rescored.sort(key=lambda x: -x[0])
        top = rescored[: self.k]
        return Retrieval(
            docs=[d for _, d in top],
            scores=[s for s, _ in top],
            query_entities=q_ents,
            facts=self.onto.facts_for(q_ents, self.max_facts),
        )


def format_context(retrieval: Retrieval, max_chars_per_doc: int = 2500) -> str:
    parts = []
    if retrieval.facts:
        parts.append("Known facts:\n" + "\n".join(f"- {f}" for f in retrieval.facts))
    for i, doc in enumerate(retrieval.docs, 1):
        parts.append(f"Document {i}:\n{doc.page_content[:max_chars_per_doc]}")
    return "\n\n".join(parts)


@dataclass
class Generator:
    """Qwen chat model; ``use_adapter=False`` runs the base weights of a PEFT model."""

    model: object
    tokenizer: object
    use_adapter: bool = True
    max_new_tokens: int = 160

    def answer(self, question: str, context: str | None = None) -> str:
        import contextlib

        import torch

        if context is None:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
        adapter_off = self.model.disable_adapter() if not self.use_adapter else contextlib.nullcontext()
        self.model.eval()
        with torch.no_grad(), adapter_off:
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@dataclass
class RAG:
    name: str
    generator: Generator
    retriever: OntologyRetriever | None = None

    def run(self, question: str) -> dict:
        start = time.perf_counter()
        retrieval = self.retriever.retrieve(question) if self.retriever else None
        context = format_context(retrieval) if retrieval else None
        answer = self.generator.answer(question, context)
        return {
            "answer": answer,
            "seconds": time.perf_counter() - start,
            "retrieved_keys": retrieval.keys if retrieval else [],
            "query_entities": retrieval.query_entities if retrieval else [],
        }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def cosine_similarity(a: list[str], b: list[str], embeddings) -> np.ndarray:
    """Row-wise cosine between two lists of texts (bge-m3, normalized)."""
    ea = np.asarray(embeddings.embed_documents(list(a)))
    eb = np.asarray(embeddings.embed_documents(list(b)))
    return (ea * eb).sum(axis=1) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1))


def evaluate_rags(rags: list[RAG], questions: list[dict], embeddings, log=print) -> pd.DataFrame:
    """Run every RAG on every question; cosine similarity to the expected answer
    and wall-clock seconds per answer (retrieval + generation). Each RAG answers
    one untimed warm-up question first so CUDA start-up is not counted."""
    rows = []
    for rag in rags:
        rag.run("warm-up question")
        start = time.perf_counter()
        for i, q in enumerate(questions):
            out = rag.run(q["prompt"])
            rows.append(
                {
                    "rag": rag.name,
                    "question_id": i,
                    "prompt": q["prompt"],
                    "expected": q["answer"],
                    "answerable": q["answerable"],
                    "source": q.get("source"),
                    "answer": out["answer"],
                    "seconds": out["seconds"],
                    "retrieved_keys": out["retrieved_keys"],
                    "source_retrieved": (
                        q["source"] in out["retrieved_keys"] if q.get("source") and rag.retriever else None
                    ),
                }
            )
        log(f"{rag.name}: {len(questions)} answers in {time.perf_counter() - start:.1f}s")
    results = pd.DataFrame(rows)
    results["cosine_similarity"] = cosine_similarity(results["answer"], results["expected"], embeddings)
    results["refused"] = results["answer"].str.contains(REFUSAL, case=False, regex=False)
    return results
