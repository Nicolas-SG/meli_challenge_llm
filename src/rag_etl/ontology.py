"""Build the documentation ontology and the per-section entities dataset.

Source of truth (written by Claude after reading every training section):

    data/ontology_source/rows/*.jsonl   one line per section:
        {"key": "<path>#<section>",
         "entities": [{"name", "type"}],
         "relationships": [{"source", "relation", "target"}]}
    data/ontology_source/synonyms.json  {canonical entity name: [aliases]}

Outputs:

    data/ontology.json             entity types, relation types, merged
                                   relationships (with provenance) and synonyms
    data/dataset_entities.parquet  training sections (For_training=1,
                                   is_TO_DO=0) plus an ``entities`` column:
                                   the section's annotation as a JSON string
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ENTITY_TYPES = {
    "Service": "Deployable application, API or external system",
    "Endpoint": "HTTP endpoint, written as 'METHOD /path'",
    "Component": "Internal class, module, layer, consumer or pipeline step",
    "Datastore": "Database, schema, cluster, cache or stored dataset",
    "Table": "Database table",
    "Field": "Column, payload field, header or parameter",
    "Queue": "Message queue or topic",
    "Job": "Scheduled or cron job",
    "Process": "Business or operational flow / procedure",
    "Alert": "Monitoring alert",
    "Error": "Error code, exception or failure mode",
    "Environment": "Deployment environment or scope (prod, stage, test, ...)",
    "Technology": "Language, framework, library or platform technology",
    "Tool": "External tool, platform or communication channel",
    "Config": "Configuration file, key, variable, URL or repository",
    "Concept": "Domain concept, entity or status",
    "Team": "Team, organization or contact group",
    "Person": "Named person",
}

SCHEMA_VERSION = 1
GENERATED_BY = (
    "Claude (claude-opus-5-5) reading every section of data/sections.parquet "
    "with For_training=1 and is_TO_DO=0"
)


def section_key(path: str, section: str) -> str:
    return f"{path}#{section}"


def training_sections(sections: pd.DataFrame) -> pd.DataFrame:
    """Rows used for the ontology: kept for training and not a placeholder."""
    keep = (sections["For_training"] == 1) & (sections["is_TO_DO"] == 0)
    return sections[keep].reset_index(drop=True)


def load_annotations(source_dir: str | Path) -> dict[str, dict]:
    annotations: dict[str, dict] = {}
    for file in sorted((Path(source_dir) / "rows").glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["key"] in annotations:
                raise ValueError(f"duplicate annotation key: {row['key']}")
            annotations[row["key"]] = {
                "entities": row["entities"],
                "relationships": row["relationships"],
            }
    return annotations


def load_synonyms(source_dir: str | Path) -> dict[str, list[str]]:
    return json.loads((Path(source_dir) / "synonyms.json").read_text(encoding="utf-8"))


def validate(
    keys: list[str], annotations: dict[str, dict], synonyms: dict[str, list[str]]
) -> list[str]:
    """Return a list of problems (empty when the annotations are consistent)."""
    problems = []
    missing = sorted(set(keys) - set(annotations))
    extra = sorted(set(annotations) - set(keys))
    problems += [f"section without annotation: {k}" for k in missing]
    problems += [f"annotation for unknown section: {k}" for k in extra]

    types: dict[str, set[str]] = defaultdict(set)
    for key, ann in annotations.items():
        names = set()
        for ent in ann["entities"]:
            if ent["type"] not in ENTITY_TYPES:
                problems.append(f"{key}: unknown entity type {ent['type']!r}")
            types[ent["name"]].add(ent["type"])
            names.add(ent["name"])
        for rel in ann["relationships"]:
            for end in ("source", "target"):
                if rel[end] not in names:
                    problems.append(
                        f"{key}: relationship {end} {rel[end]!r} is not an entity of the row"
                    )
    problems += [
        f"entity {name!r} has several types: {sorted(t)}"
        for name, t in types.items()
        if len(t) > 1
    ]
    for canonical, aliases in synonyms.items():
        if canonical not in types:
            problems.append(f"synonym canonical {canonical!r} is not an entity")
        problems += [
            f"alias {a!r} of {canonical!r} is also an entity name"
            for a in aliases
            if a in types
        ]
    return problems


def build_ontology(
    sections: pd.DataFrame, annotations: dict[str, dict], synonyms: dict[str, list[str]]
) -> dict:
    """Merge per-section relationships into a de-duplicated ontology."""
    entity_type = {}
    entity_projects: dict[str, set[str]] = defaultdict(set)
    merged: dict[tuple, dict] = {}
    project_of = {
        section_key(p, s): proj
        for p, s, proj in zip(sections["path"], sections["section"], sections["project_name"])
    }
    for key, ann in annotations.items():
        project = project_of[key]
        for ent in ann["entities"]:
            entity_type[ent["name"]] = ent["type"]
            entity_projects[ent["name"]].add(project)
        for rel in ann["relationships"]:
            k = (rel["source"], rel["relation"], rel["target"])
            item = merged.setdefault(
                k,
                {
                    "source": rel["source"],
                    "source_type": None,
                    "relation": rel["relation"],
                    "target": rel["target"],
                    "target_type": None,
                    "projects": set(),
                    "provenance": [],
                },
            )
            item["projects"].add(project)
            if key not in item["provenance"]:
                item["provenance"].append(key)

    relationships = []
    for item in merged.values():
        item["source_type"] = entity_type[item["source"]]
        item["target_type"] = entity_type[item["target"]]
        item["projects"] = sorted(item["projects"])
        item["count"] = len(item["provenance"])
        relationships.append(item)
    relationships.sort(key=lambda r: (r["source"], r["relation"], r["target"]))

    relation_counts = Counter(r["relation"] for r in relationships)
    entity_counts = Counter(entity_type.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "entity_types": ENTITY_TYPES,
        "relation_types": dict(sorted(relation_counts.items())),
        "relationships": relationships,
        "synonyms": {
            canonical: {
                "type": entity_type[canonical],
                "projects": sorted(entity_projects[canonical]),
                "aliases": aliases,
            }
            for canonical, aliases in sorted(synonyms.items())
        },
        "stats": {
            "sections": len(annotations),
            "entities": len(entity_type),
            "entities_by_type": dict(sorted(entity_counts.items())),
            "relationships": len(relationships),
            "canonical_with_synonyms": len(synonyms),
            "aliases": sum(len(a) for a in synonyms.values()),
        },
    }


def attach_entities(sections: pd.DataFrame, annotations: dict[str, dict]) -> pd.DataFrame:
    """Add the ``entities`` JSON-string column (entities + local relationships)."""
    out = sections.copy()
    keys = [section_key(p, s) for p, s in zip(out["path"], out["section"])]
    out["entities"] = pd.Series(
        [json.dumps(annotations[k], ensure_ascii=False) for k in keys],
        index=out.index,
        dtype="string",
    )
    return out


def build_entity_dataset(
    sections_path: str | Path,
    source_dir: str | Path,
    ontology_path: str | Path,
    dataset_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Validate the annotations, then write ontology.json and the entities parquet."""
    sections = training_sections(pd.read_parquet(sections_path))
    annotations = load_annotations(source_dir)
    synonyms = load_synonyms(source_dir)
    keys = [section_key(p, s) for p, s in zip(sections["path"], sections["section"])]
    problems = validate(keys, annotations, synonyms)
    if problems:
        raise ValueError(f"{len(problems)} annotation problems:\n" + "\n".join(problems))

    ontology = build_ontology(sections, annotations, synonyms)
    dataset = attach_entities(sections, annotations)

    Path(ontology_path).parent.mkdir(parents=True, exist_ok=True)
    Path(ontology_path).write_text(
        json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dataset.to_parquet(dataset_path, index=False)
    return dataset, ontology


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sections", default="data/sections.parquet")
    parser.add_argument("--source-dir", default="data/ontology_source")
    parser.add_argument("--ontology", default="data/ontology.json")
    parser.add_argument("--dataset", default="data/dataset_entities.parquet")
    args = parser.parse_args(argv)

    dataset, ontology = build_entity_dataset(
        args.sections, args.source_dir, args.ontology, args.dataset
    )
    print(
        f"Wrote {args.ontology} ({ontology['stats']}) and "
        f"{args.dataset} ({len(dataset)} rows)"
    )


if __name__ == "__main__":
    main()
