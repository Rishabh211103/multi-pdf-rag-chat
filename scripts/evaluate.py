"""Retrieval evaluation: hit-rate@k and MRR over a labelled question set.

Usage (from the project root, after indexing your PDFs):
    python -m scripts.evaluate eval/questions.example.json

Each item: {"question": "...", "source": "file.pdf", "page": 3}   ("page" is optional)
A retrieved chunk counts as a hit if its filename (and page, when given) matches.
"""

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.dependencies import get_vector_store
from app.rag import retrieve


def is_match(metadata: dict, item: dict) -> bool:
    if metadata["source"] != item["source"]:
        return False
    return "page" not in item or metadata["page"] == item["page"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=None, help="Override RETRIEVAL_K")
    parser.add_argument("--no-mmr", action="store_true", help="Use plain similarity search")
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.retrieval_k
    use_mmr = settings.use_mmr and not args.no_mmr
    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    store = get_vector_store()

    hits, reciprocal_ranks = 0, []
    for item in items:
        docs = retrieve(store, item["question"], k=k, fetch_k=settings.retrieval_fetch_k, use_mmr=use_mmr)
        rank = next((i for i, d in enumerate(docs, start=1) if is_match(d.metadata, item)), None)
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        print(f"{'HIT ' if rank else 'MISS'} rank={rank or '-':<3} {item['question']}")

    n = len(items)
    print(f"\nQuestions: {n} | k={k} | MMR={use_mmr}")
    print(f"Hit-rate@{k}: {hits / n:.1%}")
    print(f"MRR:         {sum(reciprocal_ranks) / n:.3f}")


if __name__ == "__main__":
    main()
