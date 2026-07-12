from __future__ import annotations

import argparse
import json

from .rag import hybrid_search, retriever, vector_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search railway education chunks with pgvector.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="vector")
    parser.add_argument("--approved-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "bm25":
        results = retriever.search(args.query, top_k=args.top_k, approved_only=args.approved_only)
    elif args.mode == "hybrid":
        results = hybrid_search(args.query, top_k=args.top_k, approved_only=args.approved_only)
    else:
        results = vector_search(args.query, top_k=args.top_k, approved_only=args.approved_only)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
