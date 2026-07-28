"""Scopus publication fetcher using the pybliometrics library."""

import os
from pathlib import Path

from pybliometrics.scopus import ScopusSearch, init
from tqdm import tqdm

from src.utils import (
    BaseAcademicSearcher,
    CacheManager,
    create_base_argparser,
    default_search_config_path,
    load_search_terms,
    run_academic_search,
)


class ScopusAnalyzer(BaseAcademicSearcher):
    def __init__(self, output_dir: Path = Path(".")):
        super().__init__(source_name="Scopus")
        self.output_dir = Path(output_dir)
        self.cache = None
        init()

    def fetch_article_counts(self, start_year, end_year, query):
        results = {}
        self.cache = CacheManager(self.output_dir, "scopus", query)

        for year in tqdm(range(start_year, end_year + 1), desc="Fetching Scopus"):
            cached = self.cache.get(year)
            if cached is not None:
                results[year] = cached
                continue

            year_query = f"{query} AND PUBYEAR IS {year}"

            try:
                search = ScopusSearch(year_query, refresh=True)
            except Exception as e:
                print(f"Error searching Scopus for year {year}: {e}")
                continue

            raw_results = search.results or []
            articles = [self._extract_result(r, year) for r in raw_results]

            year_data = {
                "count": len(articles),
                "articles": articles,
                "checked": True,
            }
            results[year] = year_data
            self.cache.set(year, year_data)

        return results

    def _extract_result(self, result, year) -> dict:
        """Extract standardized fields from one ScopusSearch result namedtuple."""
        return {
            "id": result.eid or "",
            "title": result.title or "",
            "authors": result.author_names or "",
            "venue": result.publicationName or "",
            "abstract": result.description or "",
            "url": f"https://doi.org/{result.doi}" if result.doi else "",
            "doi": result.doi or "",
            "pub_date": result.coverDate or str(year),
            "citations": int(result.citedby_count or 0),
            "keywords": result.authkeywords or "",
        }


def _build_scopus_query(config: dict) -> str:
    """Build a Scopus TITLE-ABS-KEY query from search config terms."""
    parts = []
    for key in ["brain_terms", "diagnostic_terms", "imaging_terms", "ai_terms"]:
        terms = config.get(key, [])
        if terms:
            quoted = [f'"{t}"' for t in terms]
            parts.append(f"TITLE-ABS-KEY({' OR '.join(quoted)})")

    return " AND ".join(parts)


def main():
    parser = create_base_argparser("Scopus publication analysis tool")
    args = parser.parse_args()

    config_path = args.config or default_search_config_path()
    config = load_search_terms(config_path)

    analyzer = ScopusAnalyzer(output_dir=args.output)
    query = _build_scopus_query(config)

    run_academic_search(analyzer, args, config, query)


if __name__ == "__main__":
    main()
