from pathlib import Path

import arxiv
from tqdm import tqdm

from src.utils import (
    BaseAcademicSearcher,
    CacheManager,
    create_base_argparser,
    default_search_config_path,
    load_search_terms,
    run_academic_search,
)


class ArxivAnalyzer(BaseAcademicSearcher):
    def __init__(self, output_dir: Path = Path(".")):
        super().__init__(source_name="arXiv")
        self.client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
        self.output_dir = Path(output_dir)
        self.cache = None

    def fetch_article_counts(self, start_year, end_year, query):
        """Fetch yearly article counts from arXiv using cache-backed requests."""
        results = {}
        self.cache = CacheManager(self.output_dir, "arxiv", query)

        for year in tqdm(range(start_year, end_year + 1), desc="Fetching ArXiv"):
            cached = self.cache.get(year)
            if cached is not None:
                results[year] = cached
                continue

            year_specific_query = (
                f"{query} AND submittedDate:[{year}01010000 TO {year}12312359]"
            )

            search = arxiv.Search(
                query=year_specific_query,
                max_results=None,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            articles = []
            count = 0

            try:
                for paper in self.client.results(search):
                    articles.append(self._process_paper(paper, year))
                    count += 1

                    if count % 500 == 0:
                        print(f"  ...fetched {count} papers so far for {year}")

            except Exception as e:
                print(f"Error fetching data for year {year}: {e}")
                continue

            year_data = {"count": count, "articles": articles, "checked": True}
            results[year] = year_data
            self.cache.set(year, year_data)

        return results

    def _process_paper(self, paper, year):
        """Extract standardized fields from one arXiv result item."""
        arxiv_id = paper.entry_id.split("/")[-1] if paper.entry_id else ""
        authors = ", ".join(author.name for author in paper.authors)
        primary = paper.primary_category if paper.primary_category else "Unknown"
        abstract = paper.summary.replace("\n", " ").strip() if paper.summary else ""

        pub_date = f"{year}-01-01"
        if paper.published:
            pub_date = paper.published.strftime("%Y-%m-%d")

        return {
            "id": arxiv_id,
            "title": paper.title.replace("\n", " ").strip(),
            "authors": authors,
            "venue": f"arXiv:{primary}",
            "abstract": abstract,
            "url": paper.pdf_url or paper.entry_id,
            "doi": paper.doi or "",
            "pub_date": pub_date,
            "citations": 0,
            "keywords": ", ".join(paper.categories),
        }


def main():
    parser = create_base_argparser("arXiv publication analysis tool")
    args = parser.parse_args()

    config_path = args.config or default_search_config_path()
    config = load_search_terms(config_path)

    analyzer = ArxivAnalyzer(output_dir=args.output)

    def format_arxiv_terms(terms):
        formatted = []
        for term in terms:
            clean_term = term.strip()
            if " " in clean_term:
                formatted.append(f'"{clean_term}"')
            else:
                formatted.append(clean_term)
        return formatted

    query_parts = []
    for key in ["brain_terms", "diagnostic_terms", "imaging_terms", "ai_terms"]:
        terms = config.get(key, [])
        if terms:
            formatted = format_arxiv_terms(terms)
            query_parts.append(f"({' OR '.join(formatted)})")

    query = " AND ".join(query_parts)

    if not query:
        print("Error: No valid search terms found.")
        return

    run_academic_search(analyzer, args, config, query)


if __name__ == "__main__":
    main()
