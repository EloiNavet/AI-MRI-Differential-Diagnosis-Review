import time
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils import (
    BaseAcademicSearcher,
    CacheManager,
    create_base_argparser,
    default_search_config_path,
    load_search_terms,
    run_academic_search,
)


class OpenAlexAnalyzer(BaseAcademicSearcher):
    def __init__(self, email: str, output_dir: Path = Path(".")):
        super().__init__(source_name="OpenAlex")
        self.email = email
        self.output_dir = Path(output_dir)
        self.OPENALEX_URL = "https://api.openalex.org/works"
        self.cache = None

    def fetch_article_counts(self, start_year, end_year, query):
        results = {}
        self.cache = CacheManager(self.output_dir, "openalex", query)

        session = requests.Session()
        session.headers.update({"User-Agent": f"mailto:{self.email}"})

        for year in tqdm(range(start_year, end_year + 1), desc="Fetching OpenAlex"):
            cached = self.cache.get(year)
            if cached is not None:
                results[year] = cached
                continue

            params = {
                "filter": f"publication_year:{year},type:article,is_paratext:false,is_retracted:false,has_abstract:true,title_and_abstract.search:{query}",
                "per-page": 200,
                "cursor": "*",
            }

            articles = []
            total_count = 0
            next_page = True

            while next_page:
                data = None

                for attempt in range(3):
                    try:
                        response = session.get(
                            self.OPENALEX_URL, params=params, timeout=30
                        )

                        if response.status_code == 429:
                            retry_after = response.headers.get("Retry-After")
                            wait_s = float(retry_after) if retry_after else 2.0
                            time.sleep(wait_s)
                            continue

                        if response.status_code != 200:
                            print(f"Error {response.status_code} for year {year}")
                            time.sleep(1)
                            continue

                        data = response.json()
                        break
                    except Exception as e:
                        print(f"Network error year {year}: {e}")
                        time.sleep(1 + attempt)

                if not data:
                    break

                if "meta" in data and total_count == 0:
                    total_count = data["meta"]["count"]

                for work in data.get("results", []):
                    articles.append(self._extract_work_data(work, year))

                next_page = False
                if "meta" in data and "next_cursor" in data["meta"]:
                    cursor = data["meta"]["next_cursor"]
                    if cursor and cursor != params["cursor"]:
                        params["cursor"] = cursor
                        next_page = True

                if not data.get("results"):
                    next_page = False

            year_data = {"count": total_count, "articles": articles, "checked": True}
            results[year] = year_data
            self.cache.set(year, year_data)

        return results

    def _extract_work_data(self, work, year):
        """Extract and clean data from a raw OpenAlex work object."""
        authors_list = [
            auth["author"]["display_name"]
            for auth in work.get("authorships", [])
            if "author" in auth and "display_name" in auth["author"]
        ]

        venue = "Unknown"
        if work.get("primary_location") and work["primary_location"].get("source"):
            venue = work["primary_location"]["source"].get("display_name", "Unknown")

        keywords = [
            c["display_name"] for c in work.get("concepts", []) if "display_name" in c
        ]

        doi_url = work.get("doi", "")
        doi = doi_url.replace("https://doi.org/", "") if doi_url else ""

        return {
            "id": work.get("id", "").split("/")[-1],
            "title": work.get("display_name", ""),
            "authors": ", ".join(authors_list),
            "venue": venue,
            "abstract": self._reconstruct_abstract(work),
            "url": doi_url or work.get("id", ""),
            "doi": doi,
            "pub_date": work.get("publication_date", f"{year}-01-01"),
            "citations": work.get("cited_by_count", 0),
            "keywords": ", ".join(keywords),
        }

    def _reconstruct_abstract(self, work):
        """Rebuild abstract from OpenAlex inverted index format."""
        idx = work.get("abstract_inverted_index")
        if not idx:
            return ""
        try:
            length = max(max(p) for p in idx.values()) + 1
            words = [""] * length
            for word, positions in idx.items():
                for pos in positions:
                    if pos < length:
                        words[pos] = word
            return " ".join(words)
        except Exception:
            return ""


def main():
    parser = create_base_argparser("OpenAlex publication analysis tool")
    args = parser.parse_args()

    config_path = args.config or default_search_config_path()
    config = load_search_terms(config_path)

    analyzer = OpenAlexAnalyzer(
        email=config.get("email", ""),
        output_dir=args.output,
    )

    query_parts = []
    for key in ["brain_terms", "diagnostic_terms", "imaging_terms", "ai_terms"]:
        terms = config.get(key, [])
        if terms:
            quoted = [f'"{t}"' for t in terms]
            query_parts.append(f"({' OR '.join(quoted)})")

    query = " AND ".join(query_parts)

    run_academic_search(analyzer, args, config, query)


if __name__ == "__main__":
    main()
