import hashlib
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from scholarly import ProxyGenerator, publication_parser, scholarly
from tqdm import tqdm

from src.utils import (
    BaseAcademicSearcher,
    CacheManager,
    create_base_argparser,
    default_search_config_path,
    load_search_terms,
    run_academic_search,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("scholarly").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class GoogleScholarAnalyzer(BaseAcademicSearcher):
    def __init__(
        self,
        use_proxy: bool = False,
        output_dir: str = ".",
        slow_mode: bool = False,
        checkpoint_every: int = 5,
        resume: bool = True,
    ):
        super().__init__("Google Scholar")
        self.output_dir = Path(output_dir)
        self.use_proxy = use_proxy
        self.slow_mode = slow_mode
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.cache = None
        self.proxy_active = False

        self.delay_min = 6.0 if slow_mode else 3.0
        self.delay_max = 12.0 if slow_mode else 6.0
        self.error_delay_min = 30 if slow_mode else 15
        self.error_delay_max = 60 if slow_mode else 30

        if use_proxy:
            self.api_key = os.getenv("SCRAPER_API_KEY")
            self.proxy = ProxyGenerator()
            self._setup_proxy()
            self.proxy_active = True
        else:
            self.api_key = None
            self.proxy = None

    def _setup_proxy(self):
        """Configure proxy for scholarly requests."""
        self.proxy = ProxyGenerator()
        if self.api_key:
            try:
                logging.info("Setting up ScraperAPI proxy")
                ok = self.proxy.ScraperAPI(API_KEY=self.api_key)
                if not ok:
                    logging.warning(
                        "ScraperAPI unavailable or quota reached. Falling back to free proxies."
                    )
                    self.proxy.FreeProxies()
            except Exception as e:
                logging.error(f"ScraperAPI setup failed: {e}")
                logging.info("Falling back to free proxy")
                self.proxy.FreeProxies()
        else:
            logging.info("Setting up free proxy")
            self.proxy.FreeProxies()

        scholarly.use_proxy(self.proxy)

    def _setup_free_proxy(self):
        """Enable free proxy mode dynamically when direct mode is blocked."""
        self.use_proxy = True
        self.api_key = None
        self._setup_proxy()
        self.proxy_active = True

    def fetch_article_counts(
        self,
        start_year: int,
        end_year: int,
        query: str,
        max_articles_per_year: int = 100,
    ) -> dict[str, dict[str, str]]:
        query_variants = [q.strip() for q in str(query).split("|||") if q.strip()]
        if not query_variants:
            query_variants = [str(query).strip()]

        results = {}
        cache_key = f"{query}_{max_articles_per_year}"
        self.cache = CacheManager(self.output_dir, "g_scholar", cache_key)

        logging.info(
            "Google Scholar fetch start | years=%s-%s | max_articles=%s | resume=%s",
            start_year,
            end_year,
            max_articles_per_year,
            self.resume,
        )

        for year in tqdm(range(start_year, end_year + 1), desc="Processing years"):
            cached = self.cache.get(year)
            existing_articles = []
            checked = False

            if cached is not None:
                existing_articles = cached.get("articles", []) or []
                checked = bool(cached.get("checked", False))

                if checked and self.resume:
                    logging.info(
                        "Year %s loaded from completed checkpoint (%s articles)",
                        year,
                        len(existing_articles),
                    )
                    results[year] = cached
                    continue

                if existing_articles and self.resume:
                    logging.info(
                        "Resuming year %s from partial checkpoint (%s articles)",
                        year,
                        len(existing_articles),
                    )

            articles = existing_articles if self.resume else []
            seen_keys = set(self._article_key(a) for a in articles)

            if len(articles) >= max_articles_per_year:
                year_data = {
                    "count": len(articles),
                    "articles": articles[:max_articles_per_year],
                    "checked": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                results[year] = year_data
                self.cache.set(year, year_data)
                continue

            max_retries = 3
            success = False
            starting_count = len(articles)

            for attempt in range(max_retries):
                try:
                    logging.info(
                        f"Fetching {year} (Attempt {attempt + 1}/{max_retries})"
                    )

                    search_query_iterator = scholarly.search_pubs(
                        query=query_variants[0],
                        year_low=year,
                        year_high=year,
                        sort_by="relevance",
                        include_last_year="everything",
                    )

                    completed = False
                    for variant_idx, variant in enumerate(query_variants, start=1):
                        if len(articles) >= max_articles_per_year:
                            completed = True
                            break

                        if variant_idx > 1:
                            logging.info(
                                "Year %s: switching to query variant %s/%s",
                                year,
                                variant_idx,
                                len(query_variants),
                            )
                            search_query_iterator = scholarly.search_pubs(
                                query=variant,
                                year_low=year,
                                year_high=year,
                                sort_by="relevance",
                                include_last_year="everything",
                            )

                        extracted_articles, variant_completed = self._extract_articles(
                            search_query_iterator,
                            max_articles_per_year,
                            year,
                            articles,
                            seen_keys,
                            self.checkpoint_every,
                        )
                        articles = extracted_articles

                        if len(articles) >= max_articles_per_year:
                            completed = True
                            break

                        if not variant_completed:
                            break

                    if len(articles) >= max_articles_per_year:
                        completed = True

                    fetched_new_articles = len(articles) > starting_count

                    if (
                        not fetched_new_articles
                        and not self.proxy_active
                        and attempt < (max_retries - 1)
                    ):
                        logging.warning(
                            "Year %s returned no new results in direct mode. Enabling free proxy and retrying.",
                            year,
                        )
                        self._setup_free_proxy()
                        time.sleep(random.uniform(3, 7))
                        continue

                    year_data = {
                        "count": len(articles),
                        "articles": articles,
                        "checked": completed,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    results[year] = year_data
                    self.cache.set(year, year_data)

                    success = True
                    break

                except Exception as e:
                    logging.error(f"Error fetching year {year}: {e}")
                    partial_data = {
                        "count": len(articles),
                        "articles": articles,
                        "checked": False,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    self.cache.set(year, partial_data)

                    if self.use_proxy:
                        logging.info("Rotating proxy and sleeping...")
                        self._setup_proxy()
                    time.sleep(
                        random.uniform(self.error_delay_min, self.error_delay_max)
                    )

            if not success:
                logging.error(
                    f"FAILED to fetch year {year} after {max_retries} attempts."
                )
                failed_data = {
                    "count": len(articles),
                    "articles": articles,
                    "checked": False,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                results[year] = failed_data
                self.cache.set(year, failed_data)

        return results

    @staticmethod
    def _article_key(article: dict) -> str:
        """Stable key for deduplication across retries."""
        article_id = str(article.get("id", "")).strip().lower()
        title = str(article.get("title", "")).strip().lower()
        year = str(article.get("pub_date", "")).strip().lower()
        if article_id:
            return f"id::{article_id}"
        return f"title::{title}::year::{year}"

    def _extract_article_data(self, article: dict) -> dict:
        """Extract relevant information from a scholarly article dict."""
        title = article["bib"].get("title", "")
        output = {
            "id": "+".join(article.get("author_id", []))
            or hashlib.md5(title.encode()).hexdigest()[:10],
            "title": title,
            "authors": ", ".join(article["bib"].get("author", [])),
            "venue": article["bib"].get("venue", "") + article["bib"].get("host", ""),
            "abstract": article["bib"].get("abstract", ""),
            "url": article.get("pub_url", ""),
            "pub_date": article["bib"].get("pub_year", 0),
            "citations": int(article.get("num_citations", 0)),
            "keywords": "",
        }
        try:
            output["pub_date"] = int(output["pub_date"])
        except (ValueError, TypeError):
            output["pub_date"] = None
        return output

    def _extract_articles(
        self,
        search_query: publication_parser._SearchScholarIterator,
        max_articles: int,
        year: int,
        existing_articles: list[dict[str, str]],
        seen_keys: set[str],
        checkpoint_every: int,
    ) -> list[dict[str, str]]:
        articles = list(existing_articles)
        count = len(articles)
        reached_end_of_results = False

        while count < max_articles:
            try:
                article = next(search_query)
                article_data = self._extract_article_data(article)

                pub_date = article_data.get("pub_date")
                if pub_date is None or pub_date == year:
                    key = self._article_key(article_data)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        articles.append(article_data)
                        count += 1

                        if checkpoint_every > 0 and count % checkpoint_every == 0:
                            checkpoint_data = {
                                "count": len(articles),
                                "articles": articles,
                                "checked": False,
                                "updated_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            }
                            self.cache.set(year, checkpoint_data)
                            logging.info(
                                "Checkpoint saved for year %s (%s articles)",
                                year,
                                len(articles),
                            )

                time.sleep(random.uniform(self.delay_min, self.delay_max))

            except StopIteration:
                reached_end_of_results = True
                break
            except Exception as e:
                logging.warning(f"Critical error during article iteration: {e}")
                raise

        completed = (count >= max_articles) or reached_end_of_results
        return articles, completed


def main():
    load_dotenv()

    parser = create_base_argparser("Google Scholar publication analysis tool")
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="Enable proxy usage (requires SCRAPER_API_KEY or uses free proxies)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save partial per-year checkpoint every N kept articles (default: 5)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing per-year checkpoints and fetch from scratch",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slower delays for background execution (reduces throttling risk)",
    )
    args = parser.parse_args()

    config_path = args.config or default_search_config_path()
    config = load_search_terms(config_path)

    brain_terms = "|".join(
        [f'"{term.replace("-", " ")}"' for term in config["brain_terms"]]
    )
    diagnostic_terms = "|".join(
        [f'"{term.replace("-", " ")}"' for term in config["diagnostic_terms"]]
    )
    imaging_terms = "|".join(
        [f'"{term.replace("-", " ")}"' for term in config["imaging_terms"]]
    )
    ai_terms = "|".join([f'"{term.replace("-", " ")}"' for term in config["ai_terms"]])

    query = f"({brain_terms}) ({diagnostic_terms}) ({imaging_terms}) ({ai_terms})"

    logging.info(f"Generated Query: {query}")

    analyzer = GoogleScholarAnalyzer(
        use_proxy=args.use_proxy,
        output_dir=args.output,
        slow_mode=args.slow,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )

    run_academic_search(analyzer, args, config, query)


if __name__ == "__main__":
    main()
