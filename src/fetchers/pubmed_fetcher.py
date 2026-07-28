import os
import time
from pathlib import Path

from Bio import Entrez
from tqdm import tqdm

from src.utils import (
    BaseAcademicSearcher,
    CacheManager,
    create_base_argparser,
    default_search_config_path,
    load_search_terms,
    run_academic_search,
)


class PubMedAnalyzer(BaseAcademicSearcher):
    def __init__(self, email: str, output_dir: Path = Path(".")):
        super().__init__(source_name="PubMed")
        self.email = email
        self.output_dir = Path(output_dir)

        Entrez.email = email
        api_key = os.getenv("NCBI_API_KEY")
        if api_key:
            Entrez.api_key = api_key

        self.cache = None

    def fetch_article_counts(self, start_year, end_year, query):
        results = {}
        self.cache = CacheManager(self.output_dir, "pubmed", query)

        for year in tqdm(range(start_year, end_year + 1), desc="Fetching PubMed"):
            cached = self.cache.get(year)
            if cached is not None:
                results[year] = cached
                continue

            year_query = f"{query} AND ({year}/01/01[PDAT] : {year}/12/31[PDAT])"

            try:
                handle = Entrez.esearch(
                    db="pubmed", term=year_query, usehistory="y", retmax=0
                )
                record = Entrez.read(handle)
                handle.close()
            except Exception as e:
                print(f"Error searching year {year}: {e}")
                continue

            count = int(record["Count"])
            webenv = record["WebEnv"]
            query_key = record["QueryKey"]

            articles = []

            if count > 0:
                batch_size = 100
                for start in range(0, count, batch_size):
                    success = False
                    for attempt in range(3):
                        try:
                            fetch_handle = Entrez.efetch(
                                db="pubmed",
                                webenv=webenv,
                                query_key=query_key,
                                retstart=start,
                                retmax=batch_size,
                                rettype="xml",
                                retmode="xml",
                            )
                            records = Entrez.read(fetch_handle)
                            fetch_handle.close()

                            for article in records["PubmedArticle"]:
                                parsed = self._parse_article_xml(article, year)
                                articles.append(parsed)
                            success = True
                            break
                        except Exception:
                            time.sleep(2 * (attempt + 1))

                    if not success:
                        print(f"FAILED batch start {start} for year {year}")

                    time.sleep(0.34 if getattr(Entrez, "api_key", None) else 1.0)

            year_data = {"count": count, "articles": articles, "checked": True}
            results[year] = year_data
            self.cache.set(year, year_data)

        return results

    def _parse_article_xml(self, article, year):
        """Extract standardized fields from a PubMed XML article."""
        medline = article["MedlineCitation"]
        article_data = medline["Article"]

        pmid = str(medline.get("PMID", ""))
        title = article_data.get("ArticleTitle", "")

        doi = ""
        if "ELocationID" in article_data:
            for eloc in article_data["ELocationID"]:
                if eloc.attributes.get("EIdType") == "doi":
                    doi = str(eloc)

        if (
            not doi
            and "PubmedData" in article
            and "ArticleIdList" in article["PubmedData"]
        ):
            for item in article["PubmedData"]["ArticleIdList"]:
                if item.attributes.get("IdType") == "doi":
                    doi = str(item)

        author_list = []
        if "AuthorList" in article_data:
            for author in article_data["AuthorList"]:
                last = author.get("LastName", "")
                fore = author.get("ForeName", "")
                author_list.append(f"{last} {fore}".strip())

        journal_info = article_data.get("Journal", {})
        venue = journal_info.get("Title", "Unknown Journal")

        pub_date = str(year)
        if "JournalIssue" in journal_info and "PubDate" in journal_info["JournalIssue"]:
            pd_data = journal_info["JournalIssue"]["PubDate"]
            pub_date = pd_data.get("Year", str(year))

        abstract = ""
        if "Abstract" in article_data and "AbstractText" in article_data["Abstract"]:
            abst_source = article_data["Abstract"]["AbstractText"]
            if isinstance(abst_source, list):
                abstract = " ".join([str(x) for x in abst_source])
            else:
                abstract = str(abst_source)

        return {
            "id": pmid,
            "title": title,
            "authors": ", ".join(author_list),
            "venue": venue,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "doi": doi,
            "pub_date": pub_date,
            "citations": 0,
            "keywords": "",
        }


def main():
    parser = create_base_argparser("PubMed publication analysis tool")
    args = parser.parse_args()

    config_path = args.config or default_search_config_path()
    config = load_search_terms(config_path)

    analyzer = PubMedAnalyzer(email=config.get("email", ""), output_dir=args.output)

    def format_terms(term_list):
        if not term_list:
            return ""
        quoted = [f'"{t}"[Title/Abstract]' for t in term_list]
        return " OR ".join(quoted)

    g_brain = format_terms(config.get("brain_terms", []))
    g_diag = format_terms(config.get("diagnostic_terms", []))
    g_img = format_terms(config.get("imaging_terms", []))
    g_ai = format_terms(config.get("ai_terms", []))
    g_disease = format_terms(config.get("disease_terms", []))

    query_parts = [f"({g})" for g in [g_brain, g_diag, g_img, g_ai, g_disease] if g]
    query = " AND ".join(query_parts)

    run_academic_search(analyzer, args, config, query)


if __name__ == "__main__":
    main()
