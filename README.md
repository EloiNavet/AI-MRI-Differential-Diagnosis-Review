# AI-Based Differential Diagnosis of Neurodegenerative Diseases Using Structural MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-B31B1B.svg)]()

This repository contains the source code used to conduct the systematic literature review for the article entitled **"AI-Based Differential Diagnosis of Neurodegenerative Diseases Using Structural MRI: A Systematic Review"**. It provides the scripts necessary to reproduce our automated data collection (querying academic databases), metadata mapping, and the generation of all statistical analyses and figures presented in the manuscript.

## Structure

- **`src/fetchers/`**: API connectors (PubMed, arXiv, OpenAlex, Google Scholar, Scopus) to query and cache metadata using a standardized YAML terminology configuration (`search_config.yaml`).
- **`src/review_analysis/`**: Modules for standardizing extracted terminology (`normalization.yaml`), performing statistical analyses on the selected studies, generating LaTeX tables, and producing standardized figures.
- **`data/`**: Expected directory for PDF extraction outputs, merged CSV datasets, and generated figures.

*Note: The title/abstract screening and full-text eligibility assessments (PRISMA flow) were conducted manually by independent expert reviewers. Therefore, this repository focuses solely on the reproducible retrieval scripts and the downstream statistical analysis of the final extracted data.*

## Extracted data

In this systematic review, we rigorously extracted methodological and clinical information from each included paper. The primary extracted fields (which are subsequently plotted or analyzed by our scripts) include:

- **Study Metadata**: Title, authors, venue, and year of publication.
- **Clinical Context**: Target neurodegenerative diseases (e.g., AD, FTD, PD, MSA, PSP) and detailed sample size repartition per class.
- **Data Modalities & Sources**: Imaging modalities (T1w, T2w, DTI, PET, etc.), non-imaging data (CSF, clinical scores), and specific datasets used (e.g., ADNI, NIFD, PPMI, or in-house).
- **AI Methodology**: The general paradigm (Machine Learning, Deep Learning, or Hybrid) and specific architectures (e.g., SVM, CNN, Vision Transformers).
- **Interpretability & Explainability**: Methods used to interpret predictions (e.g., SHAP, Grad-CAM, Statistical maps).
- **Validation Strategy & Performance**: Explicit tracking of in-domain vs. out-of-domain (OOD) testing, and normalized performance metrics (Accuracy, BACC, AUC, Sensitivity, Specificity, etc.).
- **PROBAST Risk of Bias**: A systematic assessment of bias risk across 4 domains (Participant Selection, Predictors, Outcome, Statistical Analysis).

The `review_analysis.py` script aggregates this extracted data to automatically generate the statistical plots (bar charts, radar charts, violin plots, and network graphs) presented in the manuscript.

## Setup & Installation

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### API Keys
Some fetchers require API keys. Copy the example `.env` file and populate it with your credentials:
```bash
cp .env.example .env
```
*(Note: Google Scholar uses a ScraperAPI proxy by default to prevent throttling; it falls back to free proxies if the key is omitted or quota is exhausted.)*

## Reproducing the analyses

### 1. Fetching Data
Fetchers read from `src/search_config.yaml` by default. They cache their results per year to allow safe interruption and resumption.

```bash
python -m src.fetchers.pubmed_fetcher --output data/
python -m src.fetchers.arxiv_fetcher --output data/
python -m src.fetchers.openalex_fetcher --output data/
python -m src.fetchers.google_scholar_fetcher --output data/
python -m src.fetchers.scopus_fetcher --output data/
```

### 2. Review Analysis & Figure Generation
After completing the PRISMA screening and manually extracting tabular data from the final selected PDFs into a CSV (e.g., `data/review_output.csv`), you can generate the statistical review plots. This script filters the data and generates all the figures used in the manuscript:

```bash
python -m src.review_analysis.review_analysis data/review_output.csv data/figures/ --format pdf
```

### 3. Bibliography Mapping
Before generating the final LaTeX table for the supplementary materials, you must cross-reference the extracted papers with their BibLaTeX keys. This requires a `.biblatex` file containing all the references (typically exported from reference managers like Zotero).

```bash
python -m src.review_analysis.csv_biblatex_mapper data/review_output.csv --biblatex data/Source.biblatex
```
This step produces an enriched CSV (`data/review_output_citations.csv`) containing the necessary citation keys.

### 4. LaTeX Table Generation
Finally, convert the enriched CSV into a formatted LaTeX table suitable for publication. This step requires the citations mapped in the previous step to correctly compile the PDF bibliography.

```bash
python -m src.review_analysis.csv_to_latex data/review_output_citations.csv \
    --columns "Title, Year, Authors, Bib citations, Repartition, Modalities, Datasets, Neuropathological, OOD, Architecture(s) Used, Code, GPUs, PROBAST" \
    --merge "Title|Authors|Year|Bib citations" "Neuropathological|OOD" "Code|GPUs"
```

## Citation

*... to be defined ...*

> **AI-Based Differential Diagnosis of Neurodegenerative Diseases Using Structural MRI: A Systematic Review**  
> *Éloi Navet, Rémi Giraud, Boris Mansencal, Andrew Zamai, Vincent Planche, Pierrick Coupé*  

## License

This project is licensed under the MIT License - see the LICENSE file for details.
