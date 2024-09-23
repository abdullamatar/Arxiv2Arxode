from pathlib import Path
from typing import List, Optional

import arxiv
import utils.arxiv.arxiv_paper as axp

# from feedparser.mixin import _FeedParserMixin

# import urllib.request, urllib.parse, urllib.error
# import feedparser
# from arxiv import SortCriterion

"""
query all:electron. This url calls the api, which returns the results in the Atom 1.0 format.
Atom 1.0 is an xml-based format that is commonly used in website syndication feeds.
It is lightweight, and human readable, and results can be cleanly read in many web browsers.
"""

# !taken from: https://github.com/zonca/python-parse-arxiv/blob/master/python_arXiv_parsing_example.py
# Opensearch metadata such as totalResults, startIndex,
# and itemsPerPage live in the opensearch namespase.
# Some entry metadata lives in the arXiv namespace.
# This is a hack to expose both of these namespaces in
# feedparser v4.1
# _FeedParserMixin.namespaces["http://a9.com/-/spec/opensearch/1.1/"] = "opensearch"
# _FeedParserMixin.namespaces["http://arxiv.org/schemas/atom"] = "arxiv"


class ArxivScraper:
    def __init__(self):
        # ???? u s e l e s s ????
        self.client = arxiv.Client()

    def search_papers(
        self,
        search_query: str,
        max_results: int = 5,
        sort_by: Optional[str] = arxiv.SortCriterion.Relevance,
    ) -> List[axp.ArxivPaper]:
        """
        Search for papers on arXiv based on the given query, returns a list of axp.ArxivPaper objects.
        --------------------
        search_query: The search query string.
        start: The starting index for the results.
        max_results: The maximum number of results to retrieve.
        sort_by: The sorting criterion to use, defaults to relevance.
        """
        if sort_by == "relevance":
            sort_by = arxiv.SortCriterion.Relevance
        elif sort_by == "lastUpdatedDate":
            sort_by = arxiv.SortCriterion.LastUpdatedDate
        elif sort_by == "submittedDate":
            sort_by = arxiv.SortCriterion.SubmittedDate

        search = arxiv.Search(
            query=search_query, max_results=max_results, sort_by=sort_by
        )
        return [
            axp.ArxivPaper.from_query(entry) for entry in self.client.results(search)
        ]

    def download_papers(
        self,
        papers: List[axp.ArxivPaper] = [],
        ids: List[str] = [],
        fname_template: str = "{title}.pdf",
        *,
        dirpath: Path,
    ) -> None:
        """Downloads all papers returned by the search query into a given directory."""

        dirpath = Path(dirpath)
        if not dirpath.exists():
            dirpath.mkdir()

        if ids:
            paper = next(arxiv.Client().results(arxiv.Search(id_list=ids)))
            paper.download_pdf(
                dirpath=dirpath,
                filename=fname_template.format(title=paper.title.replace(" ", "_")),
            )
        else:
            for paper in papers:
                piter = arxiv.Search(id_list=[paper.pid]).results()
                next(piter).download_pdf(
                    dirpath=dirpath,
                    filename=fname_template.format(title=paper.title.replace(" ", "_")),
                )
