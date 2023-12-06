import logging
import re
from dataclasses import dataclass
from logging import getLogger
from typing import List

# import sys

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(handler)
# handler.setLevel(logging.INFO)

GITHUB_URL_PATTERN = r"https?://github\.com/[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+"
GENERAL_URL_PATTERN = r"https?://[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9\-_]+)*"


@dataclass
class ArxivPaper:
    pid: str
    title: str
    authors: List[str]
    abstract: str
    link: str

    # 'forward reference' : -> "ArxivPaper"
    @classmethod
    def from_query(cls, entry: dict) -> "ArxivPaper":
        """Creates an ArxivPaper object from given entry(s) returned by arxiv api query."""
        pid = entry.entry_id.split("/")[-1]
        title = entry.title
        authors = entry.Author
        abstract = entry.summary
        link = entry.pdf_url
        # print(f"entry.pdf_url: {entry.pdf_url}")
        return cls(
            pid,
            title,
            authors,
            abstract,
            link,
        )

    def extract_github_links(self) -> List[str]:
        """
        Extract GitHub URLs from the paper's abstract.
        """
        github_urls = []
        # for match in re.findall(GITHUB_URL_PATTERN, self.abstract):
        #     github_urls.append(match)
        # other_urls = [
        #     re.findall(GENERAL_URL_PATTERN, self.abstract) if not github_urls else None
        # ]
        github_urls = re.findall(GITHUB_URL_PATTERN, self.abstract)
        all_urls = re.findall(GENERAL_URL_PATTERN, self.abstract)
        # other_urls = [url for url in all_urls if url not in github_urls]

        if github_urls or all_urls:
            logger.info(
                f"Paper '{self.title.strip()}':\n"
                f"GitHub URL(s): {', '.join(github_urls)}\n"
                f"Other URL(s): {', '.join(all_urls)}"
            )
        else:
            logger.info(
                f"No relevant URLs found in the following papers abstract: {self.title.strip()}"
            )
        x = list(set(github_urls + all_urls))
        x = [url for url in x if url]
        return x
