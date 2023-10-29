from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class ArxivPaper:
    pid: str
    title: str
    authors: List[str]
    abstract: str
    link: str

    def __next__(self):
        return self

    def __iter__(self):
        return self

    # 'forward reference' : -> "ArxivPaper"
    @classmethod
    def from_query(cls, entry: dict) -> "ArxivPaper":
        """Creates an ArxivPaper object from given entry(s)."""
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

    def get_metadata(self) -> Dict[str, Optional[str]]:
        """
        THIS DOES NOT NEED TO EXIST :D
        Return the metadata of the paper.
        """
        return {
            "pid": self.pid,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
        }

    def has_github_link(self) -> bool:
        """
        Simple check if the paper mentions a GitHub repo in abstract
        """
        github_terms = ["github.com", "github.io"]
        return any(term in self.abstract.lower() for term in github_terms)
