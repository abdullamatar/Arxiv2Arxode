from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ArxivPaper:
    id: str
    title: str
    published: str
    authors: List[str]
    abstract: str
    links: List[dict]
    journal_ref: Optional[str]
    comment: Optional[str]

    @classmethod
    # 'forward reference' : -> "ArxivPaper"
    def from_entry(cls, entry: dict) -> "ArxivPaper":
        paper_id = entry.id.split("/abs/")[-1]
        title = entry.title
        published = entry.published
        authors = [author.name for author in entry.contributors]
        abstract = entry.summary
        links = entry.links
        journal_ref = getattr(entry, "arxiv_journal_ref", None)
        comment = getattr(entry, "arxiv_comment", None)

        return cls(
            paper_id, title, published, authors, abstract, links, journal_ref, comment
        )

    def get_metadata(self) -> Dict[str, Optional[str]]:
        """
        Return the metadata of the paper.

        :return: Dictionary containing paper metadata.
        """
        return {
            "id": self.id,
            "title": self.title,
            "published": self.published,
            "authors": self.authors,
            "abstract": self.abstract,
            "journal_ref": self.journal_ref,
            "comment": self.comment,
        }

    def download(self) -> None:
        """
        Download the PDF of the paper.
        """
        pdf_link: Optional[str] = next(
            (link["href"] for link in self.links if link["title"] == "pdf"), None
        )
        if pdf_link:
            pass

    def has_github_link(self) -> bool:
        """
        Check if the paper mentions a GitHub repo.

        :return: True if a GitHub link is found, otherwise False.
        """
        github_terms: List[str] = ["github.com", "github.io"]
        return any(term in self.abstract.lower() for term in github_terms)
