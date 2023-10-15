# adopted from: https://github.com/zonca/python-parse-arxiv/blob/master/python_arXiv_parsing_example.py

###############
## Resources ##
###############
# !scraping_arxiv ex: https://www.askpython.com/python/examples/scrape-arxiv-papers-python
# !arxiv api search_query docs: https://info.arxiv.org/help/api/user-manual.html#query_details
# !arxiv package docs: https://github.com/lukasschwab/arxiv.py#example-downloading-papers

"""
python_arXiv_parsing_example.py

This sample script illustrates a basic arXiv api call
followed by parsing of the results using the
feedparser python module.

Please see the documentation at
http://export.arxiv.org/api_help/docs/user-manual.html
for more information, or email the arXiv api
mailing list at arxiv-api@googlegroups.com.

urllib is included in the standard python library.
feedparser can be downloaded from http://feedparser.org/ .

Original Author: Julius B. Lucks
Original Contributor: Andrea Zonca (andreazonca.com)

This is free software.  Feel free to do what you want
with it, but please play nice with the arXiv API!
"""

import urllib.request, urllib.parse, urllib.error
import feedparser
from feedparser.mixin import _FeedParserMixin
from datetime import datetime
import arxiv
from typing import List

# from typing import List, Dict

"""
query all:electron. This url calls the api, which returns the results in the Atom 1.0 format.
Atom 1.0 is an xml-based format that is commonly used in website syndication feeds.
It is lightweight, and human readable, and results can be cleanly read in many web browsers.
"""


# Opensearch metadata such as totalResults, startIndex,
# and itemsPerPage live in the opensearch namespase.
# Some entry metadata lives in the arXiv namespace.
# This is a hack to expose both of these namespaces in
# feedparser v4.1
_FeedParserMixin.namespaces["http://a9.com/-/spec/opensearch/1.1/"] = "opensearch"
_FeedParserMixin.namespaces["http://arxiv.org/schemas/atom"] = "arxiv"
# perform a GET request using the base_url and query
# response = urllib.request.urlopen(base_url + query).read()

BASE_URL = "http://export.arxiv.org/api/query?"
SEARCH_QUERY = "all:LLM+AND+abs:Agents"
START = 0
MAX_RESULTS = 5


def construct_query_url(
    base_url: str, search_query: str, start: int, max_results: int
) -> str:
    # query = "search_query=%s&start=%i&max_results=%i" % (search_query, start, max_results)
    return (
        f"{base_url}search_query={search_query}&start={start}&max_results={max_results}"
    )


def fetch_data_from_api(url: str) -> bytes:
    response = urllib.request.urlopen(url).read()
    return response.replace(b"author", b"contributor")


def write_data_to_file(data: bytes):
    timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    with open(f"response-{timestamp}.xml", "wb") as f:
        f.write(data)


def parse_feed(data: bytes) -> feedparser.util.FeedParserDict:
    return feedparser.parse(data)


def print_feed_info(feed: feedparser.util.FeedParserDict):
    for entry in feed.entries:
        print("e-print metadata")
        print("arxiv-id: %s" % entry.id.split("/abs/")[-1])
        print("Published: %s" % entry.published)
        print("Title:  %s" % entry.title)
        print("Authors:  %s" % ",".join(author.name for author in entry.contributors))

    for link in entry.links:
        if link.rel == "alternate":
            print("abs page link: %s" % link.href)
        elif link.title == "pdf":
            print("pdf link: %s" % link.href)

    # The journal reference, comments and primary_category sections live under
    # the arxiv namespace
    try:
        journal_ref = entry.arxiv_journal_ref
    except AttributeError:
        journal_ref = "No journal ref found"
    print("Journal reference: %s" % journal_ref)

    try:
        comment = entry.arxiv_comment
    except AttributeError:
        comment = "No comment found"
    print("Comments: %s" % comment)


def download_paper(paper_ids: List[str]) -> None:
    for pid in paper_ids:
        paper = next(arxiv.Search(id_list=[pid]).results())
        paper.download_pdf()


if __name__ == "__main__":
    query_url = construct_query_url(BASE_URL, SEARCH_QUERY, START, MAX_RESULTS)
    data = fetch_data_from_api(query_url)
    write_data_to_file(data)
    feed = parse_feed(data)
    print_feed_info(feed)

# change author -> contributors (because contributors is a list)
# response = response.replace(b"author", b"contributor")
# writefile(response)
# parse the response using feedparser
# feed = feedparser.parse(response)
# print("*" * 40)
# print(feed)
# print("*" * 40)
# print(feed.keys())
# print(type(feed))
# print(f"paper id: {feed.entries[0].id}")
# "2310.03903v1" http://arxiv.org/abs/2310.03903v1
# print(len(feed.entries))
# fst_paperid = feed.entries[0].id.split("/abs/")[-1]
# print(type(fst_paperid))
# paper = next(arxiv.Search(id_list=[fst_paperid]).results())
# paper.download_pdf()
