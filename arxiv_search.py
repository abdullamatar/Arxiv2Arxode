# Taken from: https://github.com/zonca/python-parse-arxiv/blob/master/python_arXiv_parsing_example.py

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

"""
query all:electron. This url calls the api, which returns the results in the Atom 1.0 format.
Atom 1.0 is an xml-based format that is commonly used in website syndication feeds.
It is lightweight, and human readable, and results can be cleanly read in many web browsers.
"""
# Base api query url
base_url = "http://export.arxiv.org/api/query?"

search_query = "all:LLM+AND+abs:Agents"
start = 0  # retreive the first 5 results
max_results = 5

query = "search_query=%s&start=%i&max_results=%i" % (search_query, start, max_results)

# Opensearch metadata such as totalResults, startIndex,
# and itemsPerPage live in the opensearch namespase.
# Some entry metadata lives in the arXiv namespace.
# This is a hack to expose both of these namespaces in
# feedparser v4.1
_FeedParserMixin.namespaces["http://a9.com/-/spec/opensearch/1.1/"] = "opensearch"
_FeedParserMixin.namespaces["http://arxiv.org/schemas/atom"] = "arxiv"
# perform a GET request using the base_url and query
response = urllib.request.urlopen(base_url + query).read()


def writefile(dat):
    timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    with open(f"response-{timestamp}.xml", "wb") as f:
        f.write(dat)


# change author -> contributors (because contributors is a list)
response = response.replace(b"author", b"contributor")
writefile(response)

# parse the response using feedparser
feed = feedparser.parse(response)
print("*" * 40)
print(feed)
print("*" * 40)
print(feed.keys())
print(type(feed))
print(f"paper id: {feed.entries[0].id}")
# the id is of this form we want a regex to just math the last part, ie
# "2310.03903v1" http://arxiv.org/abs/2310.03903v1
print(len(feed.entries))
fst_paperid = feed.entries[0].id.split("/abs/")[-1]
# print(type(fst_paperid))
paper = next(arxiv.Search(id_list=[fst_paperid]).results())
paper.download_pdf()

# paper = next(arxiv.Search(query=search_query).results())
exit()

# feedparser.util.FeedParserDict

# print out feed information
print("Feed title: %s" % feed.feed.title)
print("Feed last updated: %s" % feed.feed.updated)

# print opensearch metadata
print("totalResults for this query: %s" % feed.feed.opensearch_totalresults)
print("itemsPerPage for this query: %s" % feed.feed.opensearch_itemsperpage)
print("startIndex for this query: %s" % feed.feed.opensearch_startindex)

# Run through each entry, and print out information
for entry in feed.entries:
    print("e-print metadata")
    print("arxiv-id: %s" % entry.id.split("/abs/")[-1])
    print("Published: %s" % entry.published)
    print("Title:  %s" % entry.title)

    print("Authors:  %s" % ",".join(author.name for author in entry.contributors))

    # get the links to the abs page and pdf for this e-print
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

    # Since the <arxiv:primary_category> element has no data, only
    # attributes, feedparser does not store anything inside
    # entry.arxiv_primary_category
    # This is a dirty hack to get the primary_category, just take the
    # first element in entry.tags.  If anyone knows a better way to do
    # this, please email the list!
    print("Primary Category: %s" % entry.tags[0]["term"])

    # Lets get all the categories
    all_categories = [t["term"] for t in entry.tags]
    print("All Categories: %s" % (", ").join(all_categories))

    # The abstract is in the <summary> element
    print("Abstract: %s" % entry.summary)
