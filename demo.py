#!/usr/bin/env python
# coding: utf-8

# # Computer Science Project Demonstration:
# -----
import lib.embeddings as emb
import utils.arxiv.arxiv_search as axs

arx_srch = axs.ArxivScraper()
papers = arx_srch.search_papers("Graph Neural Networks")


print(
    f"type of papers: {type(papers[0])}\navailable attributes for scraper: {dir(arx_srch)[-2:]}"
)
print(f"len of paper: {len(papers)}")
papers[0]
# Downloaded just 1 paper, feel free to change...
arx_srch.download_papers(papers=papers[:1], dirpath="./temp_papers")


links_and_titles = {p.title: p.link for p in papers}
print(links_and_titles)


chunked_documents = emb.load_and_chunk_papers(
    pdf_path="./temp_papers/",
)
print(len(chunked_documents))
chunked_documents[:10]

collection_name = "cs_demo"
pgvec_connection = emb.create_embedding_collection(
    chunked_docs=chunked_documents,
    embeddings=emb.get_embedding_func(),
    collection_name=collection_name,
)

pgvec_connection.similarity_search("Graph attention network GAT", k=4)


# ##### The coordinator is the main orchestration class responsible for running and monitoring the code generation groupchat.
# ---
# - Some main functions to highlight:
#     - `marl` initializes four agents in this case each with a unique role.

from agents.agent import marl
from agents.coordinator import Coordinator

crd = Coordinator(
    team_name="test",
    agents=marl(collection_name=collection_name),
)

crd.code_gen_group_chat(
    "Create a simplified example of graph attention networks in python to help me grasp the context.",
    fname="outfile.jsonl",
    task_idx=1_000_000,  # this is a dummy id that will be used as this is currently run over an entire dataset taking the ids from there...
)
