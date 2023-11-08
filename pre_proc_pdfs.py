# import pypdf
from typing import Optional, List
import os
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores.pgvector import PGVector
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from dotenv import load_dotenv

load_dotenv()
openaikey = os.environ.get("OPENAI_APIKEY")


def load_and_chunk_papers(
    pdf_path: str, file_name_glob: Optional[str] = "*.pdf"
) -> List[Document]:
    """
    Get the chunked text from the pdf. Optionally provide glob style regex to filter files.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    print("path %s" % pdf_path)
    loader = DirectoryLoader(pdf_path, glob=file_name_glob)
    return loader.load_and_split(text_splitter)


def create_word_embeddings(docs: List[Document] = None) -> Embeddings:
    """
    Select how to create embeddings for the pdf, i.e. Word2Vec, GloVe, sentencepiece, openAI Embeddings.
    EVERYTHING IS UNDER CONSTRUCTION :D
    """
    return OpenAIEmbeddings(openai_api_key=openaikey)


def init_vectorDB(
    chunked_docs: List[Document],
    embeddings: Optional[Embeddings] = None,
    db_dir: Optional[str] = "./dbstore",
    collection_name: Optional[str] = "init_vecdb",
) -> VectorStore:
    """
    Initialize vector database, optionally provide embeddings, db persistance directory, and collection name.
    """
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
        host=os.environ.get("PGV_HOST"),
        port=int(os.environ.get("PGV_PORT")),
        database=os.environ.get("PGV_DATABASE"),
        user=os.environ.get("PGV_USER"),
        password=os.environ.get("PGV_PASSWORD"),
    )

    vdb = PGVector.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        # persist_directory=db_dir,
        # client=chrdb_connection,
        # collection_name=collection_name,
    )
    return vdb


if __name__ == "__main__":
    docs = load_and_chunk_papers("./temp")
    vdb = init_vectorDB(docs, create_word_embeddings(docs))
