# TODO: Py03 rs bindings for file manip?
# import pypdf
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.document_loaders.python import PythonLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

load_dotenv()
openaikey = os.environ.get("OPENAI_APIKEY")

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGV_HOST", ""),
    port=int(os.environ.get("PGV_PORT", "")),
    database=os.environ.get("PGV_DATABASE", ""),
    user=os.environ.get("PGV_USER", ""),
    password=os.environ.get("PGV_PASSWORD", ""),
)


def load_and_chunk_papers(
    pdf_path: Optional[str] = "./",
    file_name_glob: Optional[str] = "*.pdf",
    single_file: Optional[bool] = False,
) -> List[Document]:
    """
    Get the batched text from the pdf. Optionally provide glob style regex to filter files.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    if single_file:
        loader = PyPDFLoader(pdf_path)
    else:
        loader = DirectoryLoader(pdf_path, glob=file_name_glob)
    return loader.load_and_split(text_splitter)


def get_embedding_func(model: str = "text-embedding-ada-002") -> Embeddings:
    """
    Select how to create embeddings for the pdf, i.e. Word2Vec, GloVe, sentencepiece, openAI, llama, mistral...

    EVERYTHING IS UNDER CONSTRUCTION :D
    """
    return OpenAIEmbeddings(openai_api_key=openaikey, model=model)


def get_db_connection(cname: str, efunc: Embeddings | None = None) -> VectorStore:
    """
    Get the database connection, given collection name.
    """
    assert PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=cname,
        embedding_function=efunc if efunc else get_embedding_func(),
    ).similarity_search("test"), "Invalid collection name or connection string."

    return PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=cname,
        embedding_function=efunc if efunc else get_embedding_func(),
    )


def create_embedding_collection(
    chunked_docs: List[Document],
    embeddings: Embeddings,
    collection_name: str,
    db_dir: Optional[str] | None = None,
) -> VectorStore:
    """
    Initialize vector database, optionally provide embeddings, db persistance directory, and collection name.
    """
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


def load_and_chunk_code(
    code_path: str,
) -> List[Document]:
    """Create code embeddings given a path to a directory of python files."""
    # text-embedding-ada-002 embedding model used by default, which is also the best openai model for code embeddings

    code_path = os.path.abspath(code_path)

    loader = DirectoryLoader(
        code_path, glob="**/*.py", loader_cls=PythonLoader, show_progress=True
    )

    # print(len(python_documents), type(python_documents))
    # print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=4000, chunk_overlap=0
    )
    # python_docs = python_splitter.split_documents(python_documents)
    return loader.load_and_split(python_splitter)


if __name__ == "__main__":
    # docs = load_and_chunk_papers("./temp")
    # vdb = create_embedding_collection(docs, get_embedding_func(docs))
    # print(load_and_chunk_code("./temprepo"))
    print(CONNECTION_STRING)
    # print(type(load_and_chunk_code("./temprepo")))
