################################################
################################################
################################################
######### PLEASE IGNORE THIS FILE :D ###########
################################################
################################################
################################################

# import openai
import logging
import os

from langchain.chat_models import ChatOpenAI
# TODO: https://microsoft.github.io/autogen/blog/2023/11/20/AgentEval
# from langchain.prompts import FewShotChatMessagePromptTemplate, PipelinePromptTemplate
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import MultiQueryRetriever

from lib.embeddings import (create_embedding_collection, get_db_connection,
                            get_embedding_func, load_and_chunk_code)
from utils.misc import clone_and_clean_repo, ftos

# from langchain.schema.vectorstore import VectorStore
# from langchain.schema.document import Document


# from langchain.schema import AIMessage, HumanMessage, SystemMessage

#! ExampleSelector to do a simserach between input and fewshot examples

# import inspect


non_profit_ai_company = os.environ.get("OPENAI_APIKEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  👁‍🗨  %(levelname)s  👁‍🗨  :\n%(message)s",
)

llm = ChatOpenAI(n=4, openai_api_key=non_profit_ai_company, model="gpt-4-1106-preview")
OUTFILE_NAME = "output.py"


def m():
    from utils.arxiv import ArxivScraper

    ax = ArxivScraper()
    papers = ax.search_papers("Agent tuning generalized agents for llms")
    return [url for paper in papers for url in paper.extract_github_links()]  # ...xD


def r():
    urls = m()
    clone_and_clean_repo(urls[0], "./temprepo")


def mqr(query: str):
    """Multi-query retrieval"""
    vector_store = get_db_connection(cname="init_vecdb")
    # TODO: Consider
    retriever_model = MultiQueryRetriever.from_llm(
        llm=llm, retriever=vector_store.as_retriever()
    )
    return retriever_model.get_relevant_documents(query=query)


# !apparently autogen==0.2.0b5 is needed to use oai assitant api


# Prompt types: task init prompt | codegen prompt | summary prompt | reflection prompt
# Nothing interesting here yet...
def main():
    logging.info(f"Your cwd is : {os.getcwd()}")
    logging.info(f"Turbo encabulator initialized...")

    # prompts
    initial_system_prompt_tmpl = ftos("./prompts/init_agent.txt")
    dynamic_ic_prompt_tmpl = ftos("./prompts/curr_agent_prompt.txt")
    rflx_prompt_tmpl = ftos("./prompts/reflection.txt")
    releveant_code_tmpl = ftos("./prompts/retrieved_code.txt")

    system_init_prompt = PromptTemplate.from_template(initial_system_prompt_tmpl)
    dynamic_ic_prompt = PromptTemplate.from_template(dynamic_ic_prompt_tmpl)
    rflx_prompt = PromptTemplate.from_template(rflx_prompt_tmpl)

    init_prompt = system_init_prompt
    # print(init_prompt.input_variables)

    init_prompt = init_prompt.format_prompt(init_task_description="chicken parm")

    # dynamic_ic_prompt = dynamic_ic_prompt.format_prompt(
    #     curr_task_description="",
    #     task_state="*",
    #     working_memory="*",
    # )

    logging.info(f"init_prompt: {init_prompt}")

    # print(mqr(query="github link"))
    # You want to pass in any relevant code context.

    success_count = []
    execution_count = []

    # print(dir(vector_store))

    # print(initial_system_prompt)


if __name__ == "__main__":
    m()
    # r()
    print(478_000_432_898_444)
    # main()
