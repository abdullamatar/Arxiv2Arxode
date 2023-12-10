import asyncio
from typing import Any, Dict, List, Optional, Union, Callable

from embeddings import get_db_connection, get_embedding_func
import nest_asyncio
from autogen import AssistantAgent

# from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent

from utils.misc import create_llm_config

# U N D E R  C O N S T R U C T I O N

# This feels cursed and is a temp sol, I am missing something with the __init__ func and tiktoken embeddings?
# dbconn = get_db_connection(cname="init_vecdb", efunc=get_embedding_func())


class EmbeddingRetrieverAgent(RetrieveUserProxyAgent):
    def __init__(
        self,
        name="RetrieveChatAgent",  # default set to RetrieveChatAgent
        human_input_mode: Optional[str] = "ALWAYS",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        retrieve_config: Optional[Dict] = None,  # config for the retrieve agent
        **kwargs,
    ):
        self.embedding_function = get_embedding_func()
        self.dbconn = get_db_connection(
            cname="init_vecdb", efunc=self.embedding_function
        )
        super().__init__()

    def query_vector_db(
        self,
        query_texts: List[str],
        n_results: int = 10,
        search_string: str = "",
        **kwargs,
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        # ef = get_embedding_func()
        # embed_response = self.embedding_function.embed_query(query_texts)
        # print(embed_response)
        relevant_docs = self.dbconn.similarity_search_with_relevance_scores(
            query=query_texts,
            k=n_results,
        )

        # TODO: get actual id from langchain
        return {
            "ids": [i for i in range(len(relevant_docs))],
            "documents": [doc[0].page_content for doc in relevant_docs],
            "metadatas": [doc[0].metadata for doc in relevant_docs],
        }

    def retrieve_docs(
        self, problem: str, n_results: int = 4, search_string: str = "", **kwargs
    ):
        results = self.query_vector_db(
            query_texts=problem,
            n_results=n_results,
            search_string=search_string,
            # embedding_function=get_embedding_func(),
            # embedding_model="text-embedding-ada-002",
            **kwargs,
        )
        # print(results)
        # # TODO: The northern winds blow strong...
        self._results = results  # Why?: It is a class property; state repr i guess?


class CUserProxyAgent(UserProxyAgent):
    # async def a_get_human_input(self, prompt: str) -> str:
    #     pass
    # return await non_existent_async_func()

    # async def a_receive(
    #     self,
    #     message: Dict | str,
    #     sender: Agent,
    #     request_reply: bool | None = None,
    #     silent: bool | None = False,
    # ):
    #     pass
    # return await super().a_receive(message, sender, request_reply, silent)
    pass


class CAssAgent(AssistantAgent):
    # async def a_get_human_input(self, prompt: str) -> str:
    #     user_input = await non_existent_async_func()
    #     return str(user_input) + "hamburger"

    # async def a_receive(
    #     self,
    #     message: Union[Dict, str],
    #     sender,
    #     request_reply: Optional[bool] = None,
    #     silent: Optional[bool] = False,
    # ):
    #     # Call the superclass method to handle message reception asynchronously
    #     await super().a_receive(message, sender, request_reply, silent)
    pass


async def non_existent_async_func():
    await asyncio.sleep(4)


async def main():
    boss = CUserProxyAgent(
        name="boss",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    assistant = CAssAgent(
        name="assistant",
        #! system message, fixed typo: https://github.com/microsoft/autogen/blob/main/notebook/Async_human_input.ipynb
        system_message="Under construction.",
        llm_config=create_llm_config("gpt-4", "0.4", "22"),
    )

    await boss.a_initiate_chat(
        assistant,
        message="Under construction.",
        n_results=3,
    )


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
