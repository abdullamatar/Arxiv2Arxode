import asyncio
from typing import Dict, List, Optional, Union

from embeddings import get_db_connection, get_embedding_func
import nest_asyncio
from autogen import AssistantAgent
from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent

from utils.utils import create_llm_config

# U N D E R  C O N S T R U C T I O N


class CRetrieveUserProx(RetrieveUserProxyAgent):
    def query_vector_db(
        self,
        query_texts: List[str],
        n_results: int = 10,
        search_string: str = "",
        **kwargs,
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        pass

    def retrieve_docs(
        self, problem: str, n_results: int = 20, search_string: str = "", **kwargs
    ):
        results = self.query_vector_db(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
            **kwargs,
        )

        # TODO: The northern winds blow strong...
        dbconn = get_db_connection(cname="init_vecdb", efunc=get_embedding_func())
        results = dbconn.similarity_search_with_relevance_scores(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
            **kwargs,
        )
        self._results = results  # Why?
        return results
        # print("doc_ids: ", results["ids"])


class CUserProxyAgent(UserProxyAgent):
    async def a_get_human_input(self, prompt: str) -> str:
        return await non_existent_async_func()

    async def a_receive(
        self,
        message: Dict | str,
        sender: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        return await super().a_receive(message, sender, request_reply, silent)


class CAssAgent(AssistantAgent):
    async def a_get_human_input(self, prompt: str) -> str:
        user_input = await non_existent_async_func()
        return str(user_input) + "hamburger"

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Call the superclass method to handle message reception asynchronously
        await super().a_receive(message, sender, request_reply, silent)


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
