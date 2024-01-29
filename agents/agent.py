import asyncio

# import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import nest_asyncio
from autogen import AssistantAgent
from autogen.agentchat.agent import Agent

# from autogen.agentchat.agent import Agent
# from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent

import agents.agent_conf as agent_conf
from lib.embeddings import get_db_connection, get_embedding_func

# U N D E R  C O N S T R U C T I O N
# ◉_◉
# self.register_reply(Agent, RetrieveUserProxyAgent._generate_retrieve_user_reply, position=2)
# This agent by default must be triggered by another agent ^, from src


class EmbeddingRetrieverAgent(RetrieveUserProxyAgent):
    """Custom retriever agent that uses an embeddings database to retrieve relevant docs."""

    def __init__(
        self,
        collection_name: str,
        name: str = "RetrieveChatAgent",
        human_input_mode: Optional[str] = "ALWAYS",
        is_termination_msg: Optional[Callable[[Dict[str, Any]], bool]] = None,
        retrieve_config: Optional[
            Dict[str, str]
        ] = None,  # config for the retrieve agent
        **kwargs,
    ):
        # TODO: cname as param to __init__ (datastore_name?), ef as well?
        self.embedding_function = get_embedding_func()
        self.dbconn = get_db_connection(
            cname=collection_name if collection_name else "init_vecdb",
            efunc=self.embedding_function,
        )
        super().__init__(
            name=name,
            human_input_mode=human_input_mode,
            is_termination_msg=is_termination_msg,
            retrieve_config=retrieve_config,
            **kwargs,
        )

    def query_vector_db(
        self,
        query_texts: List[str],
        n_results: int = 10,
        search_string: str = None,
        **kwargs,
    ) -> Dict[str, List[List[str]]]:
        # ef = get_embedding_func()
        # embed_response = self.embedding_function.embed_query(query_texts)
        # print(embed_response)
        relevant_docs = self.dbconn.similarity_search_with_relevance_scores(
            query=query_texts,
            k=n_results,
        )

        # TODO: get actual id from langchain
        # They need the docs as a list of lists...
        sim_score = [relevant_docs[i][1] for i in range(len(relevant_docs))]
        return {
            "ids": [[i for i in range(len(relevant_docs))]],
            "documents": [[doc[0].page_content] for doc in relevant_docs],
            "metadatas": [
                {**doc[0].metadata, "similarity_score": score}
                for doc, score in zip(relevant_docs, sim_score)
            ],
        }

    def retrieve_docs(
        self, problem: str, n_results: int = 4, search_string: str = "", **kwargs
    ):
        """
        Args:
            problem: the problem to be solved.
            n_results: the number of results to be retrieved. Default is 20.
            search_string: only docs that contain an exact match of this string will be retrieved. Default is "".
        """
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
        # return results

    def get_content(self):
        return self._doc_contents

    def generate_init_message(
        self, problem: str, n_results: int = 20, search_string: str = ""
    ):
        return super().generate_init_message(problem, n_results, search_string)

    def generate_reply(
        self,
        messages: List[Dict[str, str]] | None = None,
        sender: Agent | None = None,
        exclude: List[Callable[..., Any]] | None = None,
    ) -> str | Dict | None:
        return super().generate_reply(messages, sender, exclude)

    def _generate_retrieve_user_reply(
        self,
        messages: List[Dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> Tuple[bool, str | Dict | None]:
        return super()._generate_retrieve_user_reply(messages, sender, config)

    # To modify the way to execute code blocks, single code block, or function call, override `execute_code_blocks`,
    # `run_code`, and `execute_function` methods respectively.

    # async def a_receive(
    #     self,
    #     message: Dict | str,
    #     sender: Agent,
    #     request_reply: bool | None = None,
    #     silent: bool | None = False,
    # ):
    #     logging.info(f"EmbeddingRetrieverAgent received message from {sender.name}")
    #     if sender.name == "coordinator" and "Retrieve relevant documents" in message:
    #         problem = message.replace("Retrieve relevant documents for: ", "")
    #         logging.info(f"Retrieving documents for problem: {problem}")
    #         retrieved_content = await self.retrieve_docs(problem)  # Ensure async call
    #         logging.info(f"Retrieved content: {retrieved_content}")
    #     return super().a_receive(message, sender, request_reply, silent)


async def non_existent_async_func():
    await asyncio.sleep(4)


async def main():
    main = UserProxyAgent(
        name="main",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    assistant = AssistantAgent(
        name="assistant",
        #! system message, fixed typo: https://github.com/microsoft/autogen/blob/main/notebook/Async_human_input.ipynb
        system_message="Under construction.",
        llm_config=agent_conf.base_cfg,
    )

    await main.a_initiate_chat(
        assistant,
        message="Under construction.",
        n_results=3,
    )


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
