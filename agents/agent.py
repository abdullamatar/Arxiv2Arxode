import asyncio
import logging
import random
# import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import nest_asyncio
from autogen import (AssistantAgent, ConversableAgent, GroupChatManager,
                     UserProxyAgent, config_list_from_json)
from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import \
    RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent

import agents.agent_conf as agent_conf
from agents.agent_conf import base_cfg
from lib.embeddings import get_db_connection, get_embedding_func

logger = logging.getLogger(__name__)

# U N D E R  C O N S T R U C T I O N
# ◉_◉
# self.register_reply(Agent, RetrieveUserProxyAgent._generate_retrieve_user_reply, position=2)
# This agent by default must be triggered by another agent ^, from src

# TODO: Dynamic agent creation, urls as rag src, md, readmes, notebooks...

termination_msg = (
    lambda x: isinstance(x, dict)
    and "TERMINATE" == str(x.get("content", ""))[-9:].upper()
)


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
        # self.customized_prompt
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
        sim_score = [relevant_docs[i][1] for i in range(len(relevant_docs))]
        return {
            "ids": [[i for i in range(len(relevant_docs))]],
            "documents": [[doc[0].page_content for doc in relevant_docs]],
            "metadatas": [
                {**doc[0].metadata, "similarity_score": score}
                for doc, score in zip(relevant_docs, sim_score)
            ],
        }  # type: ignore

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
        # TODO: The northern winds blow strong...
        self._results = results  # Why?: It is a class property; state repr i guess?
        # return results

    # def get_content(self):
    #     return self._doc_contents

    # def generate_init_message(
    #     self, problem: str, n_results: int = 20, search_string: str = ""
    # ):
    #     return super().generate_init_message(problem, n_results, search_string)

    # def generate_reply(
    #     self,
    #     messages: List[Dict[str, str]] | None = None,
    #     sender: Agent | None = None,
    #     exclude: List[Callable[..., Any]] | None = None,
    # ) -> str | Dict | None:
    #     return super().generate_reply(messages, sender, exclude)

    # def _generate_retrieve_user_reply(
    #     self,
    #     messages: List[Dict] | None = None,
    #     sender: Agent | None = None,
    #     config: Any | None = None,
    # ) -> Tuple[bool, str | Dict | None]:
    #     return super()._generate_retrieve_user_reply(messages, sender, config)

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


class CodingAssistant(UserProxyAgent):
    # https://github.com/microsoft/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb
    def __init__(
        self,
        name: str,
        system_message: str | None = None,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Callable[[Dict], bool] | None = None,
        max_consecutive_auto_reply: int | None = None,
        human_input_mode: str | None = "NEVER",
        code_execution_config: Dict | Literal[False] | None = False,
        description: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            description=description,
            system_message=system_message,
            # description,
            **kwargs,
        )

    def run_code(self, code, **kwargs):
        logger.info(f"RUNNING CODE:\n{code}")
        # TODO: More meaningful filename
        filename = f"code_gen_{random.randint(0, 1000)}"
        with open(f"sandbox/{filename}.py", "a") as f:
            f.write(code)
        return super().run_code(code, **kwargs)


class GCManager(GroupChatManager): ...


# To modify the way to execute code blocks, single code block, or function call, override execute_code_blocks, run_code, and execute_function methods respectively. (https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent)

"""
From: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb
Call RetrieveUserProxyAgent while init chat with another user proxy agent
Sometimes, there might be a need to use RetrieveUserProxyAgent in group chat without initializing the chat with it. In such scenarios, it becomes essential to create a function that wraps the RAG agents and allows them to be called from other agents. WHY?
"""

# TODO: https://microsoft.github.io/autogen/blog/2023/10/26/TeachableAgent

# rc: https://microsoft.github.io/autogen/blog/2023/10/18/RetrieveChat/


def marl() -> List[ConversableAgent]:
    # TODO: role def, assAgent for function/tool selection, user prox for eval() https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/#enhanced-inference
    agent0 = UserProxyAgent(
        name="main_userproxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        system_message="Conduct everything in a step by step manner, you are initiating the conversations, you are great at what you do. Please use the retrieve_content function graciously, the more information you have and gather the better.",
        description="Your role is to coordinate the completion of tasks related to generating code based off of machine learning and AI research. You must be diligent and operate in a step by step manner, make use of all the agents at your disposal.",
        llm_config=base_cfg,
    )

    # ret_conf = config_list_from_json(
    #     env_or_file="./agents/agent_conf.py",
    #     filter_dict={
    #         "model": {
    #             "gpt-3.5-turbo",
    #         }
    #     },
    # )
    retriever = EmbeddingRetrieverAgent(
        name="retrieval_agent",
        human_input_mode="NEVER",
        system_message="Retrieve additional information to complete the given task. Create a detailed query so that the retrieval is impactful in terms of information gained.",
        description="A retrieval augmented agent whose role is to retrieve additional information when asked, you can access an embeddings database with information related to code and research papers.",
        code_execution_config=False,
        collection_name="init_vecdb",
        llm_config=base_cfg,
        retrieve_config={
            "task": "qa",
            "client": "psycopg2",
        },
    )

    agent2 = AssistantAgent(
        name="code_reviewer",
        description="Agent used to review code, given the information retrieved by the retrieval agent and other information related to the main problem at hand. Review the code generated by the coding_agent to make sure it is executable and logically follows the ideas from the research and source code.",
        system_message="Review any generated code, add and modify it as needed. Also retrieve additional information from the retrieval agent, plan out what you want to do and why in a step by step manner.",
        llm_config=base_cfg,
        is_termination_msg=termination_msg,
        code_execution_config=False,
    )
    agent3 = CodingAssistant(
        name="coding_agent",
        system_message="Your role is to generate and execute code based off of the information provided by the retrieval agent and the code reviewer agent.",
        description="A coding agent that is tasked with iteratively generating code based off of the information provided by the retrieval agent and the code reviewer agent.",
        code_execution_config={"work_dir": "./sandbox", "use_docker": False},
        # function_map={
        #     "execute_and_save": execute_and_save,
        # },
        human_input_mode="NEVER",
        is_termination_msg=termination_msg,
        llm_config=base_cfg,
    )
    return [agent0, retriever, agent2, agent3]


if __name__ == "__main__":
    exit(9000)
    # nest_asyncio.apply()
    # asyncio.run(main())
