from typing import Any, Dict, List

from autogen import (
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
)

from agents.agent_conf import base_cfg, exec_py_conf, write_file_config
from agents.agents import EmbeddingRetrieverAgent
from agents.functions import Functions

"""
From: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb
Call RetrieveUserProxyAgent while init chat with another user proxy agent
Sometimes, there might be a need to use RetrieveUserProxyAgent in group chat without initializing the chat with it. In such scenarios, it becomes essential to create a function that wraps the RAG agents and allows them to be called from other agents.
"""


# SEED = 22
PROBLEM = "I want to understand the agent tuning paper and come out with a minimal implementation of some of the core ideas in the paper the code must be executable."

# TODO: def write python function

termination_msg = (
    lambda x: isinstance(x, dict)
    and "TERMINATE" == str(x.get("content", ""))[-9:].upper()
)
# def is_termination_msg(termination_msg

#         return x.get("page_content", " ").rstrip().endswith("TERMINATE")
#     else:
#         return False

# lmconf = {
#     **lmconf,
#     # "cache_seed": SEED,
# }

# usrproxagent can exe code, feedback provider to other agents.
# To modify the way to execute code blocks, single code block, or function call, override execute_code_blocks, run_code, and execute_function methods respectively. (https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent)


def create_research_team() -> List[ConversableAgent]:
    agent0 = UserProxyAgent(
        name="main_userproxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        system_message="Your role is to coordinate the completion of tasks related to generating code based off of machine learning and AI research. You must be diligent and operate in a step by step manner to pinpoint potentially implementable parts of the research in accordance with the over all task at hand. With the goals of either simplifying the ideas in a paper for better understanding, merging ideas, improving upon or otherwise manipulating them where you see fit.",
    )

    retriever = EmbeddingRetrieverAgent(
        name="info_hoarder",
        human_input_mode="NEVER",
        system_message="You play a pivotal role in the progression of the task at hand as you have access to databases that store embeddings of research papers and their associated code if it exists. Your main job is to provide a detailed and step by step understanding of the relevant research paper(s) and pieces of code to the other agents. You should focus on the core ideas of the paper and the code base and how they relate to each other, with the end goal of helping in providing an implementation plan.",
        code_execution_config=False,
        llm_config=base_cfg,
        retrieve_config={
            "task": "qa",
        },
        # max_consecutive_auto_reply=4,
    )

    agent2 = AssistantAgent(
        name="code_designer",
        system_message="Your role is to design an interface of functions and/or classes that will be implemented based on the research paper(s) and code you are provided. You should focus on the implementation of the ideas in the research paper(s) and code base according to the task at hand, with the end goal of making it clear and simple for the coding agent to implement the interface. When you are done reply with 'TERMINATE'",
        llm_config=base_cfg,
        is_termination_msg=termination_msg,
        code_execution_config=False,
    )
    agent3 = AssistantAgent(
        name="coding_llm",
        system_message="Your role is to implement the interface designed by the code designer. You should focus on the implementation of the interface according to the task at hand, with the end goal of creating an executable, self contained python file.",
        # code_execution_config=[True],
        # function_map={
        #     "execute_and_save": execute_and_save,
        # },
        is_termination_msg=termination_msg,
        llm_config=base_cfg,
    )

    return [agent0, retriever, agent2, agent3]


# rc: https://microsoft.github.io/autogen/blog/2023/10/18/RetrieveChat/


def _reset_agents(agents: List[ConversableAgent]) -> None:
    [agent.reset() for agent in agents]


def init_rag_gc(problem) -> None:
    agent0, retriever, agent2, agent3 = create_research_team()
    _reset_agents([agent0, retriever, agent2, agent3])
    # del agents
    groupchat = GroupChat(
        agents=[agent0, retriever, agent2, agent3],
        messages=[],
        max_round=13,
    )
    # manager = GroupChatManager(groupchat=groupchat, llm_config=lmconf)
    # manager_conf = nested_conf.copy()
    # manager_conf.pop("functions")
    manager = GroupChatManager(groupchat=groupchat)
    retriever.initiate_chat(
        manager,
        problem=problem,
    )


def call_rag_chat(problem=PROBLEM):
    agent0, retriever, agent2, agent3 = create_research_team(base_cfg, termination_msg)
    _reset_agents([agent0, retriever, agent2, agent3])

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.

    def retrieve_content(message, n_results=4):
        retriever.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = retriever._check_update_context(
            message
        )
        if (update_context_case1 or update_context_case2) and retriever.update_context:
            retriever.problem = (
                message if not hasattr(retriever, "problem") else retriever.problem
            )
            _, ret_msg = retriever._generate_retrieve_user_reply(message)
        else:
            ret_msg = retriever.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message

    # def activate_rag(message):
    #     return retriever.retrieve_docs(message)

    retriever.human_input_mode = "NEVER"
    nested_conf = {
        "functions": [
            {
                "name": "retrieve_content",
                "description": "retrieve page_content for code generation and question answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Refined message which keeps the original meaning and can be used to retrieve page_content for code generation and research summarization.",
                        }
                    },
                    "required": ["message"],
                },
            },
        ],
        "config_list": base_cfg,
        "timeout": 60,
    }

    for a in [agent2, agent3]:
        a.llm_config.update(nested_conf)

    for agent in [agent0, agent2, agent3]:
        # register functions for all agents.
        agent.register_function(
            function_map={
                "retrieve_content": retrieve_content,
                # "activate_rag": activate_rag,
            }
        )

    groupchat = GroupChat(
        agents=[agent0, agent2, agent3],
        messages=[],
        max_round=13,
    )
    # manager = GroupChatManager(groupchat=groupchat, llm_config=lmconf)
    # manager_conf = nested_conf.copy()
    # manager_conf.pop("functions")
    manager = GroupChatManager(groupchat=groupchat, llm_config=base_cfg)
    retriever.initiate_chat(
        manager,
        problem=problem,
    )


if __name__ == "__main__":
    from agents.coordinator import Coordinator
    from agents.functions import Functions

    multiagent_coordinator = Coordinator(
        team_name="research_team",
        agents=create_research_team(),
        instruments=Functions,
    )

    res = multiagent_coordinator.sequential_conversation(prompt=PROBLEM)
    print(res)

    # call_rag_chat(
    #     problem="Please summarize the agent tuning paper for me, use the retrieve content function to get more information about the agent tuning paper"
    # ) or init_rag_gc(
    #     problem="Please summarize the agent tuning paper for me, use the retrieve content function to get more information about the agent tuning paper"
    # )
