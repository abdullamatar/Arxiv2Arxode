import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import autogen
from autogen import (ConversableAgent, GroupChat, GroupChatManager,
                     gather_usage_summary)

import lib.functions as functions
from agents.agent import EmbeddingRetrieverAgent, GCManager, marl
from agents.agent_conf import base_cfg, gcconf, retrieve_conf

logger = logging.getLogger("coordinator")
logging.basicConfig(
    level=logging.INFO,
    filename="./logs/coordinator.log",
    format="%(asctime)s  👁‍🗨  %(levelname)s  👁‍🗨 from mod %(module)s:\n%(message)s",
)

"""
?ATTENTION:
Registering reply functions, and potentially further sublclassing the CodingAgent are the way to move forward I believe.
"""


@dataclass
class Chat:
    sender: str
    receiver: str
    message: str


@dataclass
class ConversationResult:
    success: bool
    messages: List[Chat]
    cost: float
    tokens: int
    last_message_str: str
    error_message: str


class Coordinator:
    """Manage sequential multi-agent conversations"""

    def __init__(
        self,
        team_name: str,
        agents: List[autogen.ConversableAgent],
    ):
        self.team_name = team_name
        self.agents = agents
        self.messages = []
        # List of chats - {sender, receiver, message}
        self.chats: List[Chat] = []

        assert len(self.agents) > 1, "Must have at least 2 agents"

    def _reset_agents(self, agents: List[ConversableAgent]) -> None:
        [agent.reset() for agent in agents]

    def log_gc_messages(self, msgs: List[dict]) -> None:
        for m in msgs:
            agent_name = m.get("name", "Unknown")
            content = m.get("content", "Error getting content")
            logger.info(f"SENDER {agent_name}:\nCONTENT: {content}")

    async def a_code_gen_group_chat(self, prompt: str) -> None:
        """
        Run a group chat with the agents and generate code
        """
        self._reset_agents(self.agents)
        main_uprox, retriever, code_review, coding_llm = self.agents
        gc = GroupChat(
            agents=[main_uprox, code_review, coding_llm],
            messages=[],
            max_round=50,
            speaker_selection_method="auto",
        )

        @main_uprox.register_for_execution()
        @code_review.register_for_llm(
            description="Retrieve additional information to complete the given task. Create a detailed query so that the retrieval is impactful in terms of information gained."
        )
        @coding_llm.register_for_llm(
            description="Retrieve additional information to complete the given task. Create a detailed query so that the retrieval is impactful in terms of information gained."
        )
        def retrieve_content(
            message: str,
            n_results: int = 7,
            # retriever: ConversableAgent = retriever,
        ) -> str:
            retriever.n_results = n_results

            (
                update_context_case1,
                update_context_case2,
            ) = retriever._check_update_context(message)

            if (
                update_context_case1 or update_context_case2
            ) and retriever.update_context:
                retriever.problem = (
                    message if not hasattr(retriever, "problem") else retriever.problem
                )
                _, ret_msg = retriever._generate_retrieve_user_reply(message)
            else:
                ret_msg = retriever.generate_init_message(message, n_results=n_results)
            return ret_msg if ret_msg else message

        # for agent in [code_review, coding_llm]:
        #     if isinstance(agent, autogen.AssistantAgent):
        #         logger.info(f"updating llm_config for {agent.name}")
        #         agent.llm_config.update(retrieve_conf)

        # agent.register_function(
        #     function_map={
        #         "retrieve_content": retrieve_content,
        #     }
        # )

        gcman = GroupChatManager(groupchat=gc, llm_config=base_cfg)
        await main_uprox.a_initiate_chat(
            gcman,
            message=prompt,
        )

    def code_gen_group_chat(self, prompt: str, epochs: int = 5) -> None:
        """
        Run a group chat with the agents and generate code
        """
        self._reset_agents(self.agents)
        main_uprox, retriever, code_review, coding_llm = self.agents  # type: ignore
        retriever: EmbeddingRetrieverAgent

        gc = GroupChat(
            agents=[main_uprox, code_review, coding_llm],
            messages=[],
            max_round=10,
            speaker_selection_method="auto",
            allow_repeat_speaker=[coding_llm],
        )

        @main_uprox.register_for_execution()
        @code_review.register_for_llm(
            description="Retrieve additional information to complete the given task. Create a detailed query so that the retrieval is impactful in terms of information gained. Return several docs and use them in the context."
        )
        @coding_llm.register_for_llm(
            description="Retrieve additional information to complete the given task. Create a detailed query so that the retrieval is impactful in terms of information gained. Return several docs and use them in the context."
        )
        def retrieve_content(
            message: str,
            # n_results: int = 10,
            # retriever: ConversableAgent = retriever,
        ) -> str:
            # retriever.n_results = 7

            assert message, "Message for database query not parsed correctly"
            # logger.info(f"Retrieving content for the given QUERY: {message}")
            (
                update_context_case1,
                update_context_case2,
            ) = retriever._check_update_context(message=message)

            if (
                update_context_case1 or update_context_case2
            ) and retriever.update_context:
                retriever.problem = (
                    message if not hasattr(retriever, "problem") else retriever.problem
                )
                _, ret_msg = retriever._generate_retrieve_user_reply(messages=message)
            else:
                _context = {"problem": message}
                ret_msg = retriever.message_generator(retriever, None, _context)
                # ret_msg = retriever.generate_init_message(message=message, n_results=7)
            return ret_msg if ret_msg else message

        # !!!!!!!!!!!!! THIS FIXME IS C R I T I C A L !!!!!!!!!!!!!
        # FIXME: Runtime ratelimit error: https://microsoft.github.io/autogen/docs/Use-Cases/enhanced_inference/#runtime-error
        # TODO: Start with model conf, rag can use gpt3.5, if anything...
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: Why the coding_agent returns None sometimes...
        # TODO: https://github.com/olimoz/AI_Teams_AutoGen/blob/main/JupyterNotebooksForAutoGen.ipynb

        gcman = GCManager(groupchat=gc, llm_config=gcconf)

        # gcman.register_reply(
        #     trigger=[autogen.Agent, None],
        #     reply_func=functions.get_code_blocks,
        #     config={"callback": None},
        # )

        # # gcman._groupchat.

        # main_uprox.register_reply(
        #     trigger=[autogen.Agent, None],
        #     reply_func=functions.get_code_blocks,
        #     config={"callback": None},
        # )

        # init chat calls the main reply function registered with the gcman, which is the run_chat function def'd in the GCman class
        main_uprox.initiate_chat(
            gcman,
            message=prompt,
            # clear_history=False,
        )

        # logger.info(f"gcman last message: {gcman.last_message(coding_llm)}")
        # logger.info(f"Entire msg history: {gcman.chat_messages}")
        # logger.info(
        #     f"Agent descriptions: {[agent.description for agent in self.agents]}"
        # )

        # main_uprox.send(
        #     "What was the last message exchanged?",
        #     recipient=gcman,
        # )

        # !EVAL
        # combined_stats = {
        #     "task_description": prompt,
        #     "usage_stats": {},
        #     "exe_feedback": [],
        # }
        # all_agents = self.agents + [gcman]
        # total, _ = gather_usage_summary(all_agents)
        # combined_stats["usage_stats"] = total
        # combined_stats["exe_feedback"] = gcman.execution_feedback_list

        # exit_codes = [
        #     feedback["exit_code"] for feedback in gcman.execution_feedback_list
        # ]

        # combined_stats["exit_codes"] = exit_codes
        # with open("combined_stats.jsonl", "a") as f:
        #     f.write(json.dumps(combined_stats) + "\n")
        # !EVAL

        # logger.info(f"Total usage summary: {gather_usage_summary([gcman])}")


if __name__ == "__main__":
    # asyncio.run(
    #     Coordinator(
    #         team_name="test",
    #         agents=create_rl_team(),
    #         functions=functions.Functions,
    #     ).a_code_gen_group_chat(
    #         "Recreate a minimal concept from the agent tuning paper for me in a self-contained python file."
    #     )
    # )

    print("Starting the coordinator")
    Coordinator(
        team_name="test",
        agents=marl(collection_name="eval_db"),
    ).code_gen_group_chat(
        "What does the author say in his bibliographic remarks section?"
    )
