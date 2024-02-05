import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager

import agents.functions as functions
# TODO: Change curr_usage to actual lib file
import lib.curr_usage as curr_usage
from agents.agent import EmbeddingRetrieverAgent
from agents.agent_conf import base_cfg, retrieve_conf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename="logs/coordinator.log",
    format="%(asctime)s  👁‍🗨  %(levelname)s  👁‍🗨  :\n%(message)s",
)


@dataclass
class Chat:
    """Chat message"""

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
        functions: functions.Functions,
    ):
        self.team_name = team_name
        self.agents = agents
        self.messages = []
        self.functions = functions()
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

        logger.info(
            f"agent confs: {json.dumps(code_review.llm_config, indent=4)}\n {json.dumps(coding_llm.llm_config, indent=4)}\n {json.dumps(main_uprox.llm_config, indent=4)}\n {json.dumps(retriever.llm_config, indent=4)}"
        )
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
        logger.info(f"Entire msg history: {gcman.chat_messages}")

    def code_gen_group_chat(self, prompt: str, epochs: int = 5) -> None:
        """
        Run a group chat with the agents and generate code
        """
        self._reset_agents(self.agents)
        main_uprox, retriever, code_review, coding_llm = self.agents
        # for idx in range(epochs):
        #     break

        gc = GroupChat(
            agents=[main_uprox, code_review, coding_llm],
            messages=[],
            max_round=(4 + 1),
            speaker_selection_method="auto",
            allow_repeat_speaker=[retriever, coding_llm],
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
            retriever.n_results = 7

            logger.info(f"Retrieving content for the given QUERY: {message}")
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
                ret_msg = retriever.generate_init_message(message, n_results=7)
            return ret_msg if ret_msg else message

        # for agent in [code_review, coding_llm]:
        #     if isinstance(agent, autogen.AssistantAgent):
        #         logger.info(f"updating llm_config for {agent.name}")
        #         agent.llm_config.update(retrieve_conf)
        logger.info(
            f"agent confs: {json.dumps(code_review.llm_config, indent=4)}\n {json.dumps(coding_llm.llm_config, indent=4)}\n {json.dumps(main_uprox.llm_config, indent=4)}\n {json.dumps(retriever.llm_config, indent=4)}"
        )
        # agent.register_function(
        #     function_map={
        #         "retrieve_content": retrieve_content,
        #     }
        # )

        gcman = GroupChatManager(groupchat=gc, llm_config=base_cfg)

        # NOTE: Below comment...
        # self._groupchat_manager = gcman

        main_uprox.initiate_chat(
            gcman,
            message=prompt,
        )
        # logger.info(f"gcman last message: {gcman.last_message(coding_llm)}")
        # logger.info(f"Entire msg history: {gcman.chat_messages}")
        # logger.info(
        #     f"Agent descriptions: {[agent.description for agent in self.agents]}"
        # )

        main_uprox.send(
            "Can you combine the last python script we generated with a new concept? Query the optomisation methods to see what topics are available for us to merge with the last script.",
            recipient=gcman,
        )
        self.log_gc_messages(gc.messages)


if __name__ == "__main__":
    # asyncio.run(
    #     Coordinator(
    #         team_name="test",
    #         agents=curr_usage.create_research_team(),
    #         functions=functions.Functions,
    #     ).a_code_gen_group_chat(
    #         "Recreate a minimal concept from the agent tuning paper for me in a self-contained python file. Start by exploring the paper and codebase via the infohoarder."
    #     )
    # )

    Coordinator(
        team_name="test",
        agents=curr_usage.create_research_team(),
        functions=functions.Functions,
    ).code_gen_group_chat("Implement newtons method for optimization in python.")
