# STD LIB
import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

# autogen
import autogen
from autogen import (ConversableAgent, GroupChat, GroupChatManager,
                     gather_usage_summary)
from autogen.agentchat.contrib.capabilities import context_handling
# TruLens
from trulens_eval.tru_custom_app import instrument

# A2A
# import lib.functions as functions
from agents.agent import EmbeddingRetrieverAgent, GCManager, marl
from agents.agent_conf import base_cfg, gcconf, retrieve_conf

# from autogen.agentchat.contrib.capabilities.context_handling import (
#     truncate_str_to_tokens,
#     TransformChatHistory,
# )



logger = logging.getLogger("coordinator")
logging.basicConfig(
    level=logging.INFO,
    filename="./logs/final_convs.log",
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

    @instrument
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

        # TODO: https://github.com/olimoz/AI_Teams_AutoGen/blob/main/JupyterNotebooksForAutoGen.ipynb
        gcman = GCManager(groupchat=gc, llm_config=gcconf)

        # 16k token limits for gpt-3.5 family #The below token limitation stuff can be ignrored when using gpt3.5 see the function in agents.py file
        # it is a hacky, unstable work around as the below Transform was not working right when AutoGen released it and I had to change the code internally to get it to work. :).

        # However, there is not harm in trying to execute the below if running into token limit problems when testing models with smaller context windows :/.

        # ctx = context_handling.TransformChatHistory(
        #     transforms=[
        #         context_handling.MessageTokenLimiter(max_tokens_per_message=15000)
        #     ],
        # )

        # for agent in [main_uprox, code_review, coding_llm]:
        #     ctx.add_to_agent(agent)
        # ctx.add_to_agent(gcman)

        # init chat calls the main reply function registered with the gcman, which is the run_chat function def'd in the GCman class
        main_uprox.initiate_chat(
            gcman,
            message=prompt,
            # clear_history=False,
        )

        assert (
            gc.messages
        ), "Something has gone awry..."  # This assertion will almost certainly never fail.

        self.log_gc_messages(gc.messages)

        # !EVAL

        combined_stats = {
            "task_description": prompt,
            "usage_stats": {},
            "exe_feedback": [],
        }
        all_agents = self.agents + [gcman]
        total, _ = gather_usage_summary(all_agents)
        combined_stats["usage_stats"] = total
        combined_stats["exe_feedback"] = gcman.execution_feedback_list

        exit_codes = [
            feedback["exit_code"] for feedback in gcman.execution_feedback_list
        ]

        combined_stats["exit_codes"] = exit_codes
        with open("additional_retries_a2a_turbo.jsonl", "a") as f:
            f.write(json.dumps(combined_stats) + "\n")

        # !EVAL


if __name__ == "__main__":

    # with open("./temp.jsonl", "r") as f:
    #     tasks = [json.loads(line) for line in f]

    # for i in tasks:
    #     td = i["task_description"]
    #     vecdb = i["vec_db"]

    # print(f"Launching coordinator for task: {td} and associated db {vecdb}")

    Coordinator(
        team_name="test",
        agents=marl(collection_name="eval2_db_rlsemiparam_with_code_embeddings"),
    ).code_gen_group_chat(
        "Create a python file that highlights how exactly experience memory can be updated using a RL policy, recreate a minimal executable example for me, do not make any assumptions or fill any functions with the pass keyword or ellipses."
    )

# for task in tasks:
#     Coordinator(
#         team_name="test",
#         agents=marl(collection_name="cs_demo"),
#     ).code_gen_group_chat(task)
# Coordinator(
#     team_name="test",
#     agents=marl(collection_name="eval4_db_guidingpretraining"),
# ).code_gen_group_chat(
#     "Show me how an RL agents exploration can be guiding with LLM priors according to the paper. Create a minimal example in a self-contained python file that must be executable and produce an output, do not make any assumptions or fill any functions with the pass keyword or ellipses."
# )
