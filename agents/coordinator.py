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
)

# This class is adapted from: https://github.com/disler/multi-agent-postgres-data-analytics/blob/v10-talk-to-your-database-beta-launch/postgres_da_ai_agent/modules/orchestrator.py


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
        validate_results_func: callable = None,
    ):
        self.team_name = team_name
        self.agents = agents
        self.messages = []
        self.functions = functions()
        # List of chats - {sender, receiver, message}
        self.chats: List[Chat] = []

        self.validate_results_func: callable = validate_results_func

        assert len(self.agents) > 1, "Must have at least 2 agents"

    @property
    def total_agents(self):
        return len(self.agents)

    @property
    def last_message_is_dict(self):
        return isinstance(self.messages[-1], dict)

    @property
    def last_message_is_string(self):
        return isinstance(self.messages[-1], str)

    @property
    def last_message_is_func_call(self):
        return self.last_message_is_dict and self.latest_message.get(
            "function_call", None
        )

    @property
    def last_message_is_content(self):
        return self.last_message_is_dict and self.latest_message.get("content", None)

    @property
    def latest_message(self) -> Optional[str]:
        if not self.messages:
            return None
        return self.messages[-1]

    @property
    def last_message_as_string(self):
        if not self.messages:
            return ""
        if self.last_message_is_content:
            return self.latest_message.get("content", "")
        return str(self.messages[-1])

    def send_msg(
        self,
        from_agent: autogen.ConversableAgent,
        to_agent: autogen.ConversableAgent,
        msg: str,
    ):
        from_agent.send(msg, to_agent)
        self.chats.append(Chat(from_agent.name, to_agent.name, str(msg)))

    def store_msg(self, msg):
        self.messages.append(msg)

    def get_msg(self) -> str:
        str_msg = ""
        for m in self.messages:
            if m is None:
                continue
            elif isinstance(m, dict):
                dict_content = m.get("content", None)
                func_call = m.get("function_call", None)
                content = dict_content or func_call
                if content:
                    str_msg += content
                elif content is None:
                    continue
            elif isinstance(m, str):
                str_msg += str(m)
        return str_msg

    def basic_chat(
        self,
        agent_a: autogen.ConversableAgent,
        agent_b: autogen.ConversableAgent,
        message: str,
    ):
        logger.info(
            f"basic_chat triggered with agents: {agent_a.name} -> {agent_b.name} and \
                    message: {message}"
        )
        self.send_msg(agent_a, agent_b, message)

        reply = agent_b.generate_reply(sender=agent_a)

        self.store_msg(reply)

    def mem_chat(
        self,
        agent_a: autogen.ConversableAgent,
        agent_b: autogen.ConversableAgent,
        message: str,
    ):
        logger.info(f"mem_chat: {agent_a.name} -> {agent_b.name}")
        self.send_msg(agent_a, agent_b, message)

        reply = agent_b.generate_reply(sender=agent_a)

        # ???????
        self.send_msg(agent_b, agent_a, reply)
        self.store_msg(reply)

    def function_chat(
        self,
        agent_a: autogen.ConversableAgent,
        agent_b: autogen.ConversableAgent,
        message: str,
    ):
        logger.info(f"function_call(): {agent_a.name} -> {agent_b.name}")

        self.basic_chat(agent_a, agent_a, message)

        assert self.last_message_is_content, "No content in last message"

        self.basic_chat(agent_a, agent_b, self.latest_message)

    def self_function_chat(self, agent: autogen.ConversableAgent, message: str):
        logger.info(f"self_function_chat(): {agent.name} -> {agent.name}")

        self.send_msg(agent, agent, message)

        reply = agent.generate_reply(sender=agent)

        self.send_msg(agent, agent, message)

        self.store_msg(reply)

        logger.info(f"self_function_chat(): replied with:", reply)

    def log_curr_state(self, append_to_file: bool = True):
        conversations = []

        for chat in self.chats:
            conversations.append(asdict(chat))

        if append_to_file:
            self.functions.write_file(
                fname=f"{self.team_name}_conversations.json",
                content=json.dumps(conversations, indent=4),
            )

    def has_functions(self, agent: autogen.ConversableAgent):
        return len(agent._function_map) > 0

    def get_cost_and_tokens(self):
        return -4, -4

    def handle_validate_func(self) -> Tuple[bool, str]:
        """
        Run the validate_results_func if it exists
        """
        if self.validate_results_func:
            return self.validate_results_func()
        return True, ""

    def _reset_agents(self, agents: List[ConversableAgent]) -> None:
        [agent.reset() for agent in agents]

    def code_gen_group_chat(self, prompt: str) -> ConversationResult:
        """
        Run a group chat with the agents and generate code
        """
        self._reset_agents(self.agents)
        main_uprox, retriever, code_review, coding_llm = self.agents
        gc = GroupChat(
            agents=[main_uprox, code_review, coding_llm],
            messages=[],
            max_round=44,
            speaker_selection_method="auto",
        )

        def retrieve_content(
            message, n_results=7, retriever: ConversableAgent = retriever
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

        for agent in [main_uprox, code_review, coding_llm]:
            # register functions for all agents.
            agent.register_function(
                function_map={
                    "retrieve_content": retrieve_content,
                }
            )
        gcman = GroupChatManager(groupchat=gc, llm_config=base_cfg)
        main_uprox.initiate_chat(
            gcman,
            message=prompt,
        )

    def code_gen(self, task_description: str) -> ConversationResult:
        generated_code = ""
        self.store_msg(task_description)
        for _ in range(15):
            for idx, agent in enumerate(self.agents):
                next_agent = self.agents[idx + 1]
                if isinstance(agent, EmbeddingRetrieverAgent):
                    retrieved_info = agent.retrieve_docs(task_description, n_results=4)[
                        "documents"
                    ]

                    retrieved_info = " ".join([s for xs in retrieved_info for s in xs])
                    assert retrieved_info, "No documents retrieved"
                    self.basic_chat(agent, next_agent, retrieved_info)
                    exit(931292931939)

                elif isinstance(agent, autogen.UserProxyAgent):
                    last_message = agent.last_message()

                    if not last_message:
                        self.basic_chat(
                            agent,
                            next_agent,
                            task_description,
                        ) if not isinstance(
                            next_agent, EmbeddingRetrieverAgent
                        ) else self.basic_chat(
                            agent,
                            next_agent,
                            task_description
                            + "\ngenerate a query related to the above task so that documents may be retrieved",
                        )
                elif isinstance(agent, autogen.AssistantAgent):
                    code_generation_response_tuple = (
                        agent.generate_code_execution_reply(current_query)
                    )
                    if code_generation_response_tuple[0]:
                        code_generation_response = code_generation_response_tuple[1]
                        generated_code += code_generation_response + "\n\n"
                        current_query = code_generation_response
                    else:
                        break

        self.log_curr_state()
        print(generated_code)
        return ConversationResult(
            success=True,
            messages=self.chats,
            cost=0,
            tokens=0,
            last_message_str=generated_code,
            error_message="",
        )


if __name__ == "__main__":
    Coordinator(
        team_name="test",
        agents=curr_usage.create_research_team(),
        functions=functions.Functions,
    ).code_gen(
        "Implement an evaluation file using MMLU and hugging face that uses an idea from the agent tuning paper."
    )
