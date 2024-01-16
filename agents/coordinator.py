import json
import logging
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import autogen
from autogen import GroupChat, GroupChatManager

import agents.functions as functions

# TODO: Change curr_usage to actual lib file
import lib.curr_usage as curr_usage
from agents.agent_conf import retrieve_conf, base_cfg
from agents.agent import EmbeddingRetrieverAgent

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

    def code_gen(self, task_description: str) -> ConversationResult:
        for _ in range(15):
            for idx, agent in enumerate(self.agents):
                if isinstance(agent, EmbeddingRetrieverAgent):
                    retrieved_docs = agent.retrieve_docs(task_description, n_results=13)
                    retrieved_info = (
                        "\n\n".join([doc[0] for doc in retrieved_docs["documents"]])
                        if "documents" in retrieved_docs
                        else ""
                    )
                    self.agents[idx + 1].send({"content": retrieved_info}, agent)

                elif isinstance(agent, autogen.UserProxyAgent):
                    next_agent = self.agents[idx + 1]
                    last_message = agent.last_message()
                    if last_message is not None:
                        next_agent.send(last_message, agent)

                elif isinstance(agent, autogen.AssistantAgent):
                    # Assuming generate_code_execution_reply returns a string with code
                    code_generation_response = agent.generate_code_execution_reply(
                        current_query
                    )
                    if code_generation_response is not None:
                        generated_code += code_generation_response + "\n\n"
                        current_query = code_generation_response

        return generated_code


if __name__ == "__main__":
    # coordinator = Coordinator(
    #     team_name="test",
    #     agents=curr_usage.create_research_team(),
    #     # agents=[
    #     #     autogen.ConversableAgent(name="agent1"),
    #     #     autogen.ConversableAgent(name="agent2"),
    #     #     autogen.ConversableAgent(name="agent3"),
    #     # ],
    #     functions=functions.Functions,
    # )

    # coordinator.sequential_conversation(prompt="Create a random python function")
    Coordinator(
        team_name="test",
        agents=curr_usage.create_research_team(),
        functions=functions.Functions,
    ).code_gen(
        "Explain the newton method of solving equations for me and give me a minimal python implementation"
    )
