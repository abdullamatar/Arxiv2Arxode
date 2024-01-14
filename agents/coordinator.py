import json
import logging
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple
import autogen
import agents.functions as functions
from agents.agent_conf import retrieve_conf

from autogen import GroupChat, GroupChatManager

# TODO: Change curr_usage to actual lib file
import lib.curr_usage as curr_usage

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

    def sequential_conversation(self, prompt: str) -> ConversationResult:
        """
        Runs a sequential conversation between agents.

        The most common type of conversation.

        For example
            "Agent A" -> "Agent B" -> "Agent C" -> "Agent D" -> "Agent E"
        """

        logger.info(
            f"\n\n>>>>>>>>> {self.team_name} Orchestrator Starting <<<<<<<<<\n\n"
        )

        self.store_msg(prompt)

        for idx, _ in enumerate(self.agents):
            agent_a = self.agents[idx]
            agent_b = self.agents[idx + 1]

            logger.info(
                f"\n\n>>>>>>>>> Running iteration {idx} with (agent_a: {agent_a.name}, agent_b: {agent_b.name}) <<<<<<<<<\n\n"
            )

            logger.info(f"Latest message: {self.latest_message}")

            # agent_a -> chat -> agent_b
            if self.last_message_is_string:
                logger.info(f"basic_chat triggered last message is string")
                self.basic_chat(agent_a, agent_b, self.latest_message)

            # agent_a -> func() -> agent_b
            if self.last_message_is_func_call and self.has_functions(agent_a):
                logger.info(f"function_chat triggered last message is func")
                self.function_chat(agent_a, agent_b, self.latest_message)

            self.log_curr_state()

            logger.info(f"Latest message: {self.latest_message}")
            logger.info(f"index: {idx}, total_agents: {self.total_agents}")

            if idx == self.total_agents - 2:
                if self.has_functions(agent_b):
                    # agent_b -> func() -> agent_b
                    self.self_function_chat(agent_b, self.latest_message)

                logger.info(f">>>>>>>> Orchestrator Complete <<<<<<<<\n\n")

                was_successful, error_message = self.handle_validate_func()

                self.log_curr_state()

                cost, tokens = self.get_cost_and_tokens()

                return ConversationResult(
                    success=was_successful,
                    messages=self.messages,
                    cost=cost,
                    tokens=tokens,
                    last_message_str=self.last_message_as_string,
                    error_message=error_message,
                )

    def arxiv2arxode(self, task_description: str) -> ConversationResult:
        for agent in self.agents:
            agent.reset()
        (
            main_userprox,
            retriever,
            code_reviewer,
            coding_llm,
        ) = self.agents

        # context = retriever.query_vector_db(query_texts=task_description)
        # code_gen = ""
        groupchat = GroupChat(
            agents=[main_userprox, code_reviewer, coding_llm],
            messages=[],
            max_round=44,
            allow_repeat_speaker=False,
            speaker_selection_method="auto",
        )

        def retrieve_content(message, n_results=7, retriever=retriever):
            retriever.n_results = (
                n_results  # Set the number of results to be retrieved.
            )
            # Check if we need to update the context.
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

        for agent in [main_userprox, code_reviewer, coding_llm]:
            # register functions for all agents.
            agent.register_function(
                function_map={
                    "retrieve_content": retrieve_content,
                }
            )

        manager = GroupChatManager(groupchat=groupchat)
        main_userprox.initiate_chat(
            manager,
            message=task_description,
        )

        code_draft = ""
        for iteration in range(15):
            # Step 1: Retrieve relevant content
            retrieved_content = retrieve_content(task_description)
            manager.run_chat(retrieved_content)

            # Step 2: Generate initial code based on the retrieved content and task description
            if iteration == 0:
                initial_code_prompt = f"Based on the following information: {retrieved_content}, create Python code to accomplish the task: {task_description}"
                manager.send_message_to_group(initial_code_prompt, sender=main_userprox)
                code_draft = coding_llm.generate_code(initial_code_prompt)
            else:
                refinement_prompt = f"Refine the following Python code based on the new information: {retrieved_content}\n\n{code_draft}"
                manager.send_message_to_group(refinement_prompt, sender=main_userprox)
                code_draft = coding_llm.refine_code(refinement_prompt)

            # Step 3: Review the generated/refined code
            review_prompt = f"Please review this Python code: {code_draft}"
            manager.send_message_to_group(review_prompt, sender=main_userprox)
            review_result = code_reviewer.review_code(review_prompt)

            # Check if the code is satisfactory or needs further refinement
            if code_reviewer.is_code_satisfactory(review_result):
                break  # Exit loop if code is satisfactory

            # Update the task description with review feedback for the next iteration
            task_description = f"{task_description}\n\nFeedback: {review_result}"

        # After the conversation rounds, retrieve the final generated code
        final_code = code_draft  # The final version of the code after iterations

        # Return the final code as part of the conversation result
        return ConversationResult(
            success=True,
            messages=groupchat.messages,
            cost=0,  # Update if cost calculation is needed
            tokens=0,  # Update if token usage is tracked
            last_message_str=final_code,
            error_message="",
        )


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
    ).arxiv2arxode(task_description="Agent tuning minimal example")
