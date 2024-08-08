# STD LIB
# import asyncio
# import subprocess
# import threading
# import queue
import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import List

# autogen
import autogen
from autogen import ConversableAgent, GroupChat, gather_usage_summary
# hf
from datasets import load_from_disk

# A2A
# import lib.functions as functions
from agents.agent import EmbeddingRetrieverAgent, GCManager, marl
from agents.agent_conf import gcconf

# TruLens
# from trulens_eval.tru_custom_app import instrument


# from autogen.agentchat.contrib.capabilities import context_handling


logger = logging.getLogger("coordinator")

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    # Get the root directory of the project
    filename=os.path.join(root_dir, "logs/coord_gc.log"),
    format="%(asctime)s  👁‍🗨  %(levelname)s  👁‍🗨 from mod %(module)s:\n%(message)s",
)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

"""
?ATTENTION:
Registering reply functions, and potentially further sublclassing the CodingAgent are the way to move forward I believe.
"""


@dataclass
class Chat:
    sender: str
    receiver: str
    message: str


class Coordinator:
    """Manage sequential multi-agent conversations"""

    def __init__(
        self,
        team_name: str,
        agents: List[autogen.ConversableAgent],
    ):
        self.team_name = team_name
        self.agents = agents
        # messages & chats? Something seems off, I blame autogen :p
        self.messages = []
        # List of chats - {sender, receiver, message}
        self.chats: List[Chat] = []

        assert len(self.agents) > 1, "Must have at least 2 agents"

    def _reset_agents(self, agents: List[ConversableAgent]) -> None:
        [agent.reset() for agent in agents]

    def log_gc_messages(self, msgs: List[dict]) -> None:
        for m in msgs:
            agent_name = m.get("name", "Unknown")
            print(m)
            content = m["content"] or "fail"
            logger.info(f"SENDER {agent_name}:\nCONTENT: {content}")

    # !
    # @instrument
    def code_gen_group_chat(self, prompt: str, epochs: int = 5) -> None:
        """
        Run a group chat with the agents and generate code
        """
        self._reset_agents(self.agents)
        main_uprox, retriever, code_review, coding_llm = self.agents  # type: ignore
        retriever: EmbeddingRetrieverAgent

        gc = GroupChat(
            agents=[retriever, code_review, coding_llm],
            messages=[],
            max_round=5,
            speaker_selection_method="auto",
            # allow_repeat_speaker=[coding_llm],
        )

        @main_uprox.register_for_execution()
        @coding_llm.register_for_execution()
        @code_review.register_for_execution()
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
        retriever.initiate_chat(
            gcman,
            message=retriever.message_generator,
            problem=prompt,
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
        with open("agent_runs.jsonl", "a") as f:
            f.write(json.dumps(combined_stats) + "\n")

        # !EVAL


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException("Prompt skipped due to timeout")


signal.signal(signal.SIGALRM, handler)


def perform_iteration(mlbench, iteration):
    # function to skip over blocking stdin prompts for whatever reason (i,e. api being required via some library).
    try:
        signal.alarm(80)

        Coordinator(
            team_name="test",
            agents=marl(collection_name="MLBench_papers"),
        ).code_gen_group_chat(
            f"Here is some information from a readme related to the task at hand:\n {mlbench['quarter'][iteration]['oracle']}\n"
            f"Here is the task description:\n{mlbench['quarter'][iteration]['instruction']}\n"
            f"The type of output expected for this task is {mlbench['quarter'][iteration]['type']}"
        )

        signal.alarm(0)
        logger.info(f"Iteration {iteration + 1} completed successfully.")
    except TimeoutException:
        logger.info(f"Skipping prompt for iteration {iteration + 1}")


if __name__ == "__main__":

    mlbench = load_from_disk(root_dir + "/lib/eval/MLBench/datasets/")

    # with open("./temp.jsonl", "r") as f:
    #     tasks = [json.loads(line) for line in f]

    # for i in range(len(mlbench["quarter"][20])):
    perform_iteration(mlbench, 20)
    # Function to read output from a subprocess without blocking
    # def enqueue_output(out, q):
    #     for line in iter(out.readline, b""):
    #         q.put(line.decode("utf-8"))
    #     out.close()

    # def run_task(task_instruction):
    #     script_content = f"""
    # import subprocess

    # def main():
    # Coordinator(
    #     team_name="test",
    #     agents=marl(collection_name="MLBench_papers"),
    # ).code_gen_group_chat("retrieve some content about AC-Gans")

    # if __name__ == "__main__":
    #     main()
    # """
    #     # Write the script content to a temporary file
    #     script_file = "temp_task_script.py"
    #     with open(script_file, "w") as f:
    #         f.write(script_content)

    #     # Start the subprocess
    #     process = subprocess.Popen(
    #         ["python", script_file],
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,
    #         stdin=subprocess.PIPE,
    #         bufsize=1,
    #         universal_newlines=True,
    #     )

    #     # Create a queue to hold the subprocess output
    #     q = queue.Queue()
    #     t = threading.Thread(target=enqueue_output, args=(process.stdout, q))
    #     t.daemon = True  # Thread dies with the program
    #     t.start()

    #     # Monitor the subprocess output
    #     while True:
    #         try:
    #             line = q.get_nowait()  # Non-blocking read
    #         except queue.Empty:
    #             if process.poll() is not None:
    #                 break  # Process finished
    #             time.sleep(0.1)
    #         else:
    #             print(line, end="")  # Print the output

    #             # Check if the line contains a text prompt
    #             if "Enter" in line or "Prompt" in line:
    #                 print("Detected prompt, terminating subprocess...")
    #                 process.terminate()
    #                 return False  # Indicate the round should be skipped

    #     # Wait for the process to finish
    #     process.wait()
    #     return True  # Indicate the round completed successfully

    # # mlbench = load_from_disk(root_dir + "/lib/eval/MLBench/datasets/")

    # results = []

    # for i in range(len(mlbench["quarter"])):
    #     task_instruction = (
    #         f"Here is some information from a readme related to the task at hand:\n {mlbench['quarter'][i]['oracle']}\n"
    #         f"Here is the task description:\n{mlbench['quarter'][i]['instruction']}\n"
    #         f"The type of output expected for this task is {mlbench['quarter'][i]['type']}"
    #     )
    #     success = run_task(task_instruction)
    #     if success:
    #         print(f"Task {i + 1} completed successfully.")
    #         results.append(True)
    #     else:
    #         print(f"Task {i + 1} skipped due to prompt.")
    #         results.append(False)
    #     logger.info(f"Results: {results}")

##################old########################
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
