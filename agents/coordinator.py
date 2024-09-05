# STD LIB
import json
import logging
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from typing import List

# autogen
import autogen
from autogen import ConversableAgent, GroupChat, gather_usage_summary
# hf
from datasets import load_from_disk

# A2A
from agents.agent import EmbeddingRetrieverAgent, GCManager, marl
from agents.agent_conf import gcconf

# import multiprocessing as mp




# TruLens
# from trulens_eval.tru_custom_app import instrument

logger = logging.getLogger("coordinator")

logging.getLogger("requests").propagate = False
logging.getLogger("urllib3").propagate = False

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(
    level=logging.INFO,
    # Get the root directory of the project
    filename=os.path.join(root_dir, "logs/coord_gc.log"),
    format="%(asctime)s <>|<> %(levelname)s <>|<> from mod %(module)s:\n%(message)s",
)


class Coordinator:
    """Coordinate sequential multi-agent conversations"""

    def __init__(
        self,
        team_name: str,
        agents: List[autogen.ConversableAgent],
    ):
        self.team_name = team_name
        self.agents = agents
        self.messages = []

    def _reset_agents(self, agents: List[ConversableAgent]) -> None:
        [agent.reset() for agent in agents]

    def log_gc_messages(self, msgs: List[dict]) -> None:
        for m in msgs:
            agent_name = m.get("name", "Unknown")
            print(m)
            content = (
                m
                if isinstance(m, str)
                else m.get("content", "ERROR RETRIEVING MESSAGE CONTENT")
            )
            logger.info(f"SENDER {agent_name}:\nCONTENT: {content}")

    # !
    # @instrument
    def code_gen_group_chat(self, prompt: str, fname, task_idx, epochs: int = 7):
        """
        Run a group chat with the agents and generate code
        """
        self._reset_agents(self.agents)
        main_uprox, retriever, code_review, coding_llm = self.agents  # type: ignore
        retriever: EmbeddingRetrieverAgent

        gc = GroupChat(
            agents=[retriever, code_review, coding_llm],
            messages=[],
            max_round=epochs,
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

        gcman = GCManager(groupchat=gc, llm_config=gcconf, tid=task_idx)

        # NOTE: https://microsoft.github.io/autogen/docs/topics/groupchat/transform_messages_speaker_selection
        # Above url for transforming token lengt
        # 16k token limits for gpt-3.5 family #The below token limitation stuff can be ignrored when using gpt3.5 see the function in agents.py file
        # it is a hacky, unstable work around as the below Transform was not working right when AutoGen released it and I had to change the code internally to get it to work. :).

        # However, there is no harm in trying to execute the below if running into token limit problems when testing models with smaller context windows :/.

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

        combined_stats = {
            "task_description": prompt,
            "usage_stats": {},
            "exe_feedback": [],
        }
        # all_agents = self.agents + [gcman]
        total, _ = gather_usage_summary(self.agents + [gcman])
        combined_stats["usage_stats"] = total
        combined_stats["exe_feedback"] = gcman.execution_feedback_list
        combined_stats["exit_codes"] = [
            feedback["exit_code"] for feedback in gcman.execution_feedback_list
        ]
        combined_stats["task_idx"] = task_idx

        with open(fname, "a") as f:
            f.write(json.dumps(combined_stats) + "\n")

        return combined_stats


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException("Prompt skipped due to timeout")


signal.signal(signal.SIGALRM, handler)


# def g():
#     with ProcessPoolExecutor() as exe:
#         exe.map(execute_tasks())


def execute_tasks(agents: List[ConversableAgent], mlbench, subset: str, iteration):
    # function to skip over blocking stdin prompts for whatever reason (i,e. an api key being required via some library).
    try:
        signal.alarm(420)

        Coordinator(
            team_name="test",
            agents=agents,
        ).code_gen_group_chat(
            f"Here is some information from a readme related to the task at hand, this information is important to successfully implement the following task use it as guidance and a starting point. You are operating within a JUPYTER IPYKERNEL NOTEBOOK ENVIRONMENT:\n {mlbench[subset][iteration]['oracle']}\nTo prepare the execution environment please run this command in the notebook: {mlbench[subset][iteration]['prefix_code']}\n"
            f"Here is the task description, this is your main goal to fulfil:\n{mlbench[subset][iteration]['instruction']}\nAvoid executing any things that require standard inputs to be given, such as an API key or token.\n"
            f"If the task requires the github repository here it is, remember to prefix shell commands with an exclamation mark: {mlbench[subset][iteration]['github']}",
            # f"The type of output expected for this task is {mlbench[subset][iteration]['type']}.",
            epochs=7,
            task_idx=mlbench[subset][iteration]["id"],
            fname="mlfull.jsonl",
        )

        signal.alarm(0)
        logger.info(f"Iteration {iteration + 1} completed successfully.")
        # return results
    except TimeoutException:
        logger.info(f"Skipping prompt for iteration {iteration + 1}")


if __name__ == "__main__":
    mlbench = load_from_disk(root_dir + "/lib/eval/MLBench/datasets/")
    agents = marl(collection_name="MLBench_papers")
    subset = "full"

    # with ProcessPoolExecutor() as executor:
    #     for iteration in range(len(mlbench[subset])):
    #         executor.submit(execute_tasks, agents, mlbench, subset, iteration)

    for i in range(88, len(mlbench[subset])):
        execute_tasks(agents=agents, mlbench=mlbench, subset=subset, iteration=i)

# def mp_execute_task(task_queue, result_queue):
#     while not task_queue.empty():
#         try:
#             task = task_queue.get_nowait()
#             if task is None:
#                 break

#             signal.alarm(300)

#             result = Coordinator(
#                 team_name="test",
#                 agents=marl(collection_name="MLBench_papers"),
#             ).code_gen_group_chat(
#                 f"Here is some information from a readme related to the task at hand, this information is more than enough to successfully implement the following task and is of utmost importance, use it as guidance and a starting point to build from:\n {task['oracle']}\nTo prepare the execution environment please run this command: {task['prefix_code']}\n"
#                 f"Here is the task description:\n{task['instruction']}\nAvoid any executing things that require standard inputs to given, such as an API key or token.\n"
#                 f"The type of output expected for this task is {task['type']}.",
#                 epochs=7,
#                 task_idx=task["id"],
#             )

#             signal.alarm(0)

#             logger.info(f"Task {task['id']} completed successfully.")
#             result_queue.put(result)

#         except TimeoutException:
#             logger.info(f"Skipping task {task['id']} due to timeout")
#         except Exception as e:
#             logger.error(f"Error in task {task['id']}: {str(e)}")

#     result_queue.put("&done&")


# def listener(result_queue, output_file):
#     with open(output_file, "a") as f:
#         while True:
#             result = result_queue.get()
#             if result == "&done&":
#                 break
#             f.write(json.dumps(result) + "\n")
#             f.flush()


# def run_multiproc(
#     mlbench, subset, num_workers=4, output_file="agent_runs_multiproc.jsonl"
# ):
#     task_queue = mp.Queue()
#     result_queue = mp.Queue()

#     for task in mlbench[subset]:
#         task_queue.put(task)
#         # break

#     listener_process = mp.Process(target=listener, args=(result_queue, output_file))
#     listener_process.start()

#     worker_processes = []
#     for _ in range(num_workers):
#         worker = mp.Process(target=mp_execute_task, args=(task_queue, result_queue))
#         worker.start()
#         worker_processes.append(worker)

#     # wait for workers to finish
#     for worker in worker_processes:
#         worker.join()

#     result_queue.put(None)
#     listener_process.join()


# def process_task(task, log_dir="../logs", epochs=7):
#     try:
#         signal.alarm(300)  # Set task timeout

#         result = Coordinator(
#             team_name="test",
#             agents=marl(collection_name="MLBench_papers"),
#         ).code_gen_group_chat(
#             f"Here is some information from a readme related to the task at hand, this information is more than enough to successfully implement the following task and is of utmost importance, use it as guidance and a starting point to build from:\n {task['oracle']}\nTo prepare the execution environment please run this command: {task['prefix_code']}\n"
#             f"Here is the task description:\n{task['instruction']}\nAvoid any executing things that require standard inputs to given, such as an API key or token.\n"
#             f"The type of output expected for this task is {task['type']}.",
#             epochs=epochs,
#             task_idx=task["id"],
#         )

#         signal.alarm(0)  # Disable the alarm

#         logger.info(f"Task {task['id']} completed successfully.")
#         return result

#     except TimeoutException:
#         logger.info(f"Skipping task {task['id']} due to timeout")
#         return None
#     except Exception as e:
#         logger.error(f"Error in task {task['id']}: {str(e)}")
#         return None


# def run_multiproc2(
#     mlbench, subset, output_file="agent_runs_multiproc.jsonl", max_workers=4
# ):
#     tasks = mlbench[subset]
#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(process_task, task): task for task in tasks}
#         with open(output_file, "a") as f:
#             for future in as_completed(futures):
#                 result = future.result()
#                 if result is not None:
#                     f.write(json.dumps(result) + "\n")
#                     f.flush()

#     logger.info(f"Finished processing {len(results)} tasks.")
