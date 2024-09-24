import asyncio
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List

import autogen
import ray
from autogen import ConversableAgent, GroupChat, gather_usage_summary
from datasets import load_from_disk
from filelock import FileLock

from arxiv2arxode.agents.agent import EmbeddingRetrieverAgent, GCManager, marl
from arxiv2arxode.agents.agent_conf import gcconf

# import multiprocessing as mp


# TruLens
# from trulens_eval.tru_custom_app import instrument

logger = logging.getLogger("coordinator")

logging.getLogger("requests").propagate = False
logging.getLogger("urllib3").propagate = False

# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: Fix async logging, thats some sh*t I DO NOT know about.
logging.basicConfig(
    level=logging.INFO,
    # Get the root directory of the project
    filename=os.path.join(root_dir, "logs/coord_gc.log"),
    format="%(asctime)s <>|<> %(levelname)s <>|<> from mod %(module)s:\n%(message)s",
)


class Coordinator:
    """Initiate and track multi-agent conversations"""

    def __init__(
        self,
        agents: List[autogen.ConversableAgent],
    ):
        self.agents = agents

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
    def code_gen_group_chat(self, prompt: str, task_idx, fname=None, epochs: int = 7):
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
        # Above url for transforming token length
        # 16k token limits for gpt-3.5 family
        # The below token limitation stuff can be ignored when using gpt3.5 see the _get_context function in agents.py file
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
        total, _ = gather_usage_summary(self.agents + [gcman])  # type: ignore
        combined_stats["usage_stats"] = total
        combined_stats["exe_feedback"] = gcman.execution_feedback_list
        combined_stats["exit_codes"] = [
            feedback["exit_code"] for feedback in gcman.execution_feedback_list
        ]
        combined_stats["task_idx"] = task_idx

        # with FileLock(fname + ".lock"):
        #     with open(fname, "a") as f:
        #         logger.info("WE ARE TRYING TO WRITE")
        #         f.write(json.dumps(combined_stats) + "\n")
        #         f.flush()
        #         os.fsync(f.fileno())

        return combined_stats


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException("Prompt skipped due to timeout")


# NOTE: Apparently signal.alarm can be problematic in multiproc'd/threaded envs
# Yes it can... If code executed in docker triggers an api token request from stdin the execution will hang indefinitely. No timeouts present have stopped it, not from the docker executor, or the signal.alarm...


# @ray.remote
# class AgentHolder:
#     def __init__(self, fname):
#         self.agents = marl(collection_name="MLBench_papers")
#         self.fname = fname
#         self.logger = logger

#     def process_task(self, entry):
#         try:
#             self.logger.info(f"Starting code_gen_group_chat for task id {entry['id']}.")

#             x = Coordinator(
#                 agents=self.agents,
#             ).code_gen_group_chat(
#                 prompt=(
#                     f"Here is some information from a readme related to the task at hand; "
#                     f"use it as guidance and a starting point. You are operating within a JUPYTER "
#                     f"IPYKERNEL NOTEBOOK ENVIRONMENT:\n {entry['oracle']}\nTo prepare the execution "
#                     f"environment please run this command in the notebook: {entry['prefix_code']}\n"
#                     f"Here is the task description, this is your main goal to fulfil:\n{entry['instruction']}\n"
#                     f"Avoid executing any things that require standard inputs to be given, such as an API key or token.\n"
#                     f"If the task requires the GitHub repository, here it is; remember to prefix shell commands with "
#                     f"an exclamation mark: {entry['github']}"
#                 ),
#                 epochs=7,
#                 task_idx=entry["id"],
#                 # fname=None,
#             )

#             self.logger.info(f"Finished code_gen_group_chat for task id {entry['id']}.")

#             self.logger.info(
#                 f"Writing results for task id {entry['id']} to file {self.fname}."
#             )
#             with FileLock(self.fname + ".lock"):
#                 with open(self.fname, "a") as f:
#                     f.write(json.dumps(x) + "\n")
#                     f.flush()
#                     os.fsync(f.fileno())
#             self.logger.info(
#                 f"Successfully wrote results for task id {entry['id']} to file."
#             )

#         except TimeoutException:
#             self.logger.info(
#                 f"Skipping prompt for iteration {entry['id']} due to timeout."
#             )
#         except Exception as e:
#             self.logger.error(
#                 f"Error processing task {entry['id']}: {e}", exc_info=True
#             )


@ray.remote
class AsyncAgentHolder:
    def __init__(self, fname):
        # self.logger = logging.getLogger("coordinator")

        # self.agents = marl(collection_name="MLBench_papers")
        self.fname = fname

        # self.executor = ProcessPoolExecutor(max_workers=6)

    async def process_task(self, entry):
        try:
            logger.info(f"Starting process_task for task id {entry['id']}.")

            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=4) as executor:
                await loop.run_in_executor(
                    executor, self.blocking_process_and_write, entry
                )

            logger.info(
                f"Successfully processed and wrote results for task id {entry['id']}."
            )

        except Exception as e:
            logger.error(f"Error processing task {entry['id']}: {e}", exc_info=True)

    def blocking_code_gen_group_chat(self, entry):
        # agents = marl()
        agents = marl(collection_name="MLBench_papers")
        prompt = (
            f"Here is some information from a readme related to the task at hand:"
            f"use it as guidance and a starting point. You are operating within a JUPYTER "
            f"IPYKERNEL NOTEBOOK ENVIRONMENT:\n {entry['oracle']}\nTo prepare the execution "
            f"environment please run this command in the notebook: {entry['prefix_code']}\n"
            f"Here is the task description, this is your main goal to fulfil:\n{entry['instruction']}\n"
            f"Avoid executing any things that require standard inputs to be given, such as an API key or token.\n"
            f"If the task requires the GitHub repository, here it is; remember to prefix shell commands with "
            f"an exclamation mark: {entry['github']}"
        )

        x = Coordinator(
            agents=agents,
        ).code_gen_group_chat(
            prompt=prompt,
            epochs=7,
            task_idx=entry["id"],
            fname=None,
        )

        return x

    def blocking_process_and_write(self, entry):
        x = self.blocking_code_gen_group_chat(entry)
        logger.info(f"Writing results for task id {entry['id']} to file {self.fname}.")
        with FileLock(self.fname + ".lock"):
            with open(self.fname, "a") as f:
                f.write(json.dumps(x) + "\n")
                f.flush()
                os.fsync(f.fileno())

        return x


if __name__ == "__main__":

    ray.init(_temp_dir="/mnt/mainHD/tmp")

    mlbench = load_from_disk(root_dir + "/lib/eval/MLBench/datasets/")
    subset = "full"

    fname = os.path.abspath("claudefull.jsonl")

    num_actors = os.cpu_count()
    agents = [AsyncAgentHolder.remote(fname) for _ in range(num_actors)]

    procd_tasks = set()
    try:
        with open(fname, "r") as f:
            for line in f:
                data = json.loads(line)
                procd_tasks.add(data["task_idx"])
    except FileNotFoundError:
        pass

    tasks_to_process = [j for j in mlbench[subset] if j["id"] not in procd_tasks][:5]

    tasks = []
    for i, entry in enumerate(tasks_to_process):
        agent = agents[i % num_actors]
        tasks.append(agent.process_task.remote(entry))

    try:
        ray.get(tasks)
    except Exception as e:
        logger.error(f"Exception during ray.get(tasks): {e}", exc_info=True)

    #!NON ASYNC
    # ray.init(_temp_dir="/mnt/mainHD/tmp")

    # mlbench = load_from_disk(root_dir + "/lib/eval/MLBench/datasets/")
    # subset = "full"
    # fname = "claudefull.jsonl"

    # fname = os.path.abspath(fname)

    # procd_tasks = set()
    # try:
    #     with open(fname, "r") as f:
    #         for line in f:
    #             data = json.loads(line)
    #             procd_tasks.add(data["task_idx"])
    # except FileNotFoundError:
    #     pass

    # tasks_to_process = [j for j in mlbench[subset] if j["id"] not in procd_tasks]

    # # ?
    # tasks_to_process = tasks_to_process[:5]  # ?
    # num_actors = os.cpu_count() // 2  # ?
    # # ?

    # ray_actors = [AgentHolder.remote(fname) for _ in range(num_actors)]

    # futures = []
    # for i, t in enumerate(tasks_to_process):
    #     agent = ray_actors[i % len(ray_actors)]
    #     futures.append(agent.process_task.remote(t))

    # try:
    #     ray.get(futures)
    # except Exception as e:
    #     logger.error(f"Exception during ray.get(futures): {e}", exc_info=True)
    #! NON ASYNC END

    # tasks = mlbench[subset]

    # task_manager = TaskManager.remote(tasks, fname)

    # agent_holders_ray_actor = [AgentHolder.remote(fname) for _ in range(os.cpu_count())]

    # run_results = [actor.run.remote(task_manager) for actor in agent_holders_ray_actor]

    # ray.get(run_results)

    # procd_tasks = set()
    # try:
    #     with open(fname, "r") as f:
    #         for line in f:
    #             data = json.loads(line)
    #             procd_tasks.add(data["task_idx"])
    # except FileNotFoundError:
    #     pass

    # tasks_to_process = [j for j in mlbench[subset] if j["id"] not in procd_tasks]

    # ray_agent_actors = [AgentHolder.remote(fname) for _ in range(os.cpu_count())]
    # futures = []
    # try:
    #     for i, t in enumerate(tasks_to_process):
    #         actor = ray_agent_actors[i % int(os.cpu_count())]
    #         futures.append(actor.process_task.remote(t))

    #     remaining_futs = futures
    #     while remaining_futs:
    #         done_futures, remaining_futures = ray.wait(
    #             remaining_futs,
    #             num_returns=1,
    #             timeout=300,
    #         )  # num rets to 1 for instant file write? Moved filewritting here because Im pretty sure I was deadlocking, but it was working at some point, such is code :)
    #         for future in done_futures:
    #             result = ray.get(future)
    #             if result["status"] == "success":
    #                 combined_stats = result["result"]
    #                 with open(fname, "a") as f:
    #                     f.write(json.dumps(combined_stats) + "\n")
    # except Exception as e:
    #     logger.info(f"A7A: {e}")
    #     pass

    # try:
    #     futures = [agent_holder.process_task.remote(t) for t in tasks_to_process]

    #     ray.get(futures)
    # except Exception:
    #     # TODO: Properly ID the exception and handle it...
    #     pass

    # for i in range(len(mlbench[subset])):
    #     execute_tasks(agents=agents, mlbench=mlbench, subset=subset, iteration=i)

    # for i in (j for j in mlbench[subset] if j["type"] == "Python Code"):
    #     execute_tasks2(agents=agents, i=i)

# @ray.remote
# class TaskManager:
#     def __init__(self, tasks, output_file):
#         self.tasks = tasks
#         self.output_file = output_file
#         self.processed_task_ids = set()
#         self.index = 0
#         self._load_processed_tasks()

#     def _load_processed_tasks(self):
#         try:
#             with open(self.output_file, "r") as f:
#                 for line in f:
#                     data = json.loads(line)
#                     self.processed_task_ids.add(data["task_idx"])
#         except FileNotFoundError:
#             pass

#     def get_next_task(self):
#         while self.index < len(self.tasks):
#             task = self.tasks[self.index]
#             self.index += 1
#             task_id = task["id"]
#             if task_id in self.processed_task_ids:
#                 continue
#             else:
#                 self.processed_task_ids.add(task_id)
#                 return task
#         return None
