# NOTE: This file was made using nbonvert with some minor changes, there were some issues using ray within a note book and I cant fix them rn

# stdlib
import json
import logging
import os
import signal
import sys

from filelock import FileLock

# relative import error with python modules and what not, the path needs to change according to cwd :)
project_root = os.path.dirname("./")
if project_root not in sys.path:
    sys.path.append(project_root)

# anthropic
import anthropic
# ray
import ray
# autogen
from autogen import OpenAIWrapper

from agents.agent import marl
# a2a
# from ...agents.agent import CodingAssistant
from agents.agent_conf import base_cfg, base_claude_cfg
from lib.embeddings import get_db_connection

# import lib.eval.MLBench.MLBench_eval as mlb
# from agents.coordinator import Coordinator
# from agents.agent import marl


def load_tasks(file_path: str):
    with open(file_path, "r") as file:
        for line in file:
            task = json.loads(line)
            task_description = task["task_description"]
            task_idx = task["task_idx"]

            yield {
                "task_description": task_description,
                "task_idx": task_idx,
            }


# avg words/prompt
# sum([len(t.split())for t in tasks])/len(tasks)
# below is more efficient than sum([i for i in j]) -> [] creates an unnecessary intermediate array
# whereas using purely the generator (which is valid because sum expects and iterable) yields the lens for sum to consume one at a time.
def avg_words_per_task_description(task_generator):
    total_words, task_count = map(
        sum,
        zip(*((len(task["task_description"].split()), 1) for task in task_generator)),
    )
    return total_words / task_count if task_count > 0 else 0


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException("Prompt skipped due to timeout")


signal.signal(signal.SIGALRM, handler)


# Note: Do not set the signal handler at the global level in the worker processes
# We'll set it within the remote function if needed


@ray.remote
def process_task(task_desc, fname):
    signal.signal(signal.SIGALRM, handler)
    # Recreate client, dbconn, and coding_agent within the function
    # client = OpenAIWrapper(config_list=base_claude_cfg["config_list"])
    #! NOTE: CLIENT
    client = anthropic.Anthropic()
    dbconn = get_db_connection("MLBench_papers")

    coding_agent = marl("MLBench_papers")[-1]

    try:
        signal.alarm(300)

        vec_db_query_response = client.messages.create(
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": f"Create a query for the following task:\n{task_desc['task_description']}\n"
                    "The query will be used to search a vector database with relevant research papers in it, keep the query short, useful, and relevant. "
                    "DO NOT RESPOND WITH ANYTHING BUT A QUERY".strip(),
                }
            ],
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            # cache_seed=33, #full
            # cache_seed=89,  # ID Python ONLY
        )
        vec_db_query = vec_db_query_response.content[0].text
        assert vec_db_query, "No query generated for vector database search."

        # Retrieve context from the vector database
        retrieved_excerpts = dbconn.similarity_search(vec_db_query)
        context = [d.page_content for d in retrieved_excerpts]
        formatted_context = "".join(
            [f"Context {i + 1}: {doc}\n" for i, doc in enumerate(context)]
        )

        exe_feedback = []
        attempts = 0
        generated_content = ""
        # Change to anthropic

        while attempts < 5:
            system_message = (
                "You are tasked with generating code based on research papers."
                "Think through the process step by step and generate complete working code."
                "Do not fill in functions with 'pass' or ellipses, and do not make assumptions without justification. "
                f"Here is the code generated from your previous attempt:\n{exe_feedback[-1]['code'] if exe_feedback else ''}\n"
                f"Here is the previous execution log:\n{exe_feedback[-1]['logs'] if exe_feedback else ''}".strip()
            )
            messages = [
                # {
                #     "role": "system",
                #     "content": "You are tasked with generating code based on research papers. "
                #     "Think through the process step by step and generate complete working code. "
                #     "Do not fill in functions with 'pass' or ellipses, and do not make assumptions without justification. "
                #     f"Here is the code generated from your previous attempt:\n{exe_feedback[-1]['code'] if exe_feedback else ''}\n"
                #     f"Here is the previous execution log:\n{exe_feedback[-1]['logs'] if exe_feedback else ''}",
                # },
                {
                    "role": "user",
                    "content": task_desc["task_description"].strip(),
                },
                {
                    "role": "assistant",
                    "content": f"Here is retrieved information related to the task:\n{formatted_context}".strip(),
                },
            ]

            try:
                signal.alarm(120)
                # !NOTE
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_message,
                    temperature=1,
                    messages=messages,
                    max_tokens=8192,
                )
                # generated_content = response.content
                generated_content = response.content[0].text
                assert generated_content
                # Execute the generated code
                ef = coding_agent.detect_and_execute_code(generated_content)
                exe_feedback.append(*ef)
                signal.alarm(0)
            except Exception as e:
                exe_feedback.append(
                    {"code": generated_content, "exit_code": -1, "logs": str(e)}
                )
            finally:
                attempts += 1
                signal.alarm(0)

        exit_codes = [feedback["exit_code"] for feedback in exe_feedback]
        combined_stats = {
            "task_description": task_desc["task_description"],
            "exe_feedback": exe_feedback,
            "task_idx": task_desc["task_idx"],
            "exit_codes": exit_codes,
        }

        with FileLock(fname + ".lock"):
            with open(fname, "a") as f:
                f.write(json.dumps(combined_stats) + "\n")

        logger.info(f"Task {task_desc['task_idx']} processed successfully.")

    except TimeoutException:
        logger.warning(f"Task {task_desc['task_idx']} timed out.")
    except Exception as e:
        logger.error(f"Error processing task {task_desc['task_idx']}: {e}")
    finally:
        signal.alarm(0)


# temperature

if __name__ == "__main__":
    ray.init(_temp_dir="/mnt/mainHD/tmp")
    # NOTE
    tasks_path = "./runs/a2a_claude/claudequarter.jsonl"

    FNAME = "./lib/eval/baseruns/baseClaude_quarter.jsonl"
    logger = logging.getLogger(__name__)

    tasks = load_tasks(tasks_path)

    processed_task_ids = set()
    try:
        with open(FNAME, "r") as f:
            for line in f:
                data = json.loads(line)
                processed_task_ids.add(data["task_idx"])
    except FileNotFoundError:
        pass

    tasks_to_process = [
        task for task in tasks if task["task_idx"] not in processed_task_ids
    ]

    try:
        futures = [process_task.remote(t, FNAME) for t in tasks_to_process]

        ray.get(futures)
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        pass
    # finally:
    #     ray.shutdown()
