import json

from radon.cli import Config
from radon.complexity import cc_visit

# import radon.visitors as vs
from radon.metrics import h_visit


def analyze_code_complexity(code):
    try:
        # cc_visit only considers code in functions and within classes

        # complexity_results is a list of each function within an attempted generation and each functions respective complexity, an average is taken for simplicity.
        complexity_results = cc_visit(code)
        cc_avg_over_attempts = sum(
            func.complexity if isinstance(func.complexity, (float, int)) else 0
            for func in complexity_results
        ) / len(complexity_results)
        # print(cc.__next__())
        # exit()
        # average_complexity = cc / len(complexity_results)
        # print(complexity_results)
        # if
        # return complexity_results
        return {
            "avg_cc_over_functions_within_attempt": cc_avg_over_attempts,
            # "cc": complexity_results,
            # "num_functions": len(complexity_results),
        }
    except Exception as e:
        print(f"Unexpected error during analysis: {e}")
        return {"avg_cc_over_functions_within_attempt": -1}


def halstead_metrics(code):
    try:
        halstead_results = h_visit(code)
        return halstead_results.total._asdict() if halstead_results else {}
    except Exception as e:
        print(e)
        return {}


def process_jsonl_file(file_path, output_path):
    compiled_results = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            # codes_for_task = {}
            # cc_metrics_task = []
            # halstead_metrics_task = []
            attempts = []
            for i, feedback in enumerate(data.get("exe_feedback", [])):
                # print(feedback)
                # exit()
                code = feedback.get("code", "")
                # codes_for_task["attempt_{i}"] = code
                # i += 1
                # codes_for_task.append(code)

                cc_metrics = analyze_code_complexity(code)
                halstead_metrics_result = halstead_metrics(code)

                attempts.append(
                    {
                        f"attempt_{i}": {
                            "code": code,
                            "cc_metrics": cc_metrics,
                            "halstead_metrics": halstead_metrics_result,
                        }
                    }
                )
                # cc_metrics_task.append(cc_metrics)
                # halstead_metrics_task.append(halstead_metrics_result)

            compiled_results.append(
                {
                    "task_desc": data.get("task_description"),
                    "attempt_results": attempts,
                }
            )

    with open(output_path, "w") as out_file:
        json.dump(compiled_results, out_file, indent=4)


if __name__ == "__main__":
    file_path = "./lib/eval/stats/combined_stats_a2a.jsonl"
    process_jsonl_file(file_path, "./lib/eval/stats/cc_a2a_gpt4.json")

    file_path = "./lib/eval/stats/combined_stats_a2a_turbo.jsonl"
    process_jsonl_file(file_path, "./lib/eval/stats/cc_a2a_turbo.json")

    file_path = "./lib/eval/stats/combined_stats_baseGPT.jsonl"
    process_jsonl_file(file_path, "./lib/eval/stats/cc_baseGPT.json")

    file_path = "./lib/eval/stats/combined_stats_base_turbo.jsonl"
    process_jsonl_file(file_path, "./lib/eval/stats/cc_base_turbo.json")
