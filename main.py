import json
import logging
import os
import argparse
import time

from llm_autoeval.table import make_table, make_final_table
from llm_autoeval.upload import upload_to_github_gist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = os.getenv("MODEL")
BENCHMARK = os.getenv("BENCHMARK")
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")


def main(directory: str, elapsed_time: float) -> None:
    if not GITHUB_API_TOKEN:
        logger.error("GITHUB_API_TOKEN environment variable is not set.")
        raise ValueError("GITHUB_API_TOKEN is required to upload results as a Gist.")

    if not MODEL or not BENCHMARK:
        logger.error("MODEL or BENCHMARK environment variable is missing.")
        raise ValueError("MODEL and BENCHMARK are required to summarize results.")

    # Define tasks based on the benchmark
    tasks = {
        "openllm": ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"],
        "nous": ["AGIEval", "GPT4All", "TruthfulQA", "Bigbench"],
        "tiny": ["tinyArc", "tinyHellaswag", "tinyMMLU", "tinyTruthfulQA", "tinyTruthfulQA_mc1", "tinyWinogrande"]
    }.get(BENCHMARK, None)

    if tasks is None:
        logger.error(f"Invalid BENCHMARK value: {BENCHMARK}")
        raise ValueError(f"Invalid BENCHMARK value: {BENCHMARK}")

    # Load task results
    tables = []
    averages = []

    for task in tasks:
        file_path = os.path.join(directory, f"{task.lower()}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                table, average = make_table(data, task)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                table, average = f"### {task}\nError: {str(e)}\n", None
        else:
            table = f"### {task}\nError: File does not exist\n"
            average = None

        tables.append(table)
        averages.append(average)

    # Build summary
    summary = ""
    for i, task in enumerate(tasks):
        summary += tables[i]
        if averages[i] is not None:
            summary += f"Average: {averages[i]}%\n\n"
        else:
            summary += "Average: Not available due to error\n\n"

    # Final average calculation
    valid_averages = [a for a in averages if isinstance(a, (int, float))]
    if valid_averages:
        final_average = round(sum(valid_averages) / len(valid_averages), 2)
        summary += f"Average score: {final_average}%\n"
    else:
        summary += "Average score: Not available due to errors\n"

    # Elapsed time
    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    summary += f"\nElapsed time: {elapsed}"

    # Generate final table
    result_dict = {task: avg for task, avg in zip(tasks, averages) if avg is not None}
    final_table = make_final_table(result_dict, MODEL)
    summary = final_table + "\n\n" + summary

    # Upload to GitHub Gist
    logger.info("Uploading results to GitHub Gist...")
    gist_name = f"{MODEL.split('/')[-1]}-{BENCHMARK.capitalize()}.md"
    try:
        upload_url = upload_to_github_gist(summary, gist_name, GITHUB_API_TOKEN)
        logger.info(f"Gist successfully created: {upload_url}")
        print(f"Gist URL: {upload_url}")
    except Exception as e:
        logger.error(f"Failed to upload Gist: {e}")
        raise


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Summarize results and upload them.")
    parser.add_argument("directory", type=str, help="The path to the directory with the JSON results")
    parser.add_argument("elapsed_time", type=float, help="Elapsed time since the start of the evaluation")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the directory exists
    if not os.path.isdir(args.directory):
        raise ValueError(f"The directory {args.directory} does not exist.")

    # Call the main function
    main(args.directory, args.elapsed_time)
