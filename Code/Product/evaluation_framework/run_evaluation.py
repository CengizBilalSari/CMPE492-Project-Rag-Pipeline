import csv
import logging
from dataclasses import asdict
import os
from core import EvalConfig, RAGEvaluator
from tracking import CometLogger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    config = EvalConfig(
        comet_experiment_key=os.getenv("QA_EXPERIMENT_KEY"),
        rag_endpoint_url=None,
        llm_provider="openai",
        llm_model="gpt-4o",
        output_csv_path="evaluation_results.csv",
    )

    logger_comet = CometLogger(workspace=config.comet_workspace, project=config.comet_project)

    if config.comet_experiment_key:
        logging.info(
            "Downloading qa_library.csv from Comet experiment %s …",
            config.comet_experiment_key,
        )
        qa_rows = logger_comet.download_table(
            experiment_key=config.comet_experiment_key,
            asset_name="qa_library.csv",
        )
    logging.info("Loaded %s QA rows", len(qa_rows))

    evaluator = RAGEvaluator(config)
    eval_rows = evaluator.run(qa_rows)

    evaluator.write_csv(eval_rows, config.output_csv_path)

    logger_comet.log_params(asdict(config))

    with open(config.output_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        rows = list(reader)
    logger_comet.log_table_from_rows("evaluation_results.csv", rows, headers)

    agg = RAGEvaluator.aggregate(eval_rows)
    logger_comet.log_metrics(agg)
    logging.info("Aggregate metrics: %s", agg)

    logger_comet.end()
    logging.info("Done — Comet experiment finished.")


if __name__ == "__main__":
    main()
