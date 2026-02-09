import csv
import logging

from core import CSVWriter, EvaluationPipeline, PipelineConfig
from tracking import CometLogger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    # Comet settings
    WORKSPACE = "cmpe492-team"
    PROJECT = "cmpe492-rag-pipeline-evaluation-tool"

    config = PipelineConfig(
        chunk_size=400,
        chunk_overlap=50,
        splitter_type="token",
        llm_model="gpt-4o",
        personas_k=3,
        tasks_n=2,
        qas_m=2,
        rag_endpoint_url=None,
        output_csv_path="qa_library.csv",
    )

    pipeline = EvaluationPipeline(config)
    entries = pipeline.run("your_data.txt", human_labeled=None)
    CSVWriter(config.output_csv_path).write(entries)
    logging.info("Wrote %s rows to %s", len(entries), config.output_csv_path)

    logger = CometLogger(workspace=WORKSPACE, project=PROJECT)
    logger.log_params(vars(config))
    with open(config.output_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        rows = list(reader)
    logger.log_table_from_rows("qa_library.csv", rows, headers)
    logger.end()


if __name__ == "__main__":
    main()
