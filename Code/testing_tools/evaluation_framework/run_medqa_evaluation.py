import csv
import logging
import argparse
import sys
import os
from dataclasses import asdict
from typing import List, Dict

# Ensure the parent directory is in the path so we can import 'core'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import EvalConfig, RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    # Determine script directory to locate default files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "medqa_cleaned.csv")
    
    parser = argparse.ArgumentParser(description="Run RAG evaluation on a custom CSV file.")
    parser.add_argument("input_csv", nargs='?', default=default_csv, help="Path to the input CSV file with columns: id, question, correct_answer")
    parser.add_argument("--output_csv", default="medqa_evaluation_results.csv", help="Path to the output CSV file")
    parser.add_argument("--rag-url", help="URL of the RAG endpoint. If not provided, assumes MOCK mode (or uses env var).")
    parser.add_argument("--rag-post", action="store_true", help="Use POST method for RAG endpoint (default is POST already, this is just explicit).")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "groq"], help="LLM provider for judgment (default: openai)")
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model for judgment (default: gpt-4o)")
    parser.add_argument("--limit", type=int, help="Limit the number of questions to evaluate (useful for testing)")
    return parser.parse_args()

def read_input_csv(file_path: str) -> List[Dict]:
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Verify columns
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no header.")
            
            required_cols = {'id', 'question', 'correct_answer'}
            if not required_cols.issubset(set(reader.fieldnames)):
                # Try to be flexible if headers are slightly different, but for now strict based on user request
                # User request: id,question,correct_answer
                missing = required_cols - set(reader.fieldnames)
                logger.warning(f"CSV might be missing columns: {missing}. Expected: id, question, correct_answer")
            
            for row in reader:
                # Map to format expected by RAGEvaluator
                # It expects: question, answer. Optional: source, ...
                qa_row = {
                    "question": row.get("question", "").strip(),
                    "answer": row.get("correct_answer", "").strip(),
                    "source": f"id:{row.get('id', 'unknown')}",
                    "question_type": "custom",
                    # Add other fields if necessary, currently RAGEvaluator uses:
                    # question, answer, question_type, source, persona, task, chunk_id
                }
                if qa_row["question"]: # distinct from empty string
                    rows.append(qa_row)
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading input CSV: {e}")
        sys.exit(1)
    return rows

def main():
    args = parse_arguments()

    # Load environment variables if needed, RAGEvaluator does load_dotenv inside core/evaluator.py imports
    # but we can also set config explicitly.
    
    # Load environment variables explicitly from script directory if .env exists there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.info(f"Loaded .env from {env_path}")

    rag_url = args.rag_url or os.getenv("RAG_ENDPOINT_URL")

    config = EvalConfig(
        rag_endpoint_url=rag_url,
        rag_http_method="post", # Defaulting to POST
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        output_csv_path=args.output_csv,
    )

    logger.info(f"Starting evaluation with config: {config}")
    logger.info(f"Reading input from: {args.input_csv}")

    qa_rows = read_input_csv(args.input_csv)
    logger.info(f"Loaded {len(qa_rows)} questions.")

    if not qa_rows:
        logger.warning("No rows to process. Exiting.")
        return

    # Apply limit if specified
    if args.limit:
        logger.info(f"Limiting evaluation to first {args.limit} questions.")
        qa_rows = qa_rows[:args.limit]

    evaluator = RAGEvaluator(config)
    
    # Run evaluation
    logger.info("Running evaluation...")
    eval_rows = evaluator.run(qa_rows)

    # Write results
    logger.info(f"Writing results to: {args.output_csv}")
    evaluator.write_csv(eval_rows, args.output_csv)

    # Log metrics
    agg = RAGEvaluator.aggregate(eval_rows)
    logger.info("Aggregate metrics:")
    for k, v in agg.items():
        logger.info(f"  {k}: {v}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
