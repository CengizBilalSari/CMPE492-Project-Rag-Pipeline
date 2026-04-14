"""
MedQA Benchmark Service

Evaluates GPT model performance on medical question answering using the MedQA dataset.
Uses Comet Opik for LLM request tracing and experiment tracking.
"""

import os
import argparse
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in script directory
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from datasets import load_dataset
from openai import OpenAI
from opik import track
from opik.integrations.openai import track_openai
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Result of a single benchmark question."""
    question_id: str
    question: str
    options: dict
    correct_answer: str
    model_answer: str
    is_correct: bool


@dataclass
class BenchmarkSummary:
    """Summary statistics of a benchmark run."""
    total_questions: int
    correct_answers: int
    accuracy: float
    results: list[BenchmarkResult]


class MedQABenchmark:
    """Benchmark service for evaluating GPT on MedQA dataset."""
    
    def __init__(self, model: str = "gpt-4o-mini", project_name: str = "medqa-benchmark"):
        """
        Initialize the benchmark service.
        
        Args:
            model: OpenAI model to use for evaluation
            project_name: Comet Opik project name for tracking
        """
        self.model = model
        os.environ["OPIK_PROJECT_NAME"] = project_name
        
        # Initialize OpenAI client with Opik tracking
        client = OpenAI()
        self.client = track_openai(client)
    
    def load_dataset(self, split: str = "train", sample_size: Optional[int] = None) -> list:
        """
        Load MedQA dataset from HuggingFace.
        
        Args:
            split: Dataset split to load ('train', 'test', 'validation')
            sample_size: Number of samples to load (None for all)
            
        Returns:
            List of dataset samples
        """
        dataset = load_dataset("openlifescienceai/medqa", split=split)
        
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        return list(dataset)
    
    def format_question(self, sample: dict) -> str:
        """
        Format a MedQA sample into a prompt for GPT.
        
        Args:
            sample: Dataset sample with question and options
            
        Returns:
            Formatted prompt string
        """
        data = sample["data"]
        question = data["Question"]
        options = data["Options"]
        
        prompt = f"""You are a medical expert. Answer the following multiple-choice medical question.
Reply with ONLY the letter (A, B, C, D, or E) of the correct answer.

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}
"""
        # Some questions may have option E
        if 'E' in options and options['E']:
            prompt += f"E) {options['E']}\n"
        
        prompt += "\nAnswer:"
        return prompt
    
    def parse_answer(self, response: str) -> str:
        """
        Parse GPT response to extract the answer letter.
        
        Args:
            response: GPT response text
            
        Returns:
            Single letter answer (A-E) or 'INVALID'
        """
        # Clean the response
        response = response.strip().upper()
        
        # Check for single letter answer
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if response.startswith(letter):
                return letter
        
        # Try to find letter in response
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in response:
                return letter
        
        return 'INVALID'
    
    @track
    def evaluate_question(self, sample: dict) -> BenchmarkResult:
        """
        Evaluate a single question from the dataset.
        
        Args:
            sample: Dataset sample
            
        Returns:
            BenchmarkResult with evaluation details
        """
        prompt = self.format_question(sample)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        
        model_response = completion.choices[0].message.content
        model_answer = self.parse_answer(model_response)
        data = sample["data"]
        correct_answer = data["Correct Option"]
        
        return BenchmarkResult(
            question_id=sample["id"],
            question=data["Question"],
            options=data["Options"],
            correct_answer=correct_answer,
            model_answer=model_answer,
            is_correct=model_answer == correct_answer
        )
    
    @track
    def run_benchmark(self, sample_size: Optional[int] = None, split: str = "train") -> BenchmarkSummary:
        """
        Run the full benchmark on the MedQA dataset.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            split: Dataset split to use
            
        Returns:
            BenchmarkSummary with results and statistics
        """
        print(f"Loading MedQA dataset (split: {split})...")
        samples = self.load_dataset(split=split, sample_size=sample_size)
        print(f"Loaded {len(samples)} samples")
        
        results = []
        correct_count = 0
        
        print("Running benchmark...")
        for sample in tqdm(samples):
            result = self.evaluate_question(sample)
            results.append(result)
            if result.is_correct:
                correct_count += 1
        
        accuracy = (correct_count / len(samples)) * 100 if samples else 0
        
        summary = BenchmarkSummary(
            total_questions=len(samples),
            correct_answers=correct_count,
            accuracy=accuracy,
            results=results
        )
        
        print(f"\n{'='*50}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"Model: {self.model}")
        print(f"Total Questions: {summary.total_questions}")
        print(f"Correct Answers: {summary.correct_answers}")
        print(f"Accuracy: {summary.accuracy:.2f}%")
        print(f"{'='*50}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run MedQA benchmark on GPT")
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=10,
        help="Number of samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="medqa-benchmark",
        help="Comet Opik project name (default: medqa-benchmark)"
    )
    
    args = parser.parse_args()
    
    benchmark = MedQABenchmark(model=args.model, project_name=args.project_name)
    summary = benchmark.run_benchmark(sample_size=args.sample_size, split=args.split)
    
    return summary


if __name__ == "__main__":
    main()
