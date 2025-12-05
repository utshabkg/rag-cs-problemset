"""
Evaluation orchestrator for CS-PROB.
Coordinates the evaluation workflow across all components.
"""
import time
from typing import Dict, List, Optional

import pandas as pd

from config import (
    MODELS_UNDER_TEST,
    SYSTEM_PROMPT,
    get_model_prompt_template,
    IMAGE_PATH_COLUMN,
    IMAGE_BASE64_COLUMN
)
from excel_reader import read_image_from_file
from llm_client import LLMClient
from judge_client import JudgeClient, rescale_score


class Question:
    """Represents a single exam question with metadata."""
    
    def __init__(
        self,
        text: str,
        reference_answer: str,
        base64_image: Optional[str] = None,
        question_number: int = 0
    ):
        """
        Initialize question.
        
        Args:
            text: Question text
            reference_answer: Correct answer
            base64_image: Base64 encoded image (optional)
            question_number: Question index for display
        """
        self.text = text
        self.reference_answer = reference_answer
        self.base64_image = base64_image
        self.question_number = question_number
    
    def get_preview(self, max_length: int = 100) -> str:
        """Get truncated question text for display."""
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."
    
    def has_image(self) -> bool:
        """Check if question has associated image."""
        return self.base64_image is not None
    
    def get_prompt(self) -> str:
        """Build prompt for LLM under test."""
        return get_model_prompt_template(self.text, self.base64_image)


class ModelEvaluationResult:
    """Stores evaluation results for one model on one question."""
    
    def __init__(self, model_name: str):
        """Initialize result container."""
        self.model_name = model_name
        self.answer: Optional[str] = None
        self.judge_scores: Dict = {}
        self.rescaled_score: Optional[float] = None
        self.error: Optional[str] = None
    
    def set_answer(self, answer: str):
        """Store model's answer."""
        self.answer = answer
        if answer.startswith("[ERROR]"):
            self.error = answer
    
    def set_judge_scores(self, scores: Dict):
        """Store judge evaluation scores."""
        self.judge_scores = scores
        
        # Rescale overall score
        overall_score = scores.get("overall_score")
        if overall_score is not None:
            self.rescaled_score = rescale_score(overall_score)
        
        # Track errors
        if "error" in scores or scores.get("parse_error"):
            self.error = scores.get("raw") or scores.get("error")
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for DataFrame row - only final score."""
        return {
            f"{self.model_name}": self.rescaled_score,
        }


class SheetEvaluator:
    """Evaluates all questions in a single Excel sheet."""
    
    def __init__(
        self,
        sheet_name: str,
        dataframe: pd.DataFrame,
        llm_clients: Dict[str, LLMClient],
        judge_client: JudgeClient,
        question_column: str,
        reference_column: str
    ):
        """
        Initialize sheet evaluator.
        
        Args:
            sheet_name: Name of the Excel sheet
            dataframe: DataFrame containing questions
            llm_clients: Dictionary mapping model names to LLMClient instances
            judge_client: JudgeClient instance for grading
            question_column: Name of question column
            reference_column: Name of reference answer column
        """
        self.sheet_name = sheet_name
        self.dataframe = dataframe
        self.llm_clients = llm_clients
        self.judge_client = judge_client
        self.question_column = question_column
        self.reference_column = reference_column
    
    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all questions in the sheet.
        
        Returns:
            DataFrame with evaluation results
        """
        self._print_header()
        
        result_rows = []
        total_questions = len(self.dataframe)
        
        for row_index, row in self.dataframe.iterrows():
            question = self._create_question_from_row(row, row_index + 1)
            results = self._evaluate_single_question(question, total_questions)
            result_row = self._build_result_row(question, results)
            result_rows.append(result_row)
        
        return pd.DataFrame(result_rows)
    
    def _print_header(self):
        """Print section header."""
        print(f"\n{'='*80}")
        print(f"PROCESSING SHEET: {self.sheet_name}")
        print(f"Total questions: {len(self.dataframe)}")
        print(f"Models under test: {len(self.llm_clients)}")
        print(f"{'='*80}\n")
    
    def _create_question_from_row(self, row: pd.Series, question_number: int) -> Question:
        """Extract Question object from DataFrame row."""
        question_text = str(row.get(self.question_column, "")).strip()
        reference_answer = str(row.get(self.reference_column, "")).strip()
        
        # Get image (embedded or from file)
        base64_image = row.get(IMAGE_BASE64_COLUMN)
        if not base64_image and IMAGE_PATH_COLUMN in self.dataframe.columns:
            image_path = row.get(IMAGE_PATH_COLUMN)
            base64_image = read_image_from_file(image_path)
        
        return Question(
            text=question_text,
            reference_answer=reference_answer,
            base64_image=base64_image,
            question_number=question_number
        )
    
    def _evaluate_single_question(
        self,
        question: Question,
        total_questions: int
    ) -> Dict[str, ModelEvaluationResult]:
        """Evaluate one question across all models."""
        print(f"\n--- Question {question.question_number}/{total_questions} ---")
        print(f"Q: {question.get_preview()}")
        print(f"Image: {'Yes' if question.has_image() else 'No'}")
        
        # Query all models
        model_answers = self._query_all_models(question)
        
        # Judge all answers
        evaluation_results = self._judge_all_answers(question, model_answers)
        
        return evaluation_results
    
    def _query_all_models(self, question: Question) -> Dict[str, str]:
        """Query all LLMs with the question."""
        print(f"\nQuerying {len(self.llm_clients)} models...")
        
        prompt = question.get_prompt()
        model_answers = {}
        
        for model_index, (model_name, llm_client) in enumerate(self.llm_clients.items(), 1):
            print(f"  [{model_index}/{len(self.llm_clients)}] {llm_client.get_short_name()}...", 
                  end=" ", flush=True)
            
            start_time = time.time()
            answer = llm_client.query(prompt)
            elapsed_time = time.time() - start_time
            
            if answer.startswith("[ERROR]"):
                print(f"✗ (error: {answer[:50]})")
            else:
                print(f"✓ ({elapsed_time:.1f}s)")
            
            model_answers[model_name] = answer
        
        return model_answers
    
    def _judge_all_answers(
        self,
        question: Question,
        model_answers: Dict[str, str]
    ) -> Dict[str, ModelEvaluationResult]:
        """Judge all model answers using the judge LLM."""
        print(f"\nJudging {len(model_answers)} answers...")
        
        results = {}
        
        for model_index, (model_name, answer) in enumerate(model_answers.items(), 1):
            result = ModelEvaluationResult(model_name)
            result.set_answer(answer)
            
            # Get short name for display
            short_name = model_name.split('/')[-1][:20]
            print(f"  [{model_index}/{len(model_answers)}] Judging {short_name}...", 
                  end=" ", flush=True)
            
            start_time = time.time()
            judge_scores = self.judge_client.grade_answer(
                question.text,
                question.reference_answer,
                answer
            )
            elapsed_time = time.time() - start_time
            
            result.set_judge_scores(judge_scores)
            
            if result.error:
                print(f"✗ (error)")
            else:
                raw_score = judge_scores.get("overall_score", "?")
                print(f"✓ (score: {raw_score}/5.0, {elapsed_time:.1f}s)")
            
            results[model_name] = result
        
        return results
    
    def _build_result_row(
        self,
        question: Question,
        results: Dict[str, ModelEvaluationResult]
    ) -> Dict:
        """Build result row for DataFrame."""
        row = {
            self.question_column: question.text,
            self.reference_column: question.reference_answer,
        }
        
        # Add all model results
        for model_result in results.values():
            row.update(model_result.to_dict())
        
        return row
