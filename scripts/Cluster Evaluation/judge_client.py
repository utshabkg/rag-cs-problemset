"""
Judge LLM client module for CS-PROB.
Handles communication with Gemini for grading answers.
"""
import json
import time
from typing import Dict, List, Iterator

import google.generativeai as genai

from config import JUDGE_MODEL_NAME, REQUEST_TIMEOUT, MAX_RETRIES, JUDGE_PROMPT_TEMPLATE


class JudgeClient:
    """Client for querying Gemini judge model with API key rotation."""
    
    def __init__(
        self,
        api_keys: List[str],
        model_name: str = JUDGE_MODEL_NAME,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES
    ):
        """
        Initialize judge client.
        
        Args:
            api_keys: List of Google AI Studio API keys for rotation
            model_name: Gemini model name
            timeout: Request timeout (not directly used by SDK, kept for consistency)
            max_retries: Maximum retry attempts
        """
        self.api_keys = api_keys
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.key_rotator = self._create_key_rotator()
    
    def _create_key_rotator(self) -> Iterator[str]:
        """Create round-robin iterator for API keys."""
        index = 0
        while True:
            yield self.api_keys[index % len(self.api_keys)]
            index += 1
    
    def grade_answer(
        self,
        question: str,
        reference_answer: str,
        model_answer: str
    ) -> Dict:
        """
        Grade a model's answer against reference.
        
        Args:
            question: The exam question
            reference_answer: Authoritative correct answer
            model_answer: Answer from model under test
            
        Returns:
            Dictionary with rubric scores or error information
        """
        judge_prompt = self._build_judge_prompt(
            question,
            reference_answer,
            model_answer
        )
        
        api_key = next(self.key_rotator)
        
        for attempt_number in range(self.max_retries + 1):
            try:
                response_text = self._query_gemini(api_key, judge_prompt)
                return self._parse_json_response(response_text, attempt_number)
                
            except Exception as error:
                if attempt_number < self.max_retries:
                    print(f"[WARN] Judge error on attempt {attempt_number + 1}: {error}, retrying...")
                    time.sleep(2)
                else:
                    return {"error": f"{type(error).__name__}: {error}"}
        
        return {"error": "Unexpected retry loop exit"}
    
    def _build_judge_prompt(
        self,
        question: str,
        reference_answer: str,
        model_answer: str
    ) -> str:
        """Build grading prompt from template."""
        return JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer
        )
    
    def _query_gemini(self, api_key: str, prompt: str) -> str:
        """
        Query Gemini API and return response text.
        
        Args:
            api_key: Google AI Studio API key
            prompt: Grading prompt
            
        Returns:
            Response text from Gemini
        """
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        
        # Try direct text attribute
        if hasattr(response, 'text') and response.text:
            return response.text
        
        # Fallback to candidates
        if hasattr(response, 'candidates') and response.candidates:
            text_parts = []
            for candidate in response.candidates:
                content = getattr(candidate, 'content', None)
                if content:
                    parts = getattr(content, 'parts', [])
                    for part in parts:
                        part_text = getattr(part, 'text', None)
                        if part_text:
                            text_parts.append(part_text)
            
            if text_parts:
                return "\n".join(text_parts)
        
        raise ValueError("Empty response from Gemini")
    
    def _parse_json_response(self, response_text: str, attempt_number: int) -> Dict:
        """
        Parse JSON from Gemini response.
        
        Args:
            response_text: Raw response text
            attempt_number: Current attempt number (for retry logic)
            
        Returns:
            Parsed JSON dictionary or error dictionary
        """
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON block
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                return json.loads(response_text[json_start:json_end + 1])
            except json.JSONDecodeError:
                pass
        
        # If parsing failed and we have retries left, raise exception
        if attempt_number < self.max_retries:
            print(f"[WARN] Failed to parse judge response on attempt {attempt_number + 1}, retrying...")
            time.sleep(2)
            raise ValueError("JSON parsing failed")
        
        # Last attempt failed, return error
        return {"parse_error": True, "raw": response_text}


def rescale_score(raw_score: float, max_score: float = 5.0) -> float:
    """
    Rescale score from 0-max_score to 0-1 range.
    
    Args:
        raw_score: Score in original range
        max_score: Maximum possible score
        
    Returns:
        Rescaled score between 0 and 1
    """
    try:
        return float(raw_score) / max_score
    except (TypeError, ValueError, ZeroDivisionError):
        return None
