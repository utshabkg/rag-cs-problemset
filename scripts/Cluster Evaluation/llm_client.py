"""
LLM client module for CS-PROB.
Handles communication with local LLM endpoints.
"""
import json
import time
from typing import Dict

import requests

from config import TEMPERATURE, REQUEST_TIMEOUT, MAX_RETRIES


class LLMClient:
    """Client for querying local LLM models under test."""
    
    def __init__(
        self, 
        model_name: str, 
        endpoint_url: str,
        system_prompt: str,
        temperature: float = TEMPERATURE,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES
    ):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the model (e.g., "Qwen/Qwen2.5-72B-Instruct")
            endpoint_url: API endpoint URL
            system_prompt: System prompt for the model
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
    
    def query(self, user_prompt: str) -> str:
        """
        Send query to LLM and return response.
        
        Args:
            user_prompt: The prompt to send to the model
            
        Returns:
            Model response text or error message
        """
        payload = self._build_request_payload(user_prompt)
        
        for attempt_number in range(self.max_retries + 1):
            try:
                response = self._send_request(payload)
                return self._extract_response_text(response)
                
            except requests.exceptions.Timeout:
                if attempt_number < self.max_retries:
                    print(f"[WARN] Timeout on attempt {attempt_number + 1}, retrying...")
                    time.sleep(2)
                else:
                    return f"[ERROR] Timeout after {self.max_retries + 1} attempts"
                    
            except Exception as error:
                if attempt_number < self.max_retries:
                    print(f"[WARN] Error on attempt {attempt_number + 1}: {error}, retrying...")
                    time.sleep(2)
                else:
                    return f"[ERROR] {type(error).__name__}: {error}"
        
        return "[ERROR] Unexpected retry loop exit"
    
    def _build_request_payload(self, user_prompt: str) -> Dict:
        """Build JSON payload for API request."""
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
    
    def _send_request(self, payload: Dict) -> Dict:
        """Send HTTP POST request to LLM endpoint."""
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _extract_response_text(self, response_data: Dict) -> str:
        """Extract text from API response."""
        if not isinstance(response_data, dict):
            return json.dumps(response_data)
        
        choices = response_data.get("choices")
        if not choices:
            return json.dumps(response_data)
        
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return json.dumps(response_data)
        
        # Try message.content first
        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content).strip()
        
        # Fallback to text field
        text = first_choice.get("text")
        if text is not None:
            return str(text).strip()
        
        return json.dumps(response_data)
    
    def get_short_name(self) -> str:
        """Get shortened version of model name for display."""
        return self.model_name.split('/')[-1][:20]
