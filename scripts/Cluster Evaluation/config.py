"""
Configuration module for CS-PROB evaluation system.
Contains all configurable settings in one place.
"""
from typing import Dict

# ============================================================================
# JUDGE LLM CONFIGURATION
# ============================================================================
# Paste your 5 Google AI Studio API keys here
# Get keys from: https://aistudio.google.com/app/apikey
JUDGE_API_KEYS = [
    "AIzaSyAbCK5BGPDtIMBtcylBFoRxNXxlqLgPav4",
    "AIzaSyAbCK5BGPDtIMBtcylBFoRxNXxlqLgPav4",
    "AIzaSyAbCK5BGPDtIMBtcylBFoRxNXxlqLgPav4",
    "AIzaSyAbCK5BGPDtIMBtcylBFoRxNXxlqLgPav4",
    "AIzaSyAbCK5BGPDtIMBtcylBFoRxNXxlqLgPav4",
]

JUDGE_MODEL_NAME = "gemini-2.0-flash"

# ============================================================================
# MODELS UNDER TEST CONFIGURATION
# ============================================================================
MODELS_UNDER_TEST: Dict[str, str] = {
    "CohereLabs/aya-expanse-32b": "http://192.168.0.219:80/v1/chat/completions",
    "openai/gpt-oss-120b": "http://192.168.0.205:80/v1/chat/completions",
    "meta-llama/Llama-3.3-70B-Instruct": "http://192.168.0.217:80/v1/chat/completions",
    "Qwen/Qwen2.5-72B-Instruct": "http://192.168.0.218:80/v1/chat/completions",
}

# ============================================================================
# LLM PARAMETERS
# ============================================================================
SYSTEM_PROMPT = "You are a helpful assistant."
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 300  # seconds
MAX_RETRIES = 2

# ============================================================================
# EXCEL COLUMN NAMES
# ============================================================================
DEFAULT_QUESTION_COLUMN = "Question"
DEFAULT_REFERENCE_COLUMN = "Answer"
FALLBACK_REFERENCE_COLUMN = "Reference"
IMAGE_PATH_COLUMN = "ImagePath"
IMAGE_BASE64_COLUMN = "ImageBase64"

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
OUTPUT_DIRECTORY = "."  # Current directory
DEFAULT_OUTPUT_FILENAME = "cs_prob_results.xlsx"
AGGREGATED_CSV_FILENAME = "aggregated_results.csv"

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================
def get_model_prompt_template(question: str, base64_image: str = None) -> str:
    """Generate the prompt sent to models under test."""
    image_part = base64_image if base64_image else "NULL"
    return f"""Question: {question.strip()}
Figure (for multimodal-format problem): {image_part}
Prompt:
Answer the given question step by step. Begin by explaining your reasoning process clearly(avoid long essays). Think step by step before answering"""


JUDGE_PROMPT_TEMPLATE = """Question:
{question}

Reference answer (short, authoritative):
{reference_answer}

Answer under evaluation (from model under test):
{model_answer}

Task:
You are grading an advanced university-level computer science exam answer.
Compare the answer under evaluation to the question and the reference answer.

1. Focus on meaning, not wording.
   - Give full credit if the answer is logically/semantically equivalent, even if phrased differently or with a different structure.
   - Allow partial credit when some parts are correct and others are missing or wrong.

2. Evaluate using this rubric (scores are integers from 0 to 5):
   - Correctness: Is the technical content correct and complete?
   - Reasoning quality: Are the key steps or arguments sound and well-justified?
   - Clarity: Is the answer well-structured and easy to follow?
   - Conciseness: Is the answer appropriately brief (not missing key points, not needlessly verbose)?

3. Compute an Overall score from 0 to 5 (you may use halves like 3.5), mainly based on correctness and reasoning, adjusted for clarity and conciseness.

4. If the question has multiple subparts, take all of them into account when scoring.

Output your result in exactly this JSON format and nothing else:

{{
  "correctness": <integer 0-5>,
  "reasoning_quality": <integer 0-5>,
  "clarity": <integer 0-5>,
  "conciseness": <integer 0-5>,
  "overall_score": <number 0-5, can be .0 or .5>
}}"""
