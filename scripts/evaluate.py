"""
Evaluation pipeline for automatic judging of model responses.
- Loads processed dataset (data/processed_csprob.csv)
- Runs each question through selected models (Llama-3-8B, Mistral-7B, Qwen-7B)
- Uses judge models (Qwen, Llama from /media/12TB/shared/datasets/) to verify responses
- Saves results to evaluation/results.csv
"""


import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import infer_auto_device_map, dispatch_model
import gc
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from datetime import datetime

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env')))
HF_TOKEN = os.getenv('HF_TOKEN')

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed_csprob.csv'))
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/results.csv'))
LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/evaluation.log'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
MODEL_CONFIGS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    # "qwen-7b": "Qwen/Qwen-7B"
}

JUDGE_MODELS = {
    "qwen-judge": "Qwen/Qwen2.5-72B-Instruct"
}

# General resource optimization settings
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Safe with your hardware - can go up to 32 if needed
torch.set_float32_matmul_precision('high')

def load_model(model_id, device="cuda", quantization_bits=8):
    """
    Load model with specified quantization.
    Args:
        quantization_bits: 4, 8, or None (for FP16)
    """
    cache_dir = "/media/12TB/shared/models"
    trust_remote_code = False
    if "Qwen" in str(model_id) or "qwen" in str(model_id):
        trust_remote_code = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=HF_TOKEN,
        trust_remote_code=trust_remote_code
    )
    # Use quantization for memory efficiency
    from transformers import BitsAndBytesConfig
    if quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config
        )
    elif quantization_bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config
        )
    else:
        # No quantization - FP16 for best quality
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote_code
        )
    # Do NOT call model.to(device) when using device_map="auto"
    return model, tokenizer

def generate_response(model, tokenizer, question, device="cuda"):
    # Validate input
    if not isinstance(question, str) or not question.strip():
        logger.warning(f"Invalid question input: {type(question)} - {question}")
        return "[Error: Invalid question input]"
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a clear, professional prompt
    prompt = f"""Question: {question}

Provide a clear and concise answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # Free up memory after generation
    gc.collect()
    torch.cuda.empty_cache()
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the response to get only the answer
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    # If response becomes empty after stripping, use original (model may have reformatted)
    if not response or not response.strip():
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

def judge_response(judge_model, judge_tokenizer, question, answer, response, device="cuda"):
    # Validate inputs
    if not isinstance(question, str) or not question.strip():
        logger.warning(f"Invalid question for judging: {type(question)}")
        return 0.0
    if not isinstance(answer, str) or not answer.strip():
        logger.warning(f"Invalid answer for judging: {type(answer)}")
        return 0.0
    if not isinstance(response, str) or not response.strip():
        logger.warning(f"Invalid response for judging: {type(response)}")
        return 0.0
    
    # Set pad token if not set
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    
    # Professional prompt for continuous scoring
    prompt = f"""You are an expert evaluator. Rate the model's answer on a scale from 0.0 to 1.0 based on correctness, completeness, relevancy, and clarity.

Question: {question}

Reference Answer: {answer}

Model's Answer: {response}

Provide ONLY a numerical score between 0.0 and 1.0 (e.g., 0.75, 0.3, 0.95). Do not explain.
Score:"""
    
    inputs = judge_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        output = judge_model.generate(
            **inputs, 
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
            eos_token_id=judge_tokenizer.eos_token_id,
            temperature=0.1
        )
    judge_output = judge_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from output
    if prompt in judge_output:
        judge_output = judge_output.replace(prompt, "").strip()
    
    # Extract score - look for decimal numbers between 0 and 1
    import re
    match = re.search(r'\b(0?\.\d+|1\.0+|0\.0+|1)\b', judge_output)
    if match:
        score_str = match.group(1)
        try:
            overall_score = float(score_str)
            # Ensure score is between 0 and 1
            overall_score = max(0.0, min(1.0, overall_score))
        except ValueError:
            overall_score = 0.0
            logger.warning(f"Could not parse score: {judge_output}")
    else:
        overall_score = 0.0
        logger.warning(f"No score found in judge output: {judge_output[:100]}")
    
    logger.info(f"Score: {overall_score}")
    
    return overall_score

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run a small batch test (first 2 rows)')
    parser.add_argument('--append', action='store_true', help='Append new model columns to existing results')
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    
    # Clean dataset: remove rows with missing Question or Answer
    original_len = len(df)
    df = df.dropna(subset=['Question', 'Answer'])
    df = df[df['Question'].astype(str).str.strip() != '']
    df = df[df['Answer'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)
    
    if len(df) < original_len:
        logger.warning(f"Removed {original_len - len(df)} rows with missing Question or Answer")
    logger.info(f"Processing {len(df)} valid questions")
    
    # Load existing results if append mode
    if args.append and os.path.exists(RESULTS_PATH):
        results_df = pd.read_csv(RESULTS_PATH)
        logger.info(f"Loaded {len(results_df)} existing results. Will add new model columns.")
    else:
        # Initialize results DataFrame with base columns
        results_df = df[['Q_ID', 'Domain', 'Difficulty', 'Question', 'Answer']].copy()
        results_df.rename(columns={'Answer': 'Reference_Answer'}, inplace=True)
    
    logger.info("Loading models...")
    # Load evaluation models WITH 4-bit quantization for speed and memory savings
    models = {name: load_model(mid, TORCH_DEVICE, quantization_bits=4) for name, mid in MODEL_CONFIGS.items()}
    # Load judge models WITH 4-bit quantization for speed and memory (they're much larger)
    judge_models = {name: load_model(path, TORCH_DEVICE, quantization_bits=4) for name, path in JUDGE_MODELS.items()}
    logger.info(f"Loaded {len(models)} evaluation model(s) and {len(judge_models)} judge model(s)")

    if args.test:
        df = df.head(2)
        results_df = results_df.head(2)
        logger.info('Running test mode: evaluating first 2 rows only.')

    # Progress bar for all questions
    total_evaluations = len(df) * len(models)
    pbar = tqdm(total=total_evaluations, desc="Evaluation Progress")
    
    # Get the first judge model (assuming we use one judge for all)
    judge_name, (judge_model, judge_tokenizer) = list(judge_models.items())[0]
    
    # Evaluate each model and add columns
    for mname, (model, tokenizer) in models.items():
        # Get model display name with quantization info
        model_id = MODEL_CONFIGS[mname]
        model_short_name = model_id.split('/')[-1]  # e.g., "Meta-Llama-3-8B"
        response_col = f"{model_short_name}(4bit)_Response"
        score_col = f"{model_short_name}(4bit)_Score"
        
        # Skip if columns already exist (append mode)
        if args.append and score_col in results_df.columns:
            logger.info(f"Skipping {mname} - already evaluated")
            pbar.update(len(df))
            continue
        
        responses = []
        scores = []
        
        # Process in batches
        processed_count = 0
        for start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start:start+BATCH_SIZE]
            for idx, row in batch.iterrows():
                q_id = row['Q_ID']
                question = row['Question']
                answer = row['Answer']
                
                # Skip if question or answer is missing/invalid
                if pd.isna(question) or pd.isna(answer):
                    logger.warning(f"Skipping Q{q_id} - missing question or answer")
                    responses.append("[Error: Missing data]")
                    scores.append(0.0)
                    processed_count += 1
                    pbar.update(1)
                    continue
                
                # Convert to string if needed
                question = str(question).strip()
                answer = str(answer).strip()
                
                pbar.set_description(f"Evaluating {mname} on Q{q_id}")
                logger.info(f"Model: {mname}, Q_ID: {q_id}")
                
                # Generate response
                response = generate_response(model, tokenizer, question, TORCH_DEVICE)
                responses.append(response)
                
                # Judge response
                overall_score = judge_response(judge_model, judge_tokenizer, question, answer, response, TORCH_DEVICE)
                scores.append(overall_score)
                
                processed_count += 1
                pbar.update(1)
                # Checkpoint every 100 queries
                if processed_count % 100 == 0:
                    # Save only completed rows for this model
                    checkpoint_df = results_df.iloc[:processed_count].copy()
                    checkpoint_df[response_col] = responses[:processed_count]
                    checkpoint_df[score_col] = scores[:processed_count]
                    checkpoint_path = RESULTS_PATH.replace('.csv', f'_checkpoint_{mname}_{processed_count}.csv')
                    checkpoint_df.to_csv(checkpoint_path, index=False)
                    logger.info(f"Checkpoint saved after {processed_count} queries for {mname} at {checkpoint_path}")
            # Free up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()
        
        # Add columns to results_df
        results_df[response_col] = responses
        results_df[score_col] = scores
        
        # Save after each model to avoid data loss
        results_df.to_csv(RESULTS_PATH, index=False)
        logger.info(f"Added {mname} results and saved to {RESULTS_PATH}")
    
    pbar.close()
    logger.info(f"Evaluation complete! Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
