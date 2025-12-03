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

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env')))
HF_TOKEN = os.getenv('HF_TOKEN')

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed_csprob.csv'))
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/results.csv'))
MODEL_CONFIGS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",  # Temporarily disabled due to access restriction
    # "mistral-7b": "mistralai/Mistral-7B-v0.1",
    # "qwen-7b": "Qwen/Qwen-7B"
}

JUDGE_MODELS = {
    "qwen-judge": "Qwen/Qwen2.5-72B-Instruct"
}

# General resource optimization settings
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Adjust based on VRAM, can be tuned
torch.set_float32_matmul_precision('high')

def load_model(model_id, device="cuda"):
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
    print(f"Generating response for question: {question[:60]}...")
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    # Free up memory after generation
    gc.collect()
    torch.cuda.empty_cache()
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Model response generated.")
    return response

def judge_response(judge_model, judge_tokenizer, question, answer, response, device="cuda"):
    # Prompt for multi-criteria scoring on a 0-1 scale
    prompt = (
        f"You are an expert judge. Evaluate the model's response based on the following criteria:\n"
        f"1. Correctness: Is the answer factually correct?\n"
        f"2. Completeness: Does it fully address the question?\n"
        f"3. Relevancy: Is the response relevant to the question?\n"
        f"4. Clarity: Is the explanation clear and well-structured?\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {answer}\n"
        f"Model Response: {response}\n\n"
        f"Provide scores for each criterion (0-1 scale) in the format:\n"
        f"Correctness: [score]\n"
        f"Completeness: [score]\n"
        f"Relevancy: [score]\n"
        f"Clarity: [score]\n"
        f"Overall Score: [average of all scores]"
    )
    print(f"Judging model response for question: {question[:60]}...")
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=128)
    judge_output = judge_tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Judge output: {judge_output}")
    
    # Extract scores from judge output
    import re
    scores = {}
    for criterion in ['Correctness', 'Completeness', 'Relevancy', 'Clarity', 'Overall Score']:
        pattern = rf"{criterion}:\s*([01](?:\.\d+)?|0?\.\d+)"
        match = re.search(pattern, judge_output, re.IGNORECASE)
        scores[criterion.lower().replace(' ', '_')] = float(match.group(1)) if match else None
    
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run a small batch test (first 2 rows)')
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    results = []
    # Load models
    models = {name: load_model(mid, TORCH_DEVICE) for name, mid in MODEL_CONFIGS.items()}
    judge_models = {name: load_model(path, TORCH_DEVICE) for name, path in JUDGE_MODELS.items()}

    if args.test:
        df = df.head(2)
        print('Running test mode: evaluating first 2 rows only.')

    # Batched inference for efficiency
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start+BATCH_SIZE]
        print(f"Processing batch {start//BATCH_SIZE + 1} / {((len(df)-1)//BATCH_SIZE)+1}")
        for idx, row in batch.iterrows():
            question = row['Question']
            answer = row['Answer']
            for mname, (model, tokenizer) in models.items():
                print(f"\n---\nEvaluating model: {mname}, question idx: {idx}")
                response = generate_response(model, tokenizer, question, TORCH_DEVICE)
                for jname, (jmodel, jtokenizer) in judge_models.items():
                    print(f"Judging with: {jname}")
                    scores = judge_response(jmodel, jtokenizer, question, answer, response, TORCH_DEVICE)
                    results.append({
                        'idx': idx,
                        'model': mname,
                        'judge': jname,
                        'question': question,
                        'reference_answer': answer,
                        'model_response': response,
                        'correctness': scores.get('correctness'),
                        'completeness': scores.get('completeness'),
                        'relevancy': scores.get('relevancy'),
                        'clarity': scores.get('clarity'),
                        'overall_score': scores.get('overall_score')
                    })
        # Free up memory after each batch
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"Evaluation results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
