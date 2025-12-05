#!/usr/bin/env python3
"""
CS-PROB: Evaluate multiple LLMs on university exam questions and grade with a Judge LLM.

This program:
1. Reads an Excel workbook with multiple sheets (one per domain)
2. For each question, queries 4 local LLMs under test
3. Sends LLM answers + reference answers to a Judge LLM (Gemini)
4. Rescales scores to 0.0-1.0 and saves results back to Excel

Features:
- Embedded image extraction from Excel cells (base64 encoding)
- API key rotation for Judge LLM (5 keys to avoid rate limits)
- Progress tracking with detailed console output
- Error handling with retry logic
- Per-sheet and consolidated outputs
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet
import google.generativeai as genai

# ------------------------
# CONFIGURATION - Paste your 5 Google AI Studio API keys here
# ------------------------
JUDGE_API_KEYS = [
    "PASTE_YOUR_KEY_1_HERE",
    "PASTE_YOUR_KEY_2_HERE",
    "PASTE_YOUR_KEY_3_HERE",
    "PASTE_YOUR_KEY_4_HERE",
    "PASTE_YOUR_KEY_5_HERE",
]

SYSTEM_PROMPT = "You are a helpful assistant."
TEMPERATURE = 0.0
TIMEOUT = 300  # seconds

# Model endpoints (4 models under test)
ENDPOINTS_BY_MODEL = {
    "CohereLabs/aya-expanse-32b": "http://192.168.0.219:80/v1/chat/completions",
    "openai/gpt-oss-120b": "http://192.168.0.205:80/v1/chat/completions",
    "meta-llama/Llama-3.3-70B-Instruct": "http://192.168.0.217:80/v1/chat/completions",
    "Qwen/Qwen2.5-72B-Instruct": "http://192.168.0.218:80/v1/chat/completions",
}

# ------------------------
# Utilities
# ------------------------

def read_workbook_with_images(path: Path, question_col_name: str) -> Dict[str, pd.DataFrame]:
    """
    Read all sheets from an Excel workbook into DataFrames and attempt to extract
    base64 images anchored to cells in the question column.

    Returns a dict: sheet_name -> DataFrame with added 'ImageBase64' column
    """
    print(f"\n[INFO] Reading workbook: {path}")
    sheets_pd = pd.read_excel(path, sheet_name=None, engine='openpyxl')
    print(f"[INFO] Found {len(sheets_pd)} sheet(s): {list(sheets_pd.keys())}")
    
    wb = load_workbook(filename=str(path), data_only=True)

    result: Dict[str, pd.DataFrame] = {}

    for sheet_name, df in sheets_pd.items():
        print(f"\n[INFO] Processing sheet '{sheet_name}' ({len(df)} rows)")
        ws: Worksheet = wb[sheet_name]

        # Build a mapping of (row, col) -> base64 image if anchored there
        img_map: Dict[Tuple[int, int], str] = {}
        img_count = 0
        try:
            # openpyxl stores images with anchors; iterate drawings
            for img in ws._images:  # type: ignore[attr-defined]
                anchor = getattr(img, 'anchor', None)
                if anchor:
                    # Handle different anchor types (OneCellAnchor, TwoCellAnchor)
                    if hasattr(anchor, '_from'):
                        # TwoCellAnchor
                        row_idx = anchor._from.row + 1  # openpyxl is 0-indexed
                        col_idx = anchor._from.col + 1
                    elif hasattr(anchor, 'row') and hasattr(anchor, 'col'):
                        row_idx = anchor.row + 1
                        col_idx = anchor.col + 1
                    else:
                        continue
                    
                    img_bytes = img._data() if hasattr(img, '_data') else None
                    if img_bytes:
                        img_map[(row_idx, col_idx)] = base64.b64encode(img_bytes).decode('utf-8')
                        img_count += 1
        except Exception as e:
            print(f"[WARN] Could not extract images from sheet '{sheet_name}': {e}")
            img_map = {}

        print(f"[INFO] Found {img_count} embedded image(s) in sheet '{sheet_name}'")

        # Identify the question column index in this sheet
        if question_col_name in df.columns:
            q_col_idx = df.columns.get_loc(question_col_name) + 1  # 1-based for openpyxl
        else:
            print(f"[WARN] Column '{question_col_name}' not found in sheet '{sheet_name}'")
            q_col_idx = None

        image_b64_list: List[Optional[str]] = []
        if q_col_idx is not None:
            # Iterate DataFrame rows and fetch image for that (row+header_offset)
            # Assumes header is on first row; DataFrame index aligns with Excel data rows starting at 2.
            for i in range(len(df)):
                excel_row = i + 2  # header at row 1
                b64 = img_map.get((excel_row, q_col_idx))
                image_b64_list.append(b64)
            df = df.copy()
            df["ImageBase64"] = image_b64_list
            embedded_count = sum(1 for x in image_b64_list if x is not None)
            print(f"[INFO] Mapped {embedded_count} image(s) to question rows")
        
        result[sheet_name] = df

    return result


def read_image_base64(image_path: Optional[str]) -> Optional[str]:
    """Return base64 string of image file, or None if not present."""
    if not image_path:
        return None
    p = Path(str(image_path)).expanduser()
    if not p.exists() or not p.is_file():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ------------------------
# LLM under test caller
# ------------------------

def call_local_llm(api_url: str, model: str, prompt: str, system_prompt: str, temperature: float = 0.0, timeout: int = 300, max_retries: int = 2) -> str:
    """Call local LLM endpoint with retry logic; return text response."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("choices"):
                choice = data["choices"][0]
                msg = None
                if isinstance(choice, dict):
                    if "message" in choice and isinstance(choice["message"], dict):
                        msg = choice["message"].get("content")
                    if msg is None:
                        msg = choice.get("text")
                if msg is not None:
                    return str(msg).strip()
            return json.dumps(data)
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"[WARN] Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2)
            else:
                return f"[ERROR] Timeout after {max_retries + 1} attempts"
        except Exception as e:
            if attempt < max_retries:
                print(f"[WARN] Error on attempt {attempt + 1}: {e}, retrying...")
                time.sleep(2)
            else:
                return f"[ERROR] {type(e).__name__}: {e}"
    
    return "[ERROR] Unexpected retry loop exit"


# ------------------------
# Judge LLM (Gemini) caller with API key rotation
# ------------------------

def rotate_keys(keys: List[str]):
    """Generator that yields keys in round-robin fashion."""
    idx = 0
    while True:
        yield keys[idx % len(keys)]
        idx += 1


def call_judge_llm(api_key: str, question: str, reference_answer: str, model_answer: str, timeout: int = 300, max_retries: int = 2) -> Dict:
    """
    Call the Judge LLM (Gemini via google.generativeai) with the rubric; return parsed JSON scores.
    """
    judge_prompt = f"""Question:
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
}}""".strip()

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-pro")
            resp = model.generate_content(judge_prompt)
            text = resp.text if hasattr(resp, 'text') else None
            if not text:
                # Fallback to candidates
                if hasattr(resp, 'candidates') and resp.candidates:
                    for c in resp.candidates:
                        parts = getattr(getattr(c, 'content', None), 'parts', [])
                        if parts:
                            collected = []
                            for p in parts:
                                t = getattr(p, 'text', None)
                                if t:
                                    collected.append(t)
                            if collected:
                                text = "\n".join(collected)
                                break
            if not text:
                return {"error": "Empty response from judge LLM"}

            # Parse JSON from response
            try:
                return json.loads(text)
            except Exception:
                # Try to extract JSON block
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(text[start:end+1])
                    except Exception:
                        pass
                # If still can't parse, return error
                if attempt < max_retries:
                    print(f"[WARN] Failed to parse judge response on attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    continue
                return {"parse_error": True, "raw": text}
        except Exception as e:
            if attempt < max_retries:
                print(f"[WARN] Judge LLM error on attempt {attempt + 1}: {e}, retrying...")
                time.sleep(2)
            else:
                return {"error": f"{type(e).__name__}: {e}"}
    
    return {"error": "Unexpected retry loop exit"}


# ------------------------
# Orchestration
# ------------------------

def build_model_prompt(question: str, b64_img: Optional[str]) -> str:
    return (
        "Question: " + question.strip() + "\n" +
        "Figure (for multimodal-format problem): " + (b64_img if b64_img else "NULL") + "\n" +
        "Prompt:\n" +
        "Answer the given question step by step. Begin by explaining your reasoning process clearly(avoid long essays). Think step by step before answering"
    )


def process_sheet(sheet_name: str, df: pd.DataFrame, endpoints_by_model: Dict[str, str], system_prompt: str,
                  judge_keys: List[str], timeout: int = 300, temperature: float = 0.0,
                  question_col: str = "Question", ref_col: str = "ReferenceAnswer") -> pd.DataFrame:
    """
    Process one sheet: query models and judge answers. Returns DataFrame with appended columns.
    """
    # Normalize reference column name
    if ref_col not in df.columns:
        alt = "Reference" if "Reference" in df.columns else None
        if alt:
            ref_col = alt
        else:
            raise ValueError(f"Reference column '{ref_col}' not found in sheet '{sheet_name}'. Available: {list(df.columns)}")

    out_rows = []
    key_cycle = rotate_keys(judge_keys)
    
    total_questions = len(df)
    print(f"\n{'='*80}")
    print(f"PROCESSING SHEET: {sheet_name}")
    print(f"Total questions: {total_questions}")
    print(f"Models under test: {len(endpoints_by_model)}")
    print(f"{'='*80}\n")

    for idx, row in df.iterrows():
        question_num = idx + 1
        print(f"\n--- Question {question_num}/{total_questions} ---")
        
        question = str(row.get(question_col, "")).strip()
        reference_answer = str(row.get(ref_col, "")).strip()
        
        # Display question preview
        q_preview = question[:100] + "..." if len(question) > 100 else question
        print(f"Q: {q_preview}")
        
        # Prefer embedded base64 from workbook if present; else try file path column if exists
        b64_img = row.get("ImageBase64")
        if not b64_img and "ImagePath" in df.columns:
            b64_img = read_image_base64(row.get("ImagePath"))
        
        has_image = b64_img is not None
        print(f"Image: {'Yes' if has_image else 'No'}")

        prompt = build_model_prompt(question, b64_img)

        # Query all models
        model_answers = {}
        print(f"\nQuerying {len(endpoints_by_model)} models...")
        for model_idx, (model, endpoint) in enumerate(endpoints_by_model.items(), 1):
            model_short = model.split('/')[-1][:20]  # Shorter name for display
            print(f"  [{model_idx}/{len(endpoints_by_model)}] {model_short}...", end=" ", flush=True)
            start_time = time.time()
            ans = call_local_llm(endpoint, model, prompt, system_prompt, temperature=temperature, timeout=timeout)
            elapsed = time.time() - start_time
            
            # Check if error
            if ans.startswith("[ERROR]"):
                print(f"✗ (error: {ans[:50]})")
            else:
                ans_preview = ans[:50] + "..." if len(ans) > 50 else ans
                print(f"✓ ({elapsed:.1f}s)")
            
            model_answers[model] = ans

        # Judge each model's answer
        judge_results = {}
        print(f"\nJudging {len(model_answers)} answers...")
        for model_idx, (model, m_ans) in enumerate(model_answers.items(), 1):
            model_short = model.split('/')[-1][:20]
            print(f"  [{model_idx}/{len(model_answers)}] Judging {model_short}...", end=" ", flush=True)
            
            api_key = next(key_cycle)
            start_time = time.time()
            jr = call_judge_llm(api_key, question, reference_answer, m_ans, timeout=timeout)
            elapsed = time.time() - start_time
            
            if "error" in jr or jr.get("parse_error"):
                print(f"✗ (error)")
            else:
                score = jr.get("overall_score", "?")
                print(f"✓ (score: {score}/5.0, {elapsed:.1f}s)")
            
            judge_results[model] = jr

        # Flatten judge results into columns - only final scores
        row_out = {
            question_col: question,
            ref_col: reference_answer,
        }
        for model in endpoints_by_model.keys():
            jr = judge_results.get(model, {})
            
            # Rescale overall score to 0.0-1.0
            ov = jr.get("overall_score")
            scaled = None
            try:
                if ov is not None:
                    scaled = float(ov) / 5.0
            except Exception:
                scaled = None
            
            # Only save the main score column (rescaled 0-1)
            row_out[f"{model}"] = scaled
        
        out_rows.append(row_out)

    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(
        description="CS-PROB: Evaluate LLMs on university CS exam questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 cs_prob.py --input_xlsx exams.xlsx
  python3 cs_prob.py --input_xlsx data/questions.xlsx --output_xlsx results.xlsx
  python3 cs_prob.py --input_xlsx exams.xlsx --question_col "Problem" --reference_col "Answer"

Note: Configure JUDGE_API_KEYS at the top of this file before running.
        """
    )
    parser.add_argument("--input_xlsx", type=str, required=True, 
                       help="Path to Excel workbook with multiple sheets")
    parser.add_argument("--output_xlsx", type=str, default="cs_prob_results.xlsx", 
                       help="Output Excel workbook with added columns (default: cs_prob_results.xlsx)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, 
                       help=f"HTTP timeout seconds (default: {TIMEOUT})")
    parser.add_argument("--question_col", type=str, default="Question", 
                       help="Column name for question text (default: Question)")
    parser.add_argument("--reference_col", type=str, default="ReferenceAnswer", 
                       help="Column name for reference answer (default: ReferenceAnswer)")

    args = parser.parse_args()

    # Validate configuration
    if not all(key and key != f"PASTE_YOUR_KEY_{i}_HERE" for i, key in enumerate(JUDGE_API_KEYS, 1)):
        print("\n" + "="*80)
        print("ERROR: Please configure JUDGE_API_KEYS at the top of cs_prob.py")
        print("="*80 + "\n")
        return

    input_path = Path(args.input_xlsx)
    if not input_path.exists():
        print(f"\n[ERROR] Input file not found: {input_path}\n")
        return

    output_dir = Path('.')
    
    print("\n" + "="*80)
    print("CS-PROB: University CS Exam LLM Evaluation")
    print("="*80)
    print(f"Input file       : {input_path}")
    print(f"Output file      : {args.output_xlsx}")
    print(f"Output directory : {output_dir.absolute()}")
    print(f"Timeout          : {args.timeout}s")
    print(f"Question column  : {args.question_col}")
    print(f"Reference column : {args.reference_col}")
    print(f"System prompt    : {SYSTEM_PROMPT}")
    print(f"Temperature      : {TEMPERATURE}")
    print(f"Judge keys       : {len(JUDGE_API_KEYS)} keys configured")
    print(f"Models under test: {len(ENDPOINTS_BY_MODEL)}")
    for model in ENDPOINTS_BY_MODEL.keys():
        print(f"  - {model}")
    print("="*80)

    # Read workbook and capture embedded images
    sheets = read_workbook_with_images(input_path, question_col_name=args.question_col)

    aggregated = []
    start_time_total = time.time()

    for sheet_name, df in sheets.items():
        sheet_start = time.time()
        res_df = process_sheet(
            sheet_name=sheet_name,
            df=df,
            endpoints_by_model=ENDPOINTS_BY_MODEL,
            system_prompt=SYSTEM_PROMPT,
            judge_keys=JUDGE_API_KEYS,
            timeout=args.timeout,
            temperature=TEMPERATURE,
            question_col=args.question_col,
            ref_col=args.reference_col,
        )
        sheet_elapsed = time.time() - sheet_start
        
        # Save per-sheet result
        out_path = output_dir / f"{sheet_name}_results.xlsx"
        res_df.to_excel(out_path, index=False)
        print(f"\n[✓] Saved sheet results: {out_path}")
        print(f"    Sheet processing time: {sheet_elapsed:.1f}s")
        
        aggregated.append(res_df.assign(__sheet=sheet_name))

    # Save aggregated CSV
    if aggregated:
        agg_df = pd.concat(aggregated, ignore_index=True)
        agg_csv = output_dir / "aggregated_results.csv"
        agg_df.to_csv(agg_csv, index=False)
        print(f"\n[✓] Saved aggregated CSV: {agg_csv}")

        # Also write back to a single Excel workbook with sheets
        with pd.ExcelWriter(Path(args.output_xlsx), engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                # Match the processed output DataFrame by sheet
                processed_df = [d for d in aggregated if len(d) and d.iloc[0].get('__sheet') == sheet_name]
                if processed_df:
                    final_df = processed_df[0].drop(columns=['__sheet'])
                else:
                    final_df = df
                final_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[✓] Saved consolidated workbook: {args.output_xlsx}")
    
    total_elapsed = time.time() - start_time_total
    print(f"\n{'='*80}")
    print(f"COMPLETED")
    print(f"Total processing time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
