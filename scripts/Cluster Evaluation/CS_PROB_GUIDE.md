# CS-PROB: Comprehensive Usage Guide

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Input File Format](#input-file-format)
6. [Running the Program](#running-the-program)
7. [Output Files](#output-files)
8. [Understanding Results](#understanding-results)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Overview

**CS-PROB** is a Python program designed to evaluate Large Language Models (LLMs) on university-level computer science exam questions. It:

1. **Reads** questions from an Excel workbook with multiple sheets (domains)
2. **Queries** 4 local LLMs under test with each question
3. **Judges** each LLM's answer using a state-of-the-art Judge LLM (Gemini)
4. **Scores** answers on a 0-1 scale based on correctness, reasoning, clarity, and conciseness
5. **Saves** results back to Excel with detailed metrics

### Key Features
- ✅ Multi-sheet Excel support (one sheet per domain)
- ✅ Embedded image extraction from Excel cells
- ✅ Base64 image encoding for multimodal questions
- ✅ API key rotation (5 keys) to avoid rate limits
- ✅ Retry logic for network failures
- ✅ Detailed progress tracking
- ✅ Per-sheet and consolidated outputs

---

## System Requirements

### Hardware
- **Network**: Access to local LLM servers on `192.168.0.x` network
- **Memory**: At least 4GB RAM recommended
- **Storage**: Sufficient space for Excel files and outputs

### Software
- **Python**: Version 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Network**: Stable connection to local LLM endpoints and internet (for Gemini API)

---

## Installation

### Step 1: Install Python Dependencies

Run the following command to install required packages:

```bash
pip install pandas requests openpyxl google-generativeai
```

Or create a `requirements.txt` file:

```txt
pandas>=2.0.0
requests>=2.28.0
openpyxl>=3.1.0
google-generativeai>=0.3.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python3 -c "import pandas, requests, openpyxl, google.generativeai; print('All dependencies installed!')"
```

---

## Configuration

### 1. Configure Judge API Keys

Open `cs_prob.py` in a text editor and find this section near the top:

```python
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
```

**Replace** the placeholder strings with your actual Google AI Studio API keys:

```python
JUDGE_API_KEYS = [
    "AIzaSyAaBbCcDdEeFf1122334455",
    "AIzaSyBbCcDdEeFfGg2233445566",
    "AIzaSyCcDdEeFfGgHh3344556677",
    "AIzaSyDdEeFfGgHhIi4455667788",
    "AIzaSyEeFfGgHhIiJj5566778899",
]
```

### 2. Obtain Google AI Studio API Keys

If you don't have API keys yet:

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it into `cs_prob.py`
5. Repeat for all 5 project members

### 3. Optional Configuration

You can also adjust these settings in `cs_prob.py`:

```python
SYSTEM_PROMPT = "You are a helpful assistant."  # Prompt for LLMs under test
TEMPERATURE = 0.0  # Sampling temperature (0 = deterministic)
TIMEOUT = 300  # HTTP timeout in seconds
```

---

## Input File Format

### Excel Workbook Structure

Your input Excel file should have:
- **Multiple sheets**: Each sheet represents a domain (e.g., "Algorithms", "Networks", "Databases")
- **Required columns** in each sheet:
  - `Question`: The exam question text
  - `ReferenceAnswer` (or `Reference`): The authoritative correct answer
- **Optional columns**:
  - `ImagePath`: File path to an image (if not embedded)
  - Any other metadata columns you want to preserve

### Example Sheet Structure

| Question | ReferenceAnswer | ImagePath |
|----------|----------------|-----------|
| What is the time complexity of binary search? | O(log n) | |
| Explain TCP handshake | Three-way handshake: SYN, SYN-ACK, ACK | diagrams/tcp.png |

### Embedded Images

**CS-PROB automatically extracts images embedded in Excel cells.**

To embed an image:
1. In Excel, click on the cell in the `Question` column
2. Insert → Picture
3. The image will be anchored to that cell
4. CS-PROB will detect and base64-encode it automatically

---

## Running the Program

### Basic Usage

```bash
python3 cs_prob.py --input_xlsx exams.xlsx
```

This will:
- Read `exams.xlsx`
- Process all sheets
- Save results to `cs_prob_results.xlsx` in the current directory

### Specify Output File

```bash
python3 cs_prob.py --input_xlsx data/exams.xlsx --output_xlsx results.xlsx
```

### Custom Column Names

If your Excel uses different column names:

```bash
python3 cs_prob.py --input_xlsx exams.xlsx \
  --question_col "Problem" \
  --reference_col "Answer"
```

### Adjust Timeout

For slower networks or larger questions:

```bash
python3 cs_prob.py --input_xlsx exams.xlsx --timeout 600
```

### Full Example

```bash
python3 cs_prob.py \
  --input_xlsx /home/llm-nieb/exam_questions.xlsx \
  --output_xlsx /home/llm-nieb/evaluation_results.xlsx \
  --timeout 300 \
  --question_col "Question" \
  --reference_col "ReferenceAnswer"
```

---

## Output Files

The program generates **three types** of output files:

### 1. Per-Sheet Excel Files

- **Location**: Current directory
- **Naming**: `{SheetName}_results.xlsx`
- **Example**: `Algorithms_results.xlsx`, `Networks_results.xlsx`
- **Content**: All questions from that sheet with model scores

### 2. Aggregated CSV

- **Location**: Current directory
- **Filename**: `aggregated_results.csv`
- **Content**: All sheets combined with an additional `__sheet` column
- **Use case**: Easy analysis in pandas, R, or Excel pivot tables

### 3. Consolidated Workbook

- **Location**: As specified by `--output_xlsx`
- **Default**: `cs_prob_results.xlsx`
- **Content**: Multi-sheet workbook matching input structure, with added columns
- **Use case**: Primary output for sharing with team

---

## Understanding Results

### Output Columns

Each model gets multiple columns in the output:

#### Main Score Column
- **Column name**: Exact model name (e.g., `Qwen/Qwen2.5-72B-Instruct`)
- **Values**: 0.0 to 1.0 (rescaled from 0-5)
- **Meaning**: Overall quality score

#### Detailed Rubric Columns
For each model `MODEL_NAME`, you'll see:

| Column | Description | Range |
|--------|-------------|-------|
| `MODEL_NAME__answer` | Full text response from the LLM | Text |
| `MODEL_NAME__correctness` | Technical accuracy | 0-5 |
| `MODEL_NAME__reasoning_quality` | Logical soundness | 0-5 |
| `MODEL_NAME__clarity` | Structure and readability | 0-5 |
| `MODEL_NAME__conciseness` | Brevity without missing points | 0-5 |
| `MODEL_NAME__overall_score_raw` | Raw score before rescaling | 0-5 |
| `MODEL_NAME__judge_raw` | Judge error/parse messages (if any) | Text |

### Example Output Row

| Question | ReferenceAnswer | Qwen/Qwen2.5-72B-Instruct | Qwen/Qwen2.5-72B-Instruct__correctness | ... |
|----------|----------------|---------------------------|----------------------------------------|-----|
| What is binary search complexity? | O(log n) | 0.9 | 5 | ... |

### Interpreting Scores

- **0.0 - 0.2**: Poor answer, fundamental errors
- **0.2 - 0.4**: Weak answer, significant gaps
- **0.4 - 0.6**: Acceptable, some correct elements
- **0.6 - 0.8**: Good answer, mostly correct
- **0.8 - 1.0**: Excellent, comprehensive and accurate

---

## Troubleshooting

### Problem: "Import errors" when running

**Solution**: Install missing dependencies

```bash
pip install pandas requests openpyxl google-generativeai
```

### Problem: "Please configure JUDGE_API_KEYS"

**Solution**: Edit `cs_prob.py` and replace placeholder keys with real Google AI Studio keys

### Problem: "Input file not found"

**Solution**: Check file path and ensure the Excel file exists

```bash
ls -lh exams.xlsx
python3 cs_prob.py --input_xlsx ./exams.xlsx
```

### Problem: Timeout errors with local LLMs

**Causes**:
- Server overloaded
- Network issues
- Question too complex

**Solutions**:
1. Increase timeout: `--timeout 600`
2. Check server status: `curl http://192.168.0.218:80/health` (if available)
3. Retry later when server is less busy

### Problem: Judge LLM rate limit errors

**Solution**: The program already rotates 5 keys. If still hitting limits:
- Add more API keys to `JUDGE_API_KEYS`
- Add a small delay between questions (edit code to add `time.sleep(1)` in process_sheet)

### Problem: Images not detected

**Possible causes**:
1. Images are not anchored to cells (floating images)
2. Images are in a separate column

**Solutions**:
1. Re-embed images directly in the Question cell
2. Use `ImagePath` column with file paths instead
3. Check console output for "Found X embedded image(s)" message

### Problem: Column not found error

**Error**: `Reference column 'ReferenceAnswer' not found`

**Solution**: Use `--reference_col` to specify your column name:

```bash
python3 cs_prob.py --input_xlsx exams.xlsx --reference_col "Answer"
```

### Problem: Empty or error responses from models

**Causes**:
- Model server down
- Network connectivity issues
- Invalid prompt format

**Debugging**:
1. Test individual model with `simple_prompt.py`:
   ```bash
   python3 simple_prompt.py "Test question" --api_url http://192.168.0.218:80/v1/chat/completions
   ```
2. Check console output for specific error messages
3. Verify server endpoints are correct in `ENDPOINTS_BY_MODEL`

---

## Advanced Usage

### Processing a Single Sheet

If you only want to process one domain, create a temporary Excel file with just that sheet.

### Batch Processing Multiple Files

Create a bash script:

```bash
#!/bin/bash
for file in data/*.xlsx; do
    echo "Processing $file..."
    python3 cs_prob.py --input_xlsx "$file" --output_xlsx "results/$(basename $file)"
done
```

### Analyzing Results

Using pandas in Python:

```python
import pandas as pd

# Load aggregated results
df = pd.read_csv('aggregated_results.csv')

# Calculate average scores per model
models = [
    'CohereLabs/aya-expanse-32b',
    'openai/gpt-oss-120b',
    'meta-llama/Llama-3.3-70B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct'
]

for model in models:
    avg_score = df[model].mean()
    print(f"{model}: {avg_score:.3f}")

# Per-sheet averages
sheet_scores = df.groupby('__sheet')[models].mean()
print(sheet_scores)
```

### Customizing Prompts

To modify the prompt sent to LLMs under test, edit the `build_model_prompt()` function in `cs_prob.py`.

To modify the judge rubric, edit the `call_judge_llm()` function.

### Adding More Models

Edit `ENDPOINTS_BY_MODEL` in `cs_prob.py`:

```python
ENDPOINTS_BY_MODEL = {
    "CohereLabs/aya-expanse-32b": "http://192.168.0.219:80/v1/chat/completions",
    "openai/gpt-oss-120b": "http://192.168.0.205:80/v1/chat/completions",
    "meta-llama/Llama-3.3-70B-Instruct": "http://192.168.0.217:80/v1/chat/completions",
    "Qwen/Qwen2.5-72B-Instruct": "http://192.168.0.218:80/v1/chat/completions",
    "YourModel/ModelName": "http://192.168.0.XXX:80/v1/chat/completions",  # Add here
}
```

---

## Performance Considerations

### Expected Runtime

For a typical workbook:
- **Per question**: ~30-60 seconds (4 models × ~5s each + 4 judge calls × ~3s each)
- **Per sheet** (20 questions): ~10-20 minutes
- **Full workbook** (5 sheets, 100 questions): ~50-100 minutes

### Optimization Tips

1. **Parallel processing**: Current version is sequential. For faster processing, modify code to use `concurrent.futures` for parallel model calls
2. **Caching**: If re-running with same questions, cache model responses
3. **Selective processing**: Process only failed questions on re-runs

---

## Support and Contact

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review console output for specific error messages
3. Contact your project team members
4. Check Google AI Studio status: https://status.cloud.google.com/

---

## Appendix: Model Details

### Models Under Test

| Model | Endpoint | Parameters |
|-------|----------|------------|
| Aya-Expanse-32B | http://192.168.0.219:80/v1/chat/completions | 32B |
| GPT-OSS-120B | http://192.168.0.205:80/v1/chat/completions | 120B |
| Llama-3.3-70B | http://192.168.0.217:80/v1/chat/completions | 70B |
| Qwen2.5-72B | http://192.168.0.218:80/v1/chat/completions | 72B |

### Judge Model

- **Model**: Gemini 1.5 Pro
- **Provider**: Google AI Studio
- **Access**: Via `google-generativeai` Python library
- **Rate limits**: Handled via 5-key rotation

---

## License

This tool is for academic research purposes. Ensure compliance with:
- Google AI Studio Terms of Service
- Your institution's research ethics guidelines
- Data privacy regulations for exam content
