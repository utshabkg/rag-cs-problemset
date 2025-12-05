# CS-PROB Quick Reference Card

## Installation (One-time Setup)

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually
pip install pandas requests openpyxl google-generativeai
```

## Configuration (One-time Setup)

1. Get 5 Google AI Studio API keys from: https://aistudio.google.com/app/apikey
2. Edit `cs_prob.py` line ~35:
   ```python
   JUDGE_API_KEYS = [
       "YOUR_KEY_1",
       "YOUR_KEY_2",
       "YOUR_KEY_3",
       "YOUR_KEY_4",
       "YOUR_KEY_5",
   ]
   ```

## Running CS-PROB

### Most Common Usage

```bash
python3 cs_prob.py --input_xlsx exams.xlsx
```

### All Options

```bash
python3 cs_prob.py \
  --input_xlsx PATH_TO_EXCEL \
  --output_xlsx OUTPUT_NAME.xlsx \
  --timeout 300 \
  --question_col "Question" \
  --reference_col "ReferenceAnswer"
```

## Input Excel Format

**Required columns:**
- `Question` (or custom name via `--question_col`)
- `ReferenceAnswer` or `Reference` (or custom via `--reference_col`)

**Optional:**
- `ImagePath` (for external image files)
- Embedded images directly in Question cells

## Output Files

Generated in current directory:

1. `{SheetName}_results.xlsx` - Per-sheet results
2. `aggregated_results.csv` - All sheets combined
3. `cs_prob_results.xlsx` - Consolidated workbook (or custom `--output_xlsx`)

## Output Columns

For each model (e.g., `Qwen/Qwen2.5-72B-Instruct`):

- `ModelName` - Main score (0.0-1.0) ⭐
- `ModelName__answer` - Full response text
- `ModelName__correctness` - Score 0-5
- `ModelName__reasoning_quality` - Score 0-5
- `ModelName__clarity` - Score 0-5
- `ModelName__conciseness` - Score 0-5
- `ModelName__overall_score_raw` - Raw 0-5 score

## Models Under Test

| Short Name | Full Model Name | Endpoint |
|------------|----------------|----------|
| Aya-32B | CohereLabs/aya-expanse-32b | 192.168.0.219:80 |
| GPT-120B | openai/gpt-oss-120b | 192.168.0.205:80 |
| Llama-70B | meta-llama/Llama-3.3-70B-Instruct | 192.168.0.217:80 |
| Qwen-72B | Qwen/Qwen2.5-72B-Instruct | 192.168.0.218:80 |

**Judge:** Gemini 1.5 Pro (Google AI Studio)

## Quick Tests

### Test a single model

```bash
python3 simple_prompt.py "What is binary search?" \
  --api_url http://192.168.0.218:80/v1/chat/completions \
  --model Qwen/Qwen2.5-72B-Instruct
```

### Test evaluation.py (older script)

```bash
python3 evaluation.py \
  --input_csv questions.csv \
  --api_url http://192.168.0.218:80/v1/chat/completions \
  --model Qwen/Qwen2.5-72B-Instruct
```

## Common Issues

| Problem | Solution |
|---------|----------|
| Import errors | `pip install pandas requests openpyxl google-generativeai` |
| "Configure JUDGE_API_KEYS" | Edit `cs_prob.py` and add your keys |
| File not found | Use full path: `--input_xlsx /full/path/to/file.xlsx` |
| Timeout | Increase: `--timeout 600` |
| Column not found | Specify: `--question_col "YourColumnName"` |
| Images not detected | Embed in Question cell or use `ImagePath` column |

## Expected Runtime

- **Single question:** ~30-60 seconds
- **20 questions:** ~10-20 minutes
- **100 questions:** ~50-100 minutes

## Analyzing Results

### Python (pandas)

```python
import pandas as pd

df = pd.read_csv('aggregated_results.csv')

# Average scores per model
models = ['Qwen/Qwen2.5-72B-Instruct', 'openai/gpt-oss-120b', ...]
for model in models:
    print(f"{model}: {df[model].mean():.3f}")

# Per-sheet breakdown
df.groupby('__sheet')[models].mean()
```

### Excel

Open `cs_prob_results.xlsx` and use:
- Pivot tables for aggregation
- Conditional formatting for score visualization
- Formulas: `=AVERAGE(B2:B100)` for column averages

## File Locations

```
LLM-Project/
├── cs_prob.py                    # Main program ⭐
├── simple_prompt.py              # Quick testing
├── evaluation.py                 # Legacy eval script
├── README.md                     # Overview
├── CS_PROB_GUIDE.md             # Full guide ⭐
├── QUICK_REFERENCE.md           # This file
└── requirements.txt              # Dependencies
```

## Getting Help

1. **Quick issues:** This file
2. **Detailed guide:** `CS_PROB_GUIDE.md`
3. **Code issues:** Check console error messages
4. **API issues:** https://status.cloud.google.com/

## Tips

✅ **DO:**
- Test with small Excel file first (5-10 questions)
- Keep terminal window open to monitor progress
- Check console for detailed progress
- Save API keys securely

❌ **DON'T:**
- Run multiple instances simultaneously (API limits)
- Close terminal during processing
- Share API keys publicly
- Modify code while running

## Emergency Stop

Press `Ctrl+C` to stop execution. Partial results will be lost; re-run from scratch.

---

**Quick Start:** `python3 cs_prob.py --input_xlsx exams.xlsx`  
**Full Docs:** See `CS_PROB_GUIDE.md`
