# CS-PROB: University CS Exam LLM Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/License-Academic-green.svg)]()

A comprehensive tool for evaluating Large Language Models (LLMs) on university-level computer science exam questions using an automated judge system.

## 🎯 Quick Start

### 1. Install Dependencies

```bash
pip install pandas requests openpyxl google-generativeai
```

### 2. Configure API Keys

Edit `cs_prob.py` and replace the placeholder keys:

```python
JUDGE_API_KEYS = [
    "YOUR_GOOGLE_AI_STUDIO_KEY_1",
    "YOUR_GOOGLE_AI_STUDIO_KEY_2",
    "YOUR_GOOGLE_AI_STUDIO_KEY_3",
    "YOUR_GOOGLE_AI_STUDIO_KEY_4",
    "YOUR_GOOGLE_AI_STUDIO_KEY_5",
]
```

Get keys from: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Run Evaluation

```bash
python3 cs_prob.py --input_xlsx exams.xlsx
```

## 📋 What It Does

1. **Reads** Excel workbook with multiple sheets (one per domain)
2. **Queries** 4 local LLMs with each question:
   - Aya-Expanse-32B
   - GPT-OSS-120B
   - Llama-3.3-70B
   - Qwen2.5-72B
3. **Judges** answers using Gemini (state-of-the-art Judge LLM)
4. **Scores** on 0-1 scale based on:
   - Correctness
   - Reasoning quality
   - Clarity
   - Conciseness
5. **Saves** results to Excel with detailed metrics

## 🎨 Features

- ✅ **Multi-sheet Excel support** - Process multiple domains at once
- ✅ **Embedded image extraction** - Automatically reads images from Excel cells
- ✅ **Base64 encoding** - Handles multimodal questions with figures
- ✅ **API key rotation** - Uses 5 keys to avoid rate limits
- ✅ **Retry logic** - Handles network failures gracefully
- ✅ **Progress tracking** - Real-time console updates
- ✅ **Multiple outputs** - Per-sheet Excel, aggregated CSV, consolidated workbook

## 📂 Project Structure

```
LLM-Project/
├── cs_prob.py              # Main evaluation program
├── evaluation.py           # Original evaluation script
├── simple_prompt.py        # Quick LLM testing tool
├── CS_PROB_GUIDE.md       # Comprehensive usage guide
└── README.md              # This file
```

## 📊 Input Format

Your Excel file should have:

**Required columns:**
- `Question` - The exam question text
- `ReferenceAnswer` (or `Reference`) - Correct answer

**Optional:**
- `ImagePath` - File path to image
- Embedded images directly in Question cells

**Example:**

| Question | ReferenceAnswer |
|----------|----------------|
| What is the time complexity of binary search? | O(log n) |
| Explain TCP three-way handshake | SYN, SYN-ACK, ACK |

## 📈 Output Format

For each model, you get:

**Main score column:**
- `{ModelName}` - Overall score (0.0 to 1.0)

**Detailed columns:**
- `{ModelName}__answer` - Full LLM response
- `{ModelName}__correctness` - Technical accuracy (0-5)
- `{ModelName}__reasoning_quality` - Logic quality (0-5)
- `{ModelName}__clarity` - Structure quality (0-5)
- `{ModelName}__conciseness` - Brevity score (0-5)
- `{ModelName}__overall_score_raw` - Raw score (0-5)

## 🚀 Usage Examples

### Basic

```bash
python3 cs_prob.py --input_xlsx exams.xlsx
```

### Custom Output Path

```bash
python3 cs_prob.py --input_xlsx data/exams.xlsx --output_xlsx results/evaluation.xlsx
```

### Different Column Names

```bash
python3 cs_prob.py --input_xlsx exams.xlsx \
  --question_col "Problem" \
  --reference_col "Answer"
```

### Longer Timeout

```bash
python3 cs_prob.py --input_xlsx exams.xlsx --timeout 600
```

## 📖 Documentation

See [CS_PROB_GUIDE.md](CS_PROB_GUIDE.md) for:
- Detailed setup instructions
- Troubleshooting guide
- Advanced usage examples
- Performance optimization tips
- Result analysis methods

## 🔧 Quick Test

Test individual models with `simple_prompt.py`:

```bash
python3 simple_prompt.py "What is binary search?" \
  --api_url http://192.168.0.218:80/v1/chat/completions \
  --model Qwen/Qwen2.5-72B-Instruct
```

## 🤖 Models Under Test

| Model | Parameters | Endpoint |
|-------|-----------|----------|
| Aya-Expanse-32B | 32B | `192.168.0.219:80` |
| GPT-OSS-120B | 120B | `192.168.0.205:80` |
| Llama-3.3-70B | 70B | `192.168.0.217:80` |
| Qwen2.5-72B | 72B | `192.168.0.218:80` |

**Judge:** Gemini 1.5 Pro (via Google AI Studio)

## ⚙️ Configuration

Edit these constants in `cs_prob.py`:

```python
JUDGE_API_KEYS = [...]      # Your 5 Google AI Studio keys
SYSTEM_PROMPT = "..."       # Prompt for LLMs under test
TEMPERATURE = 0.0           # Sampling temperature
TIMEOUT = 300               # HTTP timeout (seconds)
```

## 🐛 Troubleshooting

### "Import errors"
```bash
pip install pandas requests openpyxl google-generativeai
```

### "Configure JUDGE_API_KEYS"
Edit `cs_prob.py` and add your Google AI Studio keys

### "Input file not found"
Use full path: `--input_xlsx /home/user/exams.xlsx`

### Timeout errors
Increase timeout: `--timeout 600`

### More help
See [CS_PROB_GUIDE.md](CS_PROB_GUIDE.md) troubleshooting section

## 📊 Performance

**Expected runtime:**
- Per question: ~30-60 seconds
- Per sheet (20 questions): ~10-20 minutes
- Full workbook (100 questions): ~50-100 minutes

**Factors:**
- Network speed to local LLM servers
- Google AI Studio API response time
- Question complexity
- Number of embedded images

## 🔒 Requirements

- Python 3.8+
- Network access to `192.168.0.x` (local LLM servers)
- Internet access (for Gemini API)
- Google AI Studio API keys (5 recommended)
- 4GB+ RAM

## 📝 Citation

If you use this tool in your research, please cite:

```
CS-PROB: A framework for evaluating Large Language Models on university-level
computer science exam questions using automated judging.
[Your Institution], 2025.
```

## 🤝 Contributing

This is a research project. For improvements:
1. Test changes thoroughly
2. Document new features
3. Share with project team

## 📧 Support

For issues:
1. Check [CS_PROB_GUIDE.md](CS_PROB_GUIDE.md)
2. Review console error messages
3. Contact project team members

## ⚖️ License

Academic research use only. Ensure compliance with:
- Google AI Studio Terms of Service
- Institutional research ethics
- Data privacy regulations

---

**Authors:** CS-PROB Project Team  
**Last Updated:** December 2025  
**Version:** 1.0
