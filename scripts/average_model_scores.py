import pandas as pd
import os


def print_avg_scores():
    base = os.path.join(os.path.dirname(__file__), '../evaluation')
    files = [
        'results_ayaExpanse8B+qwen7B_Qwen72B_judge.csv',
        'results_llama8B+mistral7B_qwen72B_judge.csv',
        'results_DeepSeek7B_Qwen7B_judge'
    ]
    model_columns = {
        files[0]: ['aya-expanse-8b(4bit)_Score', 'Qwen2.5-7B-Instruct(4bit)_Score'],
        files[1]: ['Meta-Llama-3-8B(4bit)_Score', 'Mistral-7B-v0.1(4bit)_Score'],
        files[2]: ['deepseek-llm-7b-chat(4bit)_Score_qwen2.5_7b']
    }
    results = []
    judge_models = {
        files[0]: 'Qwen2.5-72B',
        files[1]: 'Qwen2.5-72B',
        files[2]: 'Qwen2.5-7B'
    }
    for fname in files:
        fpath = os.path.join(base, fname)
        judge_model = judge_models.get(fname, 'Unknown')
        if fname.endswith('.csv'):
            df = pd.read_csv(fpath)
        else:
            df = pd.read_csv(fpath, sep=',')
        for col in model_columns[fname]:
            if col in df.columns:
                avg = df[col].mean()
                results.append(
                    {'Model': col, 'Average_Score': avg, 'Judge_Model': judge_model})
            else:
                results.append(
                    {'Model': col, 'Average_Score': 'column not found', 'Judge_Model': judge_model})
    # Save results to CSV
    outpath = os.path.join(base, 'average_model_scores.csv')
    pd.DataFrame(results).to_csv(outpath, index=False)
    print(f"Saved average scores to {outpath}")
    # Save results to JSON
    import json
    json_outpath = os.path.join(base, 'average_model_scores.json')
    with open(json_outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved average scores to {json_outpath}")


if __name__ == "__main__":
    print_avg_scores()
