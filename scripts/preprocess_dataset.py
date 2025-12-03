"""
Script to preprocess CS-PROB Dataset.xlsx and extract main columns from specific tabs.
Each sheet is mapped to a domain and rows are assigned Q_ID starting from 1.
Columns: Q_ID, Domain, Question, Answer, Difficulty, Source
Output: Combined CSV in data/processed_csprob.csv
"""

import pandas as pd
import os

INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/CS-PROB Dataset.xlsx'))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed_csprob.csv'))
MAIN_COLUMNS = ['Question', 'Answer', 'Difficulty', 'Source']

# Map sheet names to domains
SHEET_TO_DOMAIN = {
    'Tasmia done': 'Networking',
    'Utshab': 'ML',
    'Sazedur': 'Database',
    'Nawfal': 'Algo & DS',
    'Nieb': 'SWE'
}

def preprocess_excel(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    dfs = []
    q_id = 1
    
    for sheet in xls.sheet_names:
        # Skip 'Full' tab and any sheets not in our mapping
        if sheet == 'Full' or sheet not in SHEET_TO_DOMAIN:
            continue
            
        df = pd.read_excel(xls, sheet_name=sheet)
        # Only keep main columns that exist in this sheet
        cols = [col for col in MAIN_COLUMNS if col in df.columns]
        if cols:
            df_filtered = df[cols].copy()
            # Add Q_ID and Domain
            df_filtered.insert(0, 'Q_ID', range(q_id, q_id + len(df_filtered)))
            df_filtered.insert(1, 'Domain', SHEET_TO_DOMAIN[sheet])
            q_id += len(df_filtered)
            dfs.append(df_filtered)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path} with {len(combined)} questions")
    print(f"Domains: {combined['Domain'].value_counts().to_dict()}")

if __name__ == "__main__":
    preprocess_excel(INPUT_PATH, OUTPUT_PATH)
