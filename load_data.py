import pandas as pd
import json

def load_jsonl_to_df(filepath):
    papers = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line))
    df = pd.DataFrame(papers)
    return df

if __name__ == "__main__":
    df = load_jsonl_to_df("data/philosophy_papers.jsonl")
    print(df.head())
