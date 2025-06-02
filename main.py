import os
import pandas as pd
from load_data import load_jsonl_to_df
from analyze_trends import run_basic_analysis
from analyze_abstracts import run_advanced_nlp
from fetch_semanticscholar import fetch_semantic_scholar_papers
from trend_analysis_with_bert import analyze_trends_with_bert
from trend_analysis_with_bert import show_topic_trend_table
from trend_analysis_with_bert import get_top_keywords_per_topic
import trend_analysis_with_bert
import keyword_trends

pd.set_option('display.max_columns', None)


from local_bart_summarizer import PegasusSummarizer, summarize_yearly_abstracts

#
JSONL_PATH = "data/philosophy_papers.jsonl"
DF_PICKLE_PATH = "data/philosophy_papers.pkl"

# Initialize summarizer
summarizer = PegasusSummarizer()


def main():
    # Step 1: Fetch if JSONL does not exist
    if not os.path.exists(JSONL_PATH):
        print("JSONL file not found, fetching data...")
        fetch_semantic_scholar_papers()
    else:
        print("JSONL file found, skipping fetch.")

    # Step 2: Load DataFrame from pickle if exists, else from JSONL
    if os.path.exists(DF_PICKLE_PATH):
        print("Loading DataFrame from pickle...")
        df = pd.read_pickle(DF_PICKLE_PATH)
    else:
        print("Loading DataFrame from JSONL...")
        df = load_jsonl_to_df(JSONL_PATH)
        print("Saving DataFrame to pickle for faster future loads...")
        df.to_pickle(DF_PICKLE_PATH)

    # Run basic analysis (titles, years, etc.)
    #run_basic_analysis(df)

    # Run advanced NLP on abstracts
    #run_advanced_nlp(df)

    keyword_trends.run(df)

    # Run trends analysis with BERT
    #df_with_topics = analyze_trends_with_bert(df)
    #show_topic_trend_table(df_with_topics['topic'], df_with_topics['year'])
    #top_keywords = get_top_keywords_per_topic(df_with_topics)
    #print("\nTop keywords per topic:")
    #print(top_keywords)

    #for cluster_id in sorted(df_with_topics['topic'].unique()):
    #    print(f"\nCluster {cluster_id} keywords: {top_keywords.get(cluster_id, [])}")
    #    summary = trend_analysis_with_bert.summarize_cluster_abstracts(df_with_topics, cluster_id, summarizer)
    #    print(f"Summary: {summary}")

    #summaries = summarize_yearly_abstracts(df, summarizer)
    #for year, summary in summaries.items():
    #    print(f"Year {year} summary:\n{summary}\n")


if __name__ == "__main__":
    main()
