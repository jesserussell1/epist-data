import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


def analyze_trends_with_bert(df, num_clusters=10, output_plot_path="bert_topic_trends.png"):
    df = df.dropna(subset=["abstract", "year"]).copy()

    if df.empty:
        print("No valid data after dropping missing abstracts or years.")
        return df

    print("Generating embeddings with Sentence-BERT...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["abstract"].tolist(), show_progress_bar=True)

    print(f"Clustering into {num_clusters} topics...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["topic"] = kmeans.fit_predict(embeddings)

    trend_data = df.groupby(["year", "topic"]).size().unstack(fill_value=0)
    trend_data_pct = trend_data.div(trend_data.sum(axis=1), axis=0)

    print("Plotting topic trends...")
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=trend_data_pct)
    plt.title("Topic Trends Over Time (Proportion of Abstracts)")
    plt.xlabel("Year")
    plt.ylabel("Proportion of Abstracts")
    plt.legend(title="Topic", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()

    print(f"Done. Plot saved to {output_plot_path}")

    return df


def get_top_keywords_per_topic(df, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    keywords = {}

    for topic_num in sorted(df['topic'].unique()):
        abstracts = df[df['topic'] == topic_num]['abstract']
        tfidf_matrix = vectorizer.fit_transform(abstracts)
        mean_tfidf = tfidf_matrix.mean(axis=0).A1  # average TF-IDF per term

        top_indices = mean_tfidf.argsort()[-num_keywords:][::-1]
        top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]

        keywords[topic_num] = top_terms

    return keywords


def show_topic_trend_table(topic_labels, years):
    import numpy as np  # make sure np is imported

    topic_labels = np.array(topic_labels).ravel()
    years = np.array(years).ravel()

    if len(topic_labels) != len(years):
        raise ValueError(f"Length mismatch: {len(topic_labels)} topics vs {len(years)} years")

    df_trends = pd.DataFrame({'year': years, 'topic': topic_labels})

    if df_trends.empty:
        print("No topic trends to display. DataFrame is empty.")
        return

    trend_table = df_trends.groupby(['year', 'topic']).size().unstack(fill_value=0)
    print("\n=== Topic Trends Table (Counts per Year) ===")
    print(trend_table)

    trend_props = trend_table.div(trend_table.sum(axis=1), axis=0)
    print("\n=== Topic Trends Table (Proportions per Year) ===")
    print(trend_props)

    return trend_table
