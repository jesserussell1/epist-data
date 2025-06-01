import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    # Basic cleaning: lowercase, remove non-letters, extra spaces
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_advanced_nlp(df, n_topics=10, max_features=1000):
    abstracts = df['abstract'].dropna().map(clean_text).reset_index(drop=True)
    years = df.loc[df['abstract'].dropna().index, 'year'].reset_index(drop=True)

    print(f"Running advanced NLP on {len(abstracts)} abstracts...")

    # Stopwords
    stop_words = list(stopwords.words('english'))

    # Vectorize abstracts with CountVectorizer for LDA
    vectorizer = CountVectorizer(max_df=0.95, min_df=5, max_features=max_features,
                                 stop_words=stop_words)

    doc_term_matrix = vectorizer.fit_transform(abstracts)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    # Function to print top words per topic
    def print_top_words(model, feature_names, n_top_words=10):
        for topic_idx, topic in enumerate(model.components_):
            print(f"\nTopic #{topic_idx}:")
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            top_weights = topic[top_features_ind]
            print(", ".join(top_features))

    print_top_words(lda, feature_names)

    # Topic distribution per document
    doc_topic_dist = lda.transform(doc_term_matrix)  # shape (n_docs, n_topics)

    # Add dominant topic and its proportion to DataFrame
    dominant_topic = doc_topic_dist.argmax(axis=1)
    dominant_topic_proportion = doc_topic_dist.max(axis=1)

    df = df.loc[abstracts.index].copy()  # align with cleaned abstracts
    df['dominant_topic'] = dominant_topic
    df['dominant_topic_proportion'] = dominant_topic_proportion

    # Plot topic prevalence over years
    topic_years = pd.DataFrame({
        'year': years,
        'dominant_topic': dominant_topic
    })

    topic_counts_per_year = topic_years.groupby(['year', 'dominant_topic']).size().unstack(fill_value=0)

    topic_counts_per_year.plot(kind='line', figsize=(12,6), colormap='tab10')
    plt.title("Topic Prevalence Over Years")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.legend(title="Topic")
    plt.tight_layout()
    plt.show()

    print("\nAdvanced NLP analysis complete.")



if __name__ == "__main__":
    print("This script is meant to be imported and used as a module.")
