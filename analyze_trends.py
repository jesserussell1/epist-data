import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def run_basic_analysis(df):
    # Example: Basic title word trends over time

    df = df.dropna(subset=['title', 'year'])
    df['year'] = df['year'].astype(int)

    # Filter papers in a time range if you want
    df = df[(df['year'] >= 2000) & (df['year'] <= 2025)]

    # Simple example: count frequent words in titles by year
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        tokens = text.lower().split()
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        return " ".join(tokens)

    df['clean_title'] = df['title'].apply(clean_text)

    # Vectorize titles
    vectorizer = TfidfVectorizer(max_features=50)
    X = vectorizer.fit_transform(df['clean_title'])
    feature_names = vectorizer.get_feature_names_out()

    # Aggregate tf-idf scores by year
    df_tfidf = pd.DataFrame(X.toarray(), columns=feature_names)
    df_tfidf['year'] = df['year'].values

    yearly_trends = df_tfidf.groupby('year').mean()

    # Plot trends for top 5 words
    top_words = yearly_trends.mean().sort_values(ascending=False).head(5).index

    for word in top_words:
        plt.plot(yearly_trends.index, yearly_trends[word], label=word)
    plt.xlabel('Year')
    plt.ylabel('Average TF-IDF')
    plt.title('Top Title Words Trends Over Time')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Quick test example
    df = pd.read_json("data/philosophy_papers.jsonl", lines=True)
    run_analysis(df)