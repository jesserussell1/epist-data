import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)



def keyword_trend_analysis(df, year_col, text_col, keywords):
    years = sorted(df[year_col].dropna().unique())
    counts = {kw: [] for kw in keywords}

    for year in years:
        texts = df[df[year_col] == year][text_col].dropna().str.lower()
        for kw in keywords:
            count = texts.str.contains(kw.lower()).sum()
            counts[kw].append(count)

    trend_df = pd.DataFrame(counts, index=years)
    trend_df.index.name = 'Year'
    return trend_df

def run(df, year_col='year', text_col='abstract', keywords=None):
    if keywords is None:
        keywords = [
            "wittgenstein", "heidegger", "postmodernism", "deconstruction", "heisenberg", "uncertainty",
            "knowledge", "belief", "justification", "truth", "evidence", "rationality", "skepticism", "epistemology",
            "coherentism", "foundationalism", "reliabilism", "internalism", "externalism", "virtue epistemology",
            "testimony", "disagreement", "epistemic injustice", "social epistemology", "ignorance",
            "epistemic agency", "epistemic oppression",
            "Gettier", "Quine", "Goldman", "Alvin Plantinga", "Laurence BonJour", "Linda Zagzebski", "Miranda Fricker"
        ]

    trend_df = keyword_trend_analysis(df, year_col, text_col, keywords)
    # Transpose the DataFrame so topics are rows and years are columns
    pivot_df = trend_df.T  # Now topics are the index (rows)

    # Add a total column for each topic (i.e., sum across years)
    pivot_df['total'] = pivot_df.sum(axis=1)

    # Filter to keep only topics with more than 10 total mentions
    filtered_df = pivot_df[pivot_df['total'] > 10]

    # (Optional) Sort by total mentions
    filtered_df = filtered_df.sort_values(by='total', ascending=False)

    #print(filtered_df)

    print(filtered_df.to_csv())

    # Optionally save it
    trend_df.to_csv("output/keyword_trends.csv")
