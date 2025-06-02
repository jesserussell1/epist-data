from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

class PegasusSummarizer:
    def __init__(self, model_name='google/pegasus-xsum'):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def generate_summary(self, text, max_length=150):
        inputs = self.tokenizer(text, truncation=True, padding='longest', return_tensors='pt').to(self.device)
        summary_ids = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


def summarize_yearly_abstracts(df, summarizer, abstracts_per_year=50):
    yearly_summaries = {}
    for year, group in df.groupby('year'):
        sampled_abstracts = group['abstract'].sample(
            n=min(len(group), abstracts_per_year), random_state=42
        ).tolist()

        clean_abstracts = [a for a in sampled_abstracts if isinstance(a, str) and a.strip()]
        combined_text = " ".join(clean_abstracts)

        prompt = f"Summarize the following scientific abstracts from the year {year}: {combined_text}"
        summary = summarizer.generate_summary(prompt, max_length=150)
        yearly_summaries[year] = summary
    return yearly_summaries
