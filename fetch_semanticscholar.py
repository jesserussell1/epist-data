import requests
import json
import time
from urllib.parse import quote_plus

def fetch_semantic_scholar_papers(query, fields, year_from=2000, max_results=1000, output_file="philosophy_papers.jsonl"):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    encoded_query = quote_plus(query)
    url = f"{base_url}?query={encoded_query}&fields={fields}&year={year_from}-"
    retrieved = 0
    total_expected = None

    with open(output_file, "w", encoding="utf-8") as file:
        while True:
            print(f"Fetching: {url}")
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Request failed with status {response.status_code}: {response.text}")
                break

            r = response.json()

            if total_expected is None:
                total_expected = r.get("total", 0)
                print(f"Total estimated papers: {total_expected}")

            if "data" in r:
                batch = r["data"]
                if not batch:
                    print("No more data returned, stopping.")
                    break
                # Adjust batch size to not exceed max_results
                batch = batch[:max_results - retrieved]
                retrieved += len(batch)
                print(f"Retrieved {retrieved} papers so far...")
                for paper in batch:
                    print(json.dumps(paper), file=file)

            if "token" in r and retrieved < max_results:
                url = f"{base_url}?query={encoded_query}&fields={fields}&year={year_from}-&token={r['token']}"
                time.sleep(0.2)  # Be polite to the API
            else:
                break

            if retrieved >= max_results:
                print(f"Reached max results limit: {max_results}")
                break

    print(f"Done! Retrieved {retrieved} papers total.")


if __name__ == "__main__":
    query = "(epistemology) OR (philosophy of science)"
    fields = "title,year,authors,abstract,venue,url"

    fetch_semantic_scholar_papers(query=query, fields=fields, year_from=2000, max_results=5000)
