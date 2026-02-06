import requests
from bs4 import BeautifulSoup
import json

def scrape_tarento(urls):
    scraped_data = []
    for url in urls:
        print(f"Scraping: {url}...")
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (adjust tags based on Tarento's site structure)
            content = " ".join([p.text.strip() for p in soup.find_all(['p', 'h1', 'h2'])])
            
            scraped_data.append({
                "url": url,
                "text": content
            })
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            
    # Save to JSONL for the vector store script to read
    with open('scraped_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in scraped_data:
            f.write(json.dumps(entry) + '\n')
    print("✅ Scraping complete. Data saved to scraped_data.jsonl")

if __name__ == "__main__":
    tarento_urls = [
        "https://www.tarento.com/",
        "https://www.tarento.com/services",
        # Add more relevant URLs here
    ]
    scrape_tarento(tarento_urls)