# Website Qdrant Assistant

Streamlit chatbot grounded on website content and stored in Qdrant. Uses Firecrawl for ingestion and Groq for responses.

## Features

- Crawl website pages and store chunks + source URLs in Qdrant
- Ask questions and get answers grounded in stored content
- Source links shown for transparency
- Prompt-driven behavior via 

## Project Structure

- `src/main.py` Streamlit app entrypoint
- `src/agent/` LLM + UI logic
- `src/ingest/` Firecrawl ingestion to Qdrant
- `src/prompts/` System prompts

## Requirements

- Python 3.10+
- Qdrant instance (Cloud or local)
- Firecrawl API key
- Groq API key

## Setup

1. Create a virtual environment and install dependencies.
2. Configure `.env` with:

   - `FIRECRAWL_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`


## Ingest (Firecrawl)

Crawl website pages and index them into Qdrant:

```powershell
python src/ingest/scraper.py
```

Notes:
- Default `limit` is set low to avoid Firecrawl credit errors.
- Adjust `limit` in `src/ingest/scraper.py` if you have credits.

## Run the App

```powershell
streamlit run src/main.py
```

## Configuration

- Collection name: `tarento_knowledge` (set in `src/ingest/scraper.py`)
- Prompt file: `src/prompts/docs_agent_prompt.py`

## Troubleshooting

- Firecrawl `Payment Required`: lower `limit` or add credits.
- Streamlit session errors: restart the app and clear cache.
- If answers are off-topic, re-run ingestion or raise relevance thresholds in `src/main.py`.
