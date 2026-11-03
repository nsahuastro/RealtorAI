# RealtorAI
An AI-powered Agent that answers questions about Singapore HDB resale prices using LLMs with RAG (Retrieval-Augmented Generation).

## Data Source
data source: https://data.gov.sg/collections/189/view

## Features
- 📊 Processes 900K+ HDB resale transactions
- 🤖 Uses RAG for accurate, data-driven answers  
- 🔍 Vector search with semantic understanding
- 📈 Real Singapore housing market insights

## Workings
- Reuses `Processed_Data/preprocessed_combined_hdb_data.csv` when available
- If the preprocessed file is missing, loads the raw CSV files, preprocesses them, and saves the canonical file
- Builds a persistent Chroma vector collection from a deterministic 10,000-row sample
- Retrieves matching transactions and uses FLAN-T5-small to generate a concise answer

## Requirements
- Python 3.10+
- Install the packages listed in `requirement.txt` (Transformers 4.x is required)

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
3. Keep the preprocessed CSV in `Processed_Data/`. If it is not present, place the raw HDB resale CSV files in `ResaleFlatPrices/` so the app can create it.

## Usage
Run the main script:
```bash
python main.py
```

## App Interface (Streamlit)
An interactive web interface is available using Streamlit. This allows users to query HDB resale data and view results in a user-friendly dashboard.

### How to Run the App
In your terminal, run:
```bash
streamlit run app.py
```
This will launch the Streamlit app in your browser.

### Startup and Data
The question box appears immediately. On the first submitted question, RealtorAI prepares the data, Chroma index, and FLAN-T5-small model. It indexes a deterministic sample of 1,000 rows into `Processed_Data/chroma_db/` in batches of 250. If a build is interrupted, the next run resumes from records already stored. Later questions and app starts reuse the complete collection while the preprocessed CSV, sample size, and document format are unchanged. A source change causes the index to be rebuilt. FLAN-T5-small runs locally; no LLM API key is required.

The app samples 1,000 of the roughly 963,000 available transactions for faster development. Results are based on that sample, not the complete dataset. Increasing the sample size causes a one-time index rebuild.

### HDB Room Terms
HDB room counts include the living room. A 3-room flat commonly has two bedrooms, so a query for a “2-bedroom flat” is filtered as `3 ROOM`. An explicit “2-room flat” query is filtered as `2 ROOM`. Town and flat-type filters are applied before semantic ranking when those terms are recognized.

Answers are generated from retrieved transactions. For example, “What was the price of a 2-bedroom flat in Ang Mo Kio?” maps to 3-room records in Ang Mo Kio. The lightweight model may produce a short answer; number-only outputs are presented with a grounded transaction sentence.

## Project Structure
- `main.py`: Main entry point for data processing and Q&A setup
- `get_hdb_data.py`: Functions for loading HDB data
- `preprocessing_hdb_data.py`: Data cleaning and preprocessing
- `rag_setup.py`: RAG document creation and vector DB setup
- `ResaleFlatPrices/`: Raw CSV data files
- `Processed_Data/`: Output folder for processed data

## Example Questions
- "What was the resale price of a 2-bedroom flat in Ang Mo Kio?"
- "What were some 4-room resale prices in Jurong West?"
- "What was the resale price per square meter of a flat in Tampines?"

## Logging
The script uses Python's logging module for progress and error reporting.

## License
MIT
