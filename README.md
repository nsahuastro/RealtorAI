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
- Loads and preprocesses HDB resale flat data from CSV files
- Cleans and saves processed data
- Creates RAG documents and sets up a vector database
- Provides a simple Q&A system for housing queries

## Requirements
- Python 3.8+
- pandas
- sentence-transformers
- chromadb
- transformers

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
3. Place HDB resale flat CSV files in the `ResaleFlatPrices/` folder.

## Usage
Run the main script:
```bash
python main.py
```

## Project Structure
- `main.py`: Main entry point for data processing and Q&A setup
- `get_hdb_data.py`: Functions for loading HDB data
- `preprocessing_hdb_data.py`: Data cleaning and preprocessing
- `rag_setup.py`: RAG document creation and vector DB setup
- `ResaleFlatPrices/`: Raw CSV data files
- `Processed_Data/`: Output folder for processed data

## Example Questions
- "Tell me the average resale price of 2 room HDB in Tampines in 2024?"
- "What's the most expensive flat type?"
- "How much do 4-room flats cost in Jurong?"

## Logging
The script uses Python's logging module for progress and error reporting.

## License
MIT
