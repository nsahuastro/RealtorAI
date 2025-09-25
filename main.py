#import requests
import pandas as pd
import os
import logging
from datetime import datetime
from pathlib import Path
#import glob
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline
from get_hdb_data import get_hdb_datasets_from_api
from get_hdb_data import load_hdb_data_from_csv
from preprocessing_hdb_data import preprocessing_hdb_dataframe
from rag_setup import (create_rag_documents, setup_vector_database, create_simple_qa_system,
                       ask_hdb_question, ask_hdb_question_txtgen)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to execute the HDB data processing and LLM based Q&A setup"""

    try:
        DATA_FOLDER = "ResaleFlatPrices/"
        OUTPUT_FOLDER = "Processed_Data/"
        SAMPLE_SIZE = 1000  # For RAG documents, set to None to use all data

        logger.info("Starting HDB data processing")
        #loading data
        if not os.path.exists(DATA_FOLDER):
            logger.error(f"Data folder '{DATA_FOLDER}' not found.")
            return False
        
        combined_hdb_df=load_hdb_data_from_csv(folder_path=DATA_FOLDER)

        if combined_hdb_df.empty:
            logger.error("No data loaded from combined CSV files.")
            return False
        
        logger.info(f"Loaded {len(combined_hdb_df)} records")
        logger.info(f"Columns: {list(combined_hdb_df.columns)}")

        # Data type conversions
        logger.info("Converting data types...")
        combined_hdb_df = combined_hdb_df.astype({
                    'month': str, #use date format later
                    'town': str,
                    'flat_type': str,
                    'block': str,
                    'street_name': str,
                    'storey_range': str, #split later
                    'floor_area_sqm': float,
                    'flat_model': str,
                    'lease_commence_date': int, # convert to flat later to consider months? or use date format
                    'remaining_lease': str, #convert to float later
                    'resale_price': float
        })

        #preprocessing data
        logger.info("Preprocessing HDB dataframe...")

        cleaned_hdb_df = preprocessing_hdb_dataframe(combined_hdb_df)
        cleaned_hdb_df.to_csv(os.path.join(OUTPUT_FOLDER,f"preprocessed_combined_hdb_data_{timestamp}.csv"), index=False)   
        
        logger.info(f"Preprocessed data saved to {OUTPUT_FOLDER}preprocessed_combined_hdb_data_{timestamp}.csv")

        # Create RAG system
        rag_documents = create_rag_documents(cleaned_hdb_df, sample_size=SAMPLE_SIZE)
        vector_db_collection, embedding_model = setup_vector_database(rag_documents)
        qa_pipeline = create_simple_qa_system(vector_db_collection)

        logger.info("HDB Housing assistant setup complete.")

        #test questions
        test_questions = [
            "Tell me the average resale price of 2 room HDB in Tampines in 2024?",
            "What's the most expensive flat type?",
            "How much do 4-room flats cost in Jurong?"
        ]

        logger.info("Testing the HDB Q&A system with sample questions...")

        for question in test_questions:
            print(f"\n{'='*25}")
            ask_hdb_question(question, vector_db_collection)
        return True
    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Script completed successfully.")
    else:
        logger.error("Script encountered errors.")

    # If you want to test the text generation based Q&A system separately, uncomment below:
    '''
    ####---Assuming vector_db_collection and embedding_model are already set up---####
    question = "Tell me the average resale price of 2 room HDB in Tampines in 2024?"
    print("\nTesting text generation based Q&A system...")
    ask_hdb_question_txtgen(question, vector_db_collection, embedding_model)
    '''


