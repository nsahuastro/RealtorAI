#import requests
import pandas as pd
import os
import logging
#import glob
from get_hdb_data import load_hdb_data_from_csv
from preprocessing_hdb_data import preprocessing_hdb_dataframe
from rag_setup import (
    create_rag_documents,
    get_persisted_vector_database,
    vector_database_is_complete,
    setup_vector_database,
    create_simple_qa_system,
    ask_hdb_question_txtgen,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to execute the HDB data processing and LLM based Q&A setup"""

    try:
        DATA_FOLDER = "ResaleFlatPrices/"
        OUTPUT_FOLDER = "Processed_Data/"
        PREPROCESSED_DATA_PATH = os.path.join(OUTPUT_FOLDER, "preprocessed_combined_hdb_data.csv")
        SAMPLE_SIZE = 1000
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        logger.info("Starting HDB data processing")
        if not os.path.exists(PREPROCESSED_DATA_PATH) and not os.path.exists(DATA_FOLDER):
            logger.error(f"Data folder '{DATA_FOLDER}' not found.")
            return False

        if not os.path.exists(PREPROCESSED_DATA_PATH):
            combined_hdb_df = load_hdb_data_from_csv(folder_path=DATA_FOLDER)
            if combined_hdb_df.empty:
                logger.error("No data loaded from combined CSV files.")
                return False
            cleaned_hdb_df = preprocessing_hdb_dataframe(combined_hdb_df)
            cleaned_hdb_df.to_csv(PREPROCESSED_DATA_PATH, index=False)
        else:
            logger.info(f"Using preprocessed data from {PREPROCESSED_DATA_PATH}")

        file_stat = os.stat(PREPROCESSED_DATA_PATH)
        source_signature = (
            f"hdb-documents-v3:{file_stat.st_size}:"
            f"{file_stat.st_mtime_ns}:{SAMPLE_SIZE}"
        )
        vector_db_collection = get_persisted_vector_database(source_signature)

        if vector_db_collection is None or not vector_database_is_complete(vector_db_collection):
            cleaned_hdb_df = pd.read_csv(
                PREPROCESSED_DATA_PATH,
                parse_dates=["month", "lease_commence_date"],
            )
            rag_documents = create_rag_documents(cleaned_hdb_df, sample_size=SAMPLE_SIZE)
            vector_db_collection = setup_vector_database(
                rag_documents,
                source_signature,
                collection=vector_db_collection,
            )

        qa_pipeline = create_simple_qa_system()

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
            ask_hdb_question_txtgen(question, vector_db_collection, qa_pipeline)
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
