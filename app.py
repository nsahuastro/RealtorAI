import os
import logging
from datetime import datetime
import streamlit as st
import pandas as pd

from get_hdb_data import load_hdb_data_from_csv
from preprocessing_hdb_data import preprocessing_hdb_dataframe
from rag_setup import (
    create_rag_documents,
    setup_vector_database,
    create_simple_qa_system,
    ask_hdb_question,
)

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_FOLDER = "ResaleFlatPrices/"
OUTPUT_FOLDER = "Processed_Data/"
SAMPLE_SIZE = 1000

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@st.cache_resource  # cache so setup runs only once
def setup_realtor_ai():
    """Load data, preprocess, and set up the RAG system."""
    if not os.path.exists(DATA_FOLDER):
        st.error(f"Data folder '{DATA_FOLDER}' not found.")
        return None, None

    combined_hdb_df = load_hdb_data_from_csv(folder_path=DATA_FOLDER)
    if combined_hdb_df.empty:
        st.error("No data loaded from combined CSV files.")
        return None, None

    # Preprocess
    cleaned_hdb_df = preprocessing_hdb_dataframe(combined_hdb_df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_hdb_df.to_csv(
        os.path.join(OUTPUT_FOLDER, f"preprocessed_combined_hdb_data_{timestamp}.csv"),
        index=False,
    )

    # Create RAG system
    rag_documents = create_rag_documents(cleaned_hdb_df, sample_size=SAMPLE_SIZE)
    vector_db_collection, embedding_model = setup_vector_database(rag_documents)
    qa_pipeline = create_simple_qa_system(vector_db_collection)

    return vector_db_collection, qa_pipeline


# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="RealtorAI", layout="wide")
st.title("RealtorAI - Singapore's HDB Housing Assistant")
st.write("Ask me anything about Singapore HDB resale prices!")

# Initialize system
vector_db_collection, qa_pipeline = setup_realtor_ai()

if vector_db_collection is not None:
    query = st.text_input("Your question:")
    if st.button("Ask") and query.strip():
        with st.spinner("Thinking..."):
            answer = ask_hdb_question(query, vector_db_collection)
        st.success("Answer")
        st.write(answer)
else:
    st.warning("RealtorAI is not initialized. Check your dataset.")


