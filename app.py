import os
import logging
import streamlit as st
import pandas as pd

from get_hdb_data import load_hdb_data_from_csv
from preprocessing_hdb_data import preprocessing_hdb_dataframe
from rag_setup import (
    create_rag_documents,
    get_persisted_vector_database,
    setup_vector_database,
    create_simple_qa_system,
    ask_hdb_question_txtgen,
)

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_FOLDER = "ResaleFlatPrices/"
OUTPUT_FOLDER = "Processed_Data/"
PREPROCESSED_DATA_PATH = os.path.join(OUTPUT_FOLDER, "preprocessed_combined_hdb_data.csv")
SAMPLE_SIZE = 10000
RAG_DOCUMENT_VERSION = "hdb-documents-v3"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def ensure_preprocessed_data():
    if os.path.exists(PREPROCESSED_DATA_PATH):
        return True

    if not os.path.exists(DATA_FOLDER):
        st.error(f"Data folder '{DATA_FOLDER}' not found.")
        return False

    combined_hdb_df = load_hdb_data_from_csv(folder_path=DATA_FOLDER)
    if combined_hdb_df.empty:
        st.error("No data loaded from combined CSV files.")
        return False

    cleaned_hdb_df = preprocessing_hdb_dataframe(combined_hdb_df)
    cleaned_hdb_df.to_csv(PREPROCESSED_DATA_PATH, index=False)
    return True


@st.cache_resource  # cache so setup runs only once
def setup_realtor_ai():
    """Reuse preprocessed HDB data and a persistent vector collection."""
    if not ensure_preprocessed_data():
        return None, None

    file_stat = os.stat(PREPROCESSED_DATA_PATH)
    source_signature = (
        f"{RAG_DOCUMENT_VERSION}:{file_stat.st_size}:"
        f"{file_stat.st_mtime_ns}:{SAMPLE_SIZE}"
    )
    vector_db_collection = get_persisted_vector_database(source_signature)

    if vector_db_collection is None:
        cleaned_hdb_df = pd.read_csv(
            PREPROCESSED_DATA_PATH,
            parse_dates=["month", "lease_commence_date"],
        )
        rag_documents = create_rag_documents(cleaned_hdb_df, sample_size=SAMPLE_SIZE)
        vector_db_collection = setup_vector_database(rag_documents, source_signature)

    qa_pipeline = create_simple_qa_system()

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
            answer = ask_hdb_question_txtgen(query, vector_db_collection, qa_pipeline)
        st.success("Answer")
        st.write(answer)
else:
    st.warning("RealtorAI is not initialized. Check your dataset.")


