
import re
from datetime import datetime

import chromadb
from transformers import pipeline

VECTOR_DB_PATH = "Processed_Data/chroma_db"
VECTOR_COLLECTION_NAME = "hdb_data"
HDB_TOWNS = (
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "LIM CHU KANG", "MARINE PARADE", "PASIR RIS",
    "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON",
    "TAMPINES", "TOA PAYOH", "WOODLANDS", "YISHUN",
)
BEDROOM_TO_FLAT_TYPES = {
    1: ["2 ROOM"],
    2: ["3 ROOM"],
    3: ["4 ROOM", "5 ROOM", "EXECUTIVE"],
}
EXPLICIT_FLAT_TYPE_PATTERNS = {
    f"{room_count} ROOM": rf"\b{room_count}\s*[- ]?\s*ROOM\b"
    for room_count in range(1, 6)
}
EXPLICIT_FLAT_TYPE_PATTERNS.update({
    "EXECUTIVE": r"\bEXECUTIVE\b",
    "MULTI GENERATION": r"\bMULTI[- ]GENERATION\b",
    "MULTI-GENERATION": r"\bMULTI[- ]GENERATION\b",
})


def build_hdb_metadata_filter(question):
    normalized_question = question.upper()
    conditions = []

    town = next(
        (
            town for town in sorted(HDB_TOWNS, key=len, reverse=True)
            if re.search(rf"(?<![A-Z]){re.escape(town)}(?![A-Z])", normalized_question)
        ),
        None,
    )
    if town:
        conditions.append({"town": {"$eq": town}})

    flat_types = [
        flat_type
        for flat_type, pattern in EXPLICIT_FLAT_TYPE_PATTERNS.items()
        if re.search(pattern, normalized_question)
    ]
    if not flat_types:
        bedroom_match = re.search(r"\b([1-9])\s*-?\s*BEDROOMS?\b", normalized_question)
        if bedroom_match:
            flat_types = BEDROOM_TO_FLAT_TYPES.get(int(bedroom_match.group(1)), [])

    if flat_types:
        conditions.append({"flat_type": {"$in": flat_types}})

    if len(conditions) == 1:
        return conditions[0]
    if conditions:
        return {"$and": conditions}
    return None


def format_retrieved_transactions(question, metadatas):
    include_price_per_sqm = bool(
        re.search(r"\b(?:sqm|per\s+(?:square\s+)?met(?:er|re))\b", question, re.IGNORECASE)
    )
    include_lease = "lease" in question.lower()
    summaries = []

    for metadata in metadatas:
        flat_type = metadata.get("flat_type", "HDB flat").lower().replace(" room", "-room")
        town = metadata.get("town", "Singapore").title()
        price = float(metadata.get("price", 0))
        sold_date = metadata.get("sold_date", "")
        try:
            sold_date = datetime.strptime(sold_date, "%Y-%m").strftime("%B %Y")
        except (TypeError, ValueError):
            pass

        summary = f"A {flat_type} HDB flat in {town} sold for ${price:,.0f}"
        if sold_date:
            summary += f" in {sold_date}"
        summary += "."

        if include_price_per_sqm and metadata.get("price_per_sqm") is not None:
            summary += f" Its price was ${float(metadata['price_per_sqm']):,.0f} per square meter."
        if include_lease and metadata.get("remaining_lease") is not None:
            summary += f" It has {float(metadata['remaining_lease']):.0f} years remaining on the lease."
        summaries.append(summary)

    return "\n".join(summaries)


def create_rag_documents(df, sample_size=1000):
    #Step 1: Convert your HDB data into text documents for RAG
    # chooseing saple_size=1000 for fast testing, use larger size for production later

    print("Creating RAG documents...")

    # Sample data to keep it fast for testing
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples from {len(df)} total records")
    else:
        df_sample = df
    
    documents = []

    for idx, row in df_sample.iterrows():
        flat_type = row["flat_type"].lower().replace(" room", "-room")
        town = row["town"].title()
        price_per_sqm = row["resale_price"] / row["floor_area_sqm"]
        doc_text = (
            f"A {flat_type} HDB flat in {town} is at Block {row['block']}, "
            f"{row['street_name']}. It has {row['floor_area_sqm']:g} square meters "
            f"of floor area and spans floors {row['storey_range_min']} to "
            f"{row['storey_range_max']}. Built in {row['lease_commence_date'].year}, "
            f"it has {row['remaining_lease']:.0f} years remaining on the lease. "
            f"It sold for ${row['resale_price']:,.0f} SGD in "
            f"{row['month'].strftime('%B %Y')}. Its model is {row['flat_model']}, "
            f"and its price was ${price_per_sqm:,.0f} per square meter."
        )
        
        documents.append({
            'id': f"hdb_{idx}",
            'text': doc_text,
            'metadata': {
                'town': row['town'],
                'flat_type': row['flat_type'],
                'price': row['resale_price'],
                'price_per_sqm': price_per_sqm,
                'sold_date': row['month'].strftime('%Y-%m'),
                'remaining_lease': row['remaining_lease']
            }
        })
    print(f"Created {len(documents)} documents for RAG, one document per flat record")
    return documents

def get_persisted_vector_database(source_signature):
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    try:
        collection = client.get_collection(name=VECTOR_COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        return None

    if (
        collection.metadata
        and collection.metadata.get("source_signature") == source_signature
        and collection.count() > 0
    ):
        print("Reusing persistent HDB vector collection")
        return collection

    client.delete_collection(name=VECTOR_COLLECTION_NAME)
    return None


def setup_vector_database(input_documents, source_signature):
    print("Building persistent vector database...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    collection = client.create_collection(
        name=VECTOR_COLLECTION_NAME,
        metadata={"source_signature": source_signature},
    )
    print("Vector database collection 'hdb_data' created")

    # Add documents to database
    print("Adding documents to vector database...")
    texts = [doc['text'] for doc in input_documents]
    ids = [doc['id'] for doc in input_documents]
    metadatas = [doc['metadata'] for doc in input_documents]

    # generate embeddings and add to vector database collection
    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    print(f"Added {len(input_documents)} documents to vector database")

    return collection

def create_simple_qa_system():

    #Create a simple Q&A pipeline/system
    print("Setting up simple Q&A system...")

    # Use a free, lightweight language model
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
    )
    print("Q&A pipeline initialized with google/flan-t5-small")
    return qa_pipeline


def ask_hdb_question(question, collection, qa_pipeline=None, top_k=3):
    """Build a summary from retrieved HDB records without text generation."""
    print(f"\n Question: {question}")
    
    # Step 1: Retrieve relevant documents
    results = collection.query(query_texts=[question], n_results=top_k)
    
    if results['documents'] and results['documents'][0]:
        print(f"Found {len(results['documents'][0])} relevant documents")
        
        # Extract prices and info from metadata (this works!)
        prices = []
        towns = []
        flat_types = []
        sold_date = []
        
        for meta in results['metadatas'][0]:
            if meta.get('price') and float(meta.get('price', 0)) > 0:
                prices.append(float(meta['price']))
            if meta.get('town'):
                towns.append(meta['town'])
            if meta.get('flat_type'):
                flat_types.append(meta['flat_type'])
            if meta.get('sold_date'):
                sold_date.append(meta['sold_date'])
        
        if prices:
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
            unique_towns = list(set(towns))
            unique_types = list(set(flat_types))
            unique_dates = list(set(sold_date))
            
            answer = f"""Based on {len(prices)} relevant documents I found about HDB transactions:
                        - Average price: ${avg_price:,.0f}
                        - Price range: ${min_price:,.0f} - ${max_price:,.0f}
                        - Locations: {', '.join(unique_towns)}
                        - Flat types: {', '.join(unique_types)}
                        - Sold dates:{', '.join(unique_dates)}"""
            
            print(f"Answer: {answer}")
            return answer
    
    return "I don't have information about that in my HDB database."

def ask_hdb_question_txtgen(question, collection, qa_pipeline, top_k=3):
    #Answer questions about HDB data
    print(f"Answering question: {question}")

    # Step 1: Retrieve relevant documents from vector database
    query = {"query_texts": [question], "n_results": top_k}
    metadata_filter = build_hdb_metadata_filter(question)
    if metadata_filter:
        query["where"] = metadata_filter
    results = collection.query(**query)
    
    print(f"Retrieved {len(results['documents'][0])} documents")
    #print(f"First document preview: {results['documents'][0][0][:200]}...")
    
    # Step 2: Combine retrieved documents into context
    if results['documents'] and results['documents'][0]:
        context = format_retrieved_transactions(question, results["metadatas"][0])
        print(f"Retrieved {len(results['documents'][0])} relevant documents from vector database")

        prompt = f"""You are an HDB resale assistant. Answer in one or two concise, natural sentences using only facts from these transactions. Answer the specific question and include only relevant details. For an unqualified price question, give the total resale price, not the price per square meter. For a per-square-meter question, use the per-square-meter figure. Do not list property fields or repeat every detail. If multiple sales are relevant, summarize them without inventing an average or range.
Question: {question}
    Relevant transactions:
{context[:1500]}
Answer:"""

        try:
            response = qa_pipeline(
                prompt,
                max_new_tokens=128,
                truncation=True,
            )
            answer = response[0]["generated_text"].strip()
            if re.fullmatch(r"[\s$,.\d]+", answer):
                answer = context.splitlines()[0]
            print(f"Answer: {answer}")
            return answer
        except Exception as e:
            print(f"Text generation failed: {e}")
            return "I found relevant records but couldn't generate an answer. Please try again."
    else:
        print("No relevant data found")
        return "I'm sorry, I couldn't find relevant information to answer your question."
