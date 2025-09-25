
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

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
        # Create a text representation of each row
        doc_text= f"""
        {row['flat_type']} HDB flat in {row['town']}:
        - Location: Block {row['block']}, {row['street_name']}, {row['town']}
        - Size: {row['floor_area_sqm']} square meters
        - Floor level: {row['storey_range']} (floors {row['storey_range_min']} to {row['storey_range_max']})
        - Building age: Built in {row['lease_commence_date'].year}, {row['remaining_lease']:.0f} years lease remaining
        - Price: Sold for ${row['resale_price']:,.0f} SGD in {row['month'].strftime('%B %Y')}
        - Model: flat model {row['flat_model']}  
        - Price per sqm: ${row['resale_price']/row['floor_area_sqm']:,.0f} per square meter
            """.strip()
        
        documents.append({
            'id': f"hdb_{idx}",
            'text': doc_text,
            'metadata': {
                'town': row['town'],
                'flat_type': row['flat_type'],
                'price': row['resale_price'],
                'sold_date': row['month'].strftime('%Y-%m'),
                'remaining_lease': row['remaining_lease']
            }
        })
    print(f"Created {len(documents)} documents for RAG, one document per flat record")
    return documents

def setup_vector_database(input_documents):
    # Create vector database for retrieval
    print("Setting up vector database...")

    # Initialize embedding model (lightweight and free)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("embedding model initialized: all-MiniLM-L6-v2 ")

    # Initialize ChromaDB (local vector database)
    client = chromadb.Client()

    # Create collection (delete if already exists)
    try:
        client.delete_collection(name="hdb_data")
        print("Deleted existing 'hdb_flats' collection")
    except:
        pass

    collection = client.create_collection("hdb_data")
    print("Vector database collection 'hdb_data' created")

    # Add documents to database
    print("Adding documents to vector database...")
    texts = [doc['text'] for doc in input_documents]
    ids = [doc['id'] for doc in input_documents]
    metadatas = [doc['metadata'] for doc in input_documents]

    # generate embeddings and add to vector database collection
    collection.add( documents=texts, ids=ids, metadatas=metadatas)
    print(f"Added {len(input_documents)} documents to vector database")

    return collection, embedding_model

def create_simple_qa_system(collection):

    #Create a simple Q&A pipeline/system
    print("Setting up simple Q&A system...")

    # Use a free, lightweight language model
    qa_pipeline = pipeline(
        "text2text-generation",
        model= "google/flan-t5-small",  # small and free model "microsoft/DialoGPT-medium"
        max_length=200,
        truncation=True,
        padding=True
        #do_sample=True,
        #temperature=0.7
       # tokenizer="google/flan-t5-small",
        #device=-1  # use CPU; set to 0 if you have a compatible GPU
    )
    print("Q&A pipeline initialized with microsoft/DialoGPT-medium model")
    return qa_pipeline


def ask_hdb_question(question, collection, qa_pipeline=None, top_k=3):
    """
    Answer questions about HDB data without using text generation model
    """
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
    results = collection.query(
        query_texts=[question],
        n_results=top_k)
    
    print(f"Retrieved {len(results['documents'][0])} documents")
    #print(f"First document preview: {results['documents'][0][0][:200]}...")
    
    # Step 2: Combine retrieved documents into context
    if results['documents'] and results['documents'][0]:
        context = "\n\n".join(results['documents'][0])
        print(f"Retrieved {len(results['documents'][0])} relevant documents from vector database")

        prompt= f""" Based on the HDB resale data provided, answer this question: {question}
        HDB data:
        {context[:1500]}  # limit context to first 1500 characters to avoid token limits
        Answer:"""

        # Step 4: Generate answer
        try:
            response = qa_pipeline(prompt, max_length=len(prompt.split()) + 50)
            answer = response[0]['generated_text'][len(prompt):].strip()
            
            print(f"💡 Answer: {answer}")
            return answer
        except Exception as e:
            # Simple fallback answer
            avg_price = sum([float(meta.get('price', 0)) for meta in results['metadatas'][0]]) / len(results['metadatas'][0])
            towns = [meta.get('town', '') for meta in results['metadatas'][0]]
            fallback_answer = f"Based on the data I found, the HDB flats in {', '.join(set(towns))} have average resale price around ${avg_price:,.0f}."
            print(f"💡 Answer: {fallback_answer}")
            return fallback_answer
    else:
        print("No relevant data found")
        return "I'm sorry, I couldn't find relevant information to answer your question."
