import os
import re
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

# Define the persistence directory globally for this module
PERSIST_DIRECTORY = "./chroma_db"

load_dotenv()

# --- Data Loading and Preprocessing ---
def load_and_prepare_data(csv_path: str):
    """
    Loads insurance policy data from a CSV, creates Document objects, 
    and a dictionary for quick policy ID lookups.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        exit()

    print("Loading insurance policy data from CSV...")
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} policy documents.")

    policy_dict = {}
    for doc in documents:
        match = re.search(r"policy_id: (POL\d+)", doc.page_content)
        if match:
            policy_id = match.group(1)
            policy_dict[policy_id] = doc
    print(f"Created dictionary for {len(policy_dict)} unique policy IDs.")
    return documents, policy_dict


def create_and_persist_vector_store(documents: list[Document], embeddings):
    """
    Creates a new Chroma vector store from documents and persists it to disk.
    If a store already exists, it will be overwritten.
    """
    print(f"Creating new vector store and persisting to {PERSIST_DIRECTORY}...")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIRECTORY)
    print("Vector store created and persisted successfully.")
    return vectorstore

def load_persisted_vector_store(embeddings=None):
    """
    Loads an existing Chroma vector store from disk.
    If no embeddings are provided, initializes GoogleGenerativeAIEmbeddings by default.
    """
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        print(f"Error: Vector store not found at {PERSIST_DIRECTORY}.")
        print("Please run utils.py once to create the database.")
        exit()
    if embeddings is None:
        print("No embeddings provided. Initializing default Google embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print(f"Loading existing vector store from {PERSIST_DIRECTORY}...")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    print("Vector store loaded successfully.")
    return vectorstore

# --- Custom Retriever Logic (Modified for strict policy ID matching) ---
def get_custom_retriever_function(vectorstore, policy_lookup_dict: dict):
    """
    Returns a callable custom retriever function that handles direct policy ID lookups.
    If a policy ID is not found or not matched, it strictly returns an empty list
    """
    def _custom_retriever_logic(query: str):
        policy_id_match = re.search(r"POL\d{3}", query.upper()) # Regex to find POLXXX format

        if policy_id_match:
            extracted_policy_id = policy_id_match.group(0)
            if extracted_policy_id in policy_lookup_dict:
                print(f"[Retrieval]: Directly retrieved policy: {extracted_policy_id}")
                return [policy_lookup_dict[extracted_policy_id]]
            else:
                print(f"[Retrieval]: Policy ID '{extracted_policy_id}' not found. No document retrieval from vectorstore.")
                return [] # Policy ID found but not in dict, so no context
        else:
            print("[Retrieval]: No specific policy ID found in query. No document retrieval from vectorstore.")
            return [] # No policy ID format detected, so no context
    return RunnableLambda(_custom_retriever_logic).with_config(run_name="CustomPolicyRetriever")

# --- Main execution block for data ingestion/testing ---
if __name__ == "__main__":
    print("--- Running Data Ingestion and Utilities Test ---")
    csv_file_path = "insurance_policies_sample_100_final.csv"

    # Step 1: Prepare data and embeddings
    all_documents, policy_id_dict = load_and_prepare_data(csv_file_path)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Step 2: Create and persist the vector store (this happens only once, or when data changes)
    # This will ensure the DB is created if it doesn't exist. We load it if it exists, otherwise create it.
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print(f"Vector store already exists at {PERSIST_DIRECTORY}. Loading it.")
        vector_store_main = load_persisted_vector_store(embeddings)
    else:
        vector_store_main = create_and_persist_vector_store(all_documents, embeddings)

    # Step 3: Test the retriever with the loaded store
    my_retriever = get_custom_retriever_function(vector_store_main, policy_id_dict)

    print("\n--- Testing Custom Retriever Function ---")
    # Test with valid policy ID
    query_test_id_valid = "What is the premium for policy POL001?"
    retrieved_docs_id_valid = my_retriever.invoke(query_test_id_valid)
    print(f"\nRetrieved documents for query (ID-based, valid): '{query_test_id_valid}':")
    if retrieved_docs_id_valid:
        for i, doc in enumerate(retrieved_docs_id_valid):
            print(f"--- Retrieved Doc {i+1} ---")
            print(f"Content:\n{doc.page_content.strip()}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 20)
    else:
        print("No documents retrieved (expected for non-matching ID or no ID).")

    # Test with invalid policy ID (format exists, but not in data)
    query_test_id_invalid = "What is the status of policy POL099?"
    retrieved_docs_id_invalid = my_retriever.invoke(query_test_id_invalid)
    print(f"\nRetrieved documents for query (ID-based, invalid): '{query_test_id_invalid}':")
    if retrieved_docs_id_invalid:
        for i, doc in enumerate(retrieved_docs_id_invalid):
            print(f"--- Retrieved Doc {i+1} ---")
            print(f"Content:\n{doc.page_content.strip()}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 20)
    else:
        print("No documents retrieved (expected for non-matching ID or no ID).")

    # Test with general query (no policy ID format)
    query_test_general = "Tell me about auto insurance"
    retrieved_docs_general = my_retriever.invoke(query_test_general)
    print(f"\nRetrieved documents for query (general, no ID): '{query_test_general}':")
    if retrieved_docs_general:
        for i, doc in enumerate(retrieved_docs_general):
            print(f"--- Retrieved Doc {i+1} ---")
            print(f"Content:\n{doc.page_content.strip()}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 20)
    else:
        print("No documents retrieved (expected for non-matching ID or no ID).")

    print("\n--- Data Ingestion and Utilities Test Completed. ---")