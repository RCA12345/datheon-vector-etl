import os
import time
import pandas as pd
from datetime import datetime
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def extract_text_and_metadata(file_path, department, access_tier):
    file_name = os.path.basename(file_path)
    ext = file_name.split('.')[-1].lower()
    raw_text = ""
    
    if ext == 'pdf':
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text: raw_text += text + "\n"
            
    elif ext == 'csv':
        df = pd.read_csv(file_path)
        raw_text = df.to_string()
        
    elif ext in ['md', 'txt']:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
    else:
        print(f"Unsupported file format: {ext}")
        return None

    metadata = {
        "source": file_name,
        "department": department,
        "access_tier": access_tier,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    
    return raw_text, metadata

def chunk_text(raw_text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    
    docs = []
    for chunk in chunks:
        docs.append({"text": chunk, "metadata": metadata})
    return docs

def load_to_vector_db(all_documents, db_path="./chroma_db"):
    print(f"\n--- Ingestion Benchmarking Start ---")
    start_time = time.time()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    texts = [doc["text"] for doc in all_documents]
    metadatas = [doc["metadata"] for doc in all_documents]
    
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=db_path
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Batch Ingestion Completed Successfully!")
    print(f"Total Documents Processed: {len(all_documents)}")
    print(f"Time Taken: {total_time:.2f} seconds")
    print(f"--- Benchmarking End ---\n")
    return vector_db

if __name__ == "__main__":
    
    with open("finance_report.md", "w") as f:
        f.write("# Finance Report 2026\nThe company net profit increased by 15% in Q3.")
    with open("hr_policy.txt", "w") as f:
        f.write("HR Policy: All employees must log hours by Friday 5 PM.")

    files_to_process = [
        {"path": "finance_report.md", "dept": "Finance", "tier": "Level-3"},
        {"path": "hr_policy.txt", "dept": "HR", "tier": "Level-1"}
    ]
    
    all_processed_chunks = []
    for file_info in files_to_process:
        result = extract_text_and_metadata(file_info["path"], file_info["dept"], file_info["tier"])
        if result:
            text, meta = result
            chunks = chunk_text(text, meta)
            all_processed_chunks.extend(chunks)
            
    if all_processed_chunks:
        db = load_to_vector_db(all_processed_chunks)
        print("Testing Retrieval with Filter:")
        results = db.similarity_search("profit", k=1, filter={"department": "Finance"})
        for doc in results:
            print(f"Found: {doc.page_content} | Metadata: {doc.metadata}")
