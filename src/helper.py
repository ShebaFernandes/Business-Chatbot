from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document  # or wherever your Document comes from



# Extracting text from PDF files in a directory
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
          glob="*/*.pdf", 
          loader_cls=PyPDFLoader
          )
    
    documents = loader.load()
   
    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []   # ✅ Correct initialization
    
    for doc in docs:
        src = doc.metadata.get("source")  # ✅ use `doc`, not `docs`
        page = doc.metadata.get("page")
        
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": src, "page": page}
        )
        
        minimal_docs.append(minimal_doc)
    
    return minimal_docs




def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # size of each chunk
        chunk_overlap=50,    # overlap between chunks
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks

# Example usage (assuming minimal_docs is loaded with PDFs)


def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
     ) # Use "cuda" if you have a compatible GPU)
    return embeddings

embeddings = download_embeddings()