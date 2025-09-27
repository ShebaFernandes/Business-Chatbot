from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split,download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader



# Load environment variables
load_dotenv()

# Load API keys from environment variables

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

pdf_path = r"C:\Users\hp\OneDrive\data\ecommerce_faq_extended.pdf"

loader = PyPDFLoader(pdf_path)
extracted_data = loader.load()

print(len(extracted_data))
print(extracted_data[0])


# extracted_data = load_pdf_files(data=r"C:\Users\hp\OneDrive\data\ecommerce_faq_extended.pdf")
filter_data = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(filter_data)

embebeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "business-chatbot"

# delete index if it exists with wrong dimension
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# create index with correct dimension = 384
pc.create_index(
    name=index_name,
    dimension=384,   # ✅ match your embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)


index = pc.Index(index_name)
print("✅ Connected to Pinecone index:", index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding= embebeddings,
    index_name=index_name
)

if len(texts_chunk) > 0:
    docsearch.add_documents(texts_chunk)
    print("✅ Documents successfully added to Pinecone!")
else:
    print("⚠️ No documents to add. Please check your PDF folder or filter logic.")



