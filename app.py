from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import prompt
import os
from flask_cors import CORS   # ✅ added for safety

app = Flask(__name__)
CORS(app)  # ✅ allow frontend → backend requests

load_dotenv()

# Get API keys with validation
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Check if API keys exist
if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not found in .env file")
    exit(1)
if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY not found in .env file")
    exit(1)

print("✅ API Keys loaded successfully")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_embeddings()

index_name = "business-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chat = ChatOpenAI(
    model="gpt-4o-mini",  # ✅ Correct OpenRouter model ID
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat_route():
    try:
        msg = request.form["msg"]
        print(f"📩 Received message: {msg}")
        response = rag_chain.invoke({"input": msg})
        print("✅ Response generated successfully")
        return jsonify({"answer": response["answer"]})  # ✅ return JSON
    except Exception as e:
        print(f"❌ Error in chat_route: {str(e)}")
        return jsonify({"answer": "Error: Unable to process your request. Please check the server logs."})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
