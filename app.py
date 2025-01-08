from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
cors = CORS(app, resources={r"/*": {"origins": "*"}})
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY is not set in the environment variables.")

genai.configure(api_key=GENAI_API_KEY)

try:
    pdf_loader = PyPDFLoader("MyResumeMarkdown.pdf")
    pages = pdf_loader.load_and_split()
    all_context = "\n\n".join(str(page.page_content) for page in pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(all_context)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GENAI_API_KEY
    )
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GENAI_API_KEY,
        temperature=0.2,
        convert_system_message_to_human=True
    )

    QA_CHAIN_PROMPT = PromptTemplate.from_template(
        """You are The ChatBot. You have the Resume of Abhishek Gupta. If anyone asks about him, provide a concise and accurate response. If there is any link related to that topic then provide it and if there any personal link then also provide it to him. If you don't know the answer then say "I don't know Go ask himself".
        Context:
        {context}

        Question: {question}
        Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
except Exception as e:
    raise RuntimeError(f"Error initializing LangChain pipeline: {e}")


app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question")
        if not question:
            return jsonify({"error": "Question is required."}), 400

        result = qa_chain({"query": question})
        answer = result["result"]
        source_docs = [doc.page_content for doc in result["source_documents"]]

        return jsonify({"answer": answer, "sources": source_docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run()
