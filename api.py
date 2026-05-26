import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embeddings import EmbeddingManager
from rag_storage import RAGStorage

from langchain_google_genai import ChatGoogleGenerativeAI

# Load env variables
load_dotenv()


# Initialize FastAPI
app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize services once
embed_manager = EmbeddingManager()
storage = RAGStorage()


# Request body schema
class ChatRequest(BaseModel):
    question: str


# Health Route
@app.get("/")
async def home():
    return {"success": True, "message": "RAG API Running"}


# Chat Route
@app.post("/chat")
async def chat(req: ChatRequest):

    try:
        question = req.question.strip()

        if not question:
            return {"success": False, "message": "Question is required"}

        # Generate embedding
        query_embedding = embed_manager.get_embedding(question)

        # Search similar docs
        matches = storage.search_similar_documents(
            query_embedding=query_embedding,
            limit=5,
        )

        # Filter weak matches
        matches = [doc for doc in matches if doc["similarity"] > 0.45]

        if not matches:
            return {
                "success": True,
                "answer": "I could not find that information in the HR policy.",
                "sources": [],
            }

        # Build context
        context_string = "\n\n".join([f"- {doc['content']}" for doc in matches])

        # Prompt
        prompt = f"""
You are an HR assistant chatbot.

Use ONLY the provided HR policy context to answer.

Instructions:
- Give concise and accurate answers.
- If the answer is not available in the context, say:
'I could not find that information in the HR policy.'

Context:
{context_string}

Question:
{question}

Answer:
"""

        # Gemini API
        gemini_key = os.getenv("GEMINI_API_KEY")

        chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0.0,
        )

        response = chat_model.invoke(prompt)

        # Sources
        sources = []

        for doc in matches:
            meta = doc["metadata"]

            sources.append(
                {
                    "source": meta.get("source"),
                    "topic": meta.get("topic", "HR Policy"),
                    "similarity": round(doc["similarity"], 4),
                }
            )

        return {
            "success": True,
            "question": question,
            "answer": response.content,
            "sources": sources,
        }

    except Exception as e:
        return {"success": False, "message": str(e)}
