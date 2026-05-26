import os
import requests
from dotenv import load_dotenv

from db import init_db
from embeddings import EmbeddingManager
from rag_storage import RAGStorage

from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


# ANSI Console Colors
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def main():
    print(
        f"\n{Colors.BOLD}{Colors.HEADER}"
        "================================================================"
        f"{Colors.ENDC}"
    )

    print(
        f"{Colors.BOLD}{Colors.CYAN}"
        "       WELCOME TO YOUR POSTGRESQL + PGVECTOR RAG PIPELINE       "
        f"{Colors.ENDC}"
    )

    print(
        f"{Colors.BOLD}{Colors.HEADER}"
        "================================================================"
        f"{Colors.ENDC}"
    )

    # Initialize embedding manager
    embed_manager = EmbeddingManager()

    # Initialize storage
    storage = RAGStorage()

    # Initialize DB
    try:
        dimension = embed_manager.get_dimension()

        success = init_db(embedding_dimension=dimension)

        if not success:
            print(
                f"\n{Colors.FAIL}"
                "[ERROR] Database initialization failed."
                f"{Colors.ENDC}"
            )
            return

    except Exception as e:
        print(
            f"\n{Colors.FAIL}"
            f"[ERROR] Database connection failed: {e}"
            f"{Colors.ENDC}"
        )
        return

    # Check document count
    doc_count = storage.get_document_count()

    print(
        f"\n[INFO] Current documents in Database: "
        f"{Colors.BOLD}{doc_count}{Colors.ENDC}"
    )

    if doc_count == 0:
        print(
            f"{Colors.WARNING}"
            "[WARNING] No documents found in database."
            f"{Colors.ENDC}"
        )

        print("Please run pdf_reader.py first to ingest HR policy PDFs.")
        return

    # Read LLM configs
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    ollama_model = os.getenv("OLLAMA_MODEL", "gemma2:2b")

    print(f"\n[CONFIG] Generation Engine: " f"{Colors.BOLD}{llm_provider}{Colors.ENDC}")

    print(
        f"\n{Colors.BOLD}{Colors.CYAN}"
        "[SYSTEM] INTERACTIVE HR POLICY CHATBOT"
        f"{Colors.ENDC}"
    )

    print("----------------------------------------------------------------")

    print(f"Type '{Colors.BOLD}exit{Colors.ENDC}' to quit")

    print(f"Type '{Colors.BOLD}clear{Colors.ENDC}' to clear all database documents")

    print("----------------------------------------------------------------\n")

    while True:
        try:
            query = input(
                f"{Colors.BOLD}{Colors.BLUE}Ask Question: {Colors.ENDC}"
            ).strip()

            if not query:
                continue

            if query.lower() == "exit":
                print(f"\n{Colors.GREEN}Goodbye 👋{Colors.ENDC}\n")
                break

            if query.lower() == "clear":
                storage.clear_all_documents()

                print("\nPlease run pdf_reader.py again to re-ingest PDFs.\n")
                break

            print(
                f"\n[INFO] Generating query embedding using "
                f"'{embed_manager.provider}'..."
            )

            # Create query embedding
            query_embedding = embed_manager.get_embedding(query)

            print("[INFO] Searching PostgreSQL for relevant chunks...")

            # Search similar docs
            matches = storage.search_similar_documents(
                query_embedding=query_embedding,
                limit=5,
            )

            # Filter weak matches
            matches = [doc for doc in matches if doc["similarity"] > 0.65]

            if not matches:
                print(
                    f"\n{Colors.WARNING}"
                    "[WARNING] No relevant HR policy context found."
                    f"{Colors.ENDC}\n"
                )

                continue

            # Print retrieved chunks
            print(
                f"\n{Colors.BOLD}{Colors.GREEN}"
                "================ RETRIEVED CONTEXT ================="
                f"{Colors.ENDC}"
            )

            for rank, doc in enumerate(matches, 1):
                meta = doc["metadata"]

                print(
                    f"\n{Colors.BOLD}"
                    f"Rank {rank} "
                    f"(Similarity: {doc['similarity']:.4f}) "
                    f"| Source: {meta.get('source')} "
                    f"| Topic: {meta.get('topic', 'HR Policy')}"
                    f"{Colors.ENDC}"
                )

                print(f"\n{doc['content'][:500]}...\n")

            print(
                f"{Colors.BOLD}{Colors.GREEN}"
                "===================================================="
                f"{Colors.ENDC}\n"
            )

            # Build context string
            context_string = "\n\n".join([f"- {doc['content']}" for doc in matches])

            # Build prompt
            prompt = f"""
You are an HR assistant chatbot.

Use ONLY the provided HR policy context to answer.

Instructions:
- Give concise and accurate answers.
- If the answer exists partially, summarize it clearly.
- If the answer is not available in the context, say:
"I could not find that information in the HR policy."

Do NOT make up information.

Context:
{context_string}

Question:
{query}

Answer:
"""

            print(
                f"{Colors.BLUE}"
                "[INFO] Sending retrieved context to LLM..."
                f"{Colors.ENDC}"
            )

            # GEMINI
            if llm_provider == "gemini":

                if not gemini_key:
                    print(
                        f"\n{Colors.WARNING}"
                        "[WARNING] GEMINI_API_KEY missing in .env"
                        f"{Colors.ENDC}"
                    )

                    continue

                try:
                    models_to_try = [
                        "gemini-2.5-flash",
                    ]

                    response = None
                    active_model = None

                    for model_name in models_to_try:
                        try:
                            print(f"[INFO] Trying Gemini model: {model_name}")

                            chat = ChatGoogleGenerativeAI(
                                model=model_name,
                                google_api_key=gemini_key,
                                temperature=0.0,
                            )

                            response = chat.invoke(prompt)

                            active_model = model_name

                            break

                        except Exception as e:
                            print("\n================ GEMINI ERROR ================\n")

                            print(f"Model: {model_name}")
                            print(f"Error: {e}")

                            print("\n==============================================\n")

                            continue

                    if response:
                        print(
                            f"\n{Colors.BOLD}{Colors.CYAN}"
                            f"[LLM RESPONSE - {active_model}]"
                            f"{Colors.ENDC}\n"
                        )

                        print(response.content)
                        print()

                    else:
                        print(
                            f"\n{Colors.FAIL}"
                            "[ERROR] All Gemini models failed."
                            f"{Colors.ENDC}\n"
                        )

                except Exception as e:
                    print(
                        f"\n{Colors.FAIL}"
                        f"[ERROR] Gemini API failed: {e}"
                        f"{Colors.ENDC}\n"
                    )

            # OPENAI
            elif llm_provider == "openai":

                if not openai_key:
                    print(
                        f"\n{Colors.WARNING}"
                        "[WARNING] OPENAI_API_KEY missing in .env"
                        f"{Colors.ENDC}"
                    )

                    continue

                try:
                    from langchain_openai import ChatOpenAI

                    chat = ChatOpenAI(
                        model="gpt-4o-mini",
                        openai_api_key=openai_key,
                        temperature=0.0,
                    )

                    response = chat.invoke(prompt)

                    print(
                        f"\n{Colors.BOLD}{Colors.CYAN}"
                        "[OPENAI RESPONSE]"
                        f"{Colors.ENDC}\n"
                    )

                    print(response.content)
                    print()

                except Exception as e:
                    print(
                        f"\n{Colors.FAIL}"
                        f"[ERROR] OpenAI API failed: {e}"
                        f"{Colors.ENDC}\n"
                    )

            # OLLAMA
            elif llm_provider == "ollama":

                try:
                    print(f"\n[INFO] Querying local Ollama model " f"'{ollama_model}'")

                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": ollama_model,
                            "prompt": prompt,
                            "stream": False,
                        },
                        timeout=60,
                    )

                    if response.status_code == 200:
                        result = response.json()

                        print(
                            f"\n{Colors.BOLD}{Colors.CYAN}"
                            "[OLLAMA RESPONSE]"
                            f"{Colors.ENDC}\n"
                        )

                        print(result.get("response"))
                        print()

                    else:
                        print(
                            f"\n{Colors.FAIL}"
                            f"[ERROR] Ollama returned "
                            f"{response.status_code}"
                            f"{Colors.ENDC}\n"
                        )

                except Exception as e:
                    print(
                        f"\n{Colors.FAIL}"
                        f"[ERROR] Ollama connection failed: {e}"
                        f"{Colors.ENDC}\n"
                    )

            else:
                print(
                    f"\n{Colors.WARNING}"
                    "[WARNING] Unsupported LLM_PROVIDER in .env"
                    f"{Colors.ENDC}\n"
                )

        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Goodbye 👋{Colors.ENDC}\n")
            break

        except Exception as e:
            print(f"\n{Colors.FAIL}[ERROR] {e}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
