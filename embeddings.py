import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class EmbeddingManager:
    def __init__(self):
        # Read provider choice from .env. Default to 'local' for a 100% free & offline experience.
        self.provider = os.getenv("EMBEDDING_PROVIDER", "local").lower().strip()
        self.embeddings_model = None
        self.dimension = 1536  # Default dimension, adjusted below

        print(f"\n[CONFIG] Configuring Embedding Engine: Provider = '{self.provider}'")

        if self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key or "openai_api_key_here" in self.api_key.lower():
                print(
                    "[WARNING] No OpenAI API key found. Defaulting to 'local' provider."
                )
                self.provider = "local"
            else:
                try:
                    from langchain_openai import OpenAIEmbeddings

                    self.embeddings_model = OpenAIEmbeddings(
                        model="text-embedding-3-small", openai_api_key=self.api_key
                    )
                    self.dimension = 1536
                    print("[SUCCESS] OpenAI Embedding engine active (1536 Dims).")
                except ImportError:
                    print(
                        "[WARNING] Could not import 'langchain-openai'. Falling back to 'local' provider."
                    )
                    self.provider = "local"

        if self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key or "gemini_api_key_here" in self.api_key.lower():
                print(
                    "[WARNING] No GEMINI_API_KEY found in .env. Defaulting to 'local' provider."
                )
                self.provider = "local"
            else:
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings

                    # Google's text-embedding-004 is free and produces 768 dimensions
                    self.embeddings_model = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004", google_api_key=self.api_key
                    )
                    self.dimension = 768
                    print(
                        "[SUCCESS] Google Gemini Embedding engine active (768 Dims - Free Developer Tier)."
                    )
                except ImportError:
                    print(
                        "[WARNING] Could not import 'langchain-google-genai'. Falling back to 'local' provider."
                    )
                    self.provider = "local"

        if self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer

                print(
                    "[INFO] Loading local SentenceTransformer model 'BAAI/bge-base-en-v1.5' (100% Free & Offline)..."
                )
                # This is a highly efficient 384-dimensional model running on your local CPU/GPU
                self.embeddings_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
                self.dimension = 384
                print(
                    "[SUCCESS] Local SentenceTransformer engine active (384 Dims - Free & Offline)."
                )
            except ImportError:
                print(
                    "[WARNING] 'sentence-transformers' not installed. Run 'pip install -r requirements.txt'"
                )
                print(
                    "[WARNING] Falling back to zero-dependency 'mock' embedding provider."
                )
                self.provider = "mock"

        if self.provider == "mock":
            self.dimension = 1536
            print(
                "[SUCCESS] Zero-Dependency Mock embedding engine active (1536 Dims - Offline Simulator)."
            )

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for the input text based on the selected provider.
        Returns a list of floats (vector).
        """
        try:
            if self.provider == "openai" and self.embeddings_model:
                return self.embeddings_model.embed_query(text)

            elif self.provider == "gemini" and self.embeddings_model:
                return self.embeddings_model.embed_query(text)

            elif self.provider == "local" and self.embeddings_model:
                # sentence-transformers returns a numpy array, convert to list
                vector = self.embeddings_model.encode(text)
                return vector.tolist()

            else:
                return self._generate_deterministic_mock_embedding(text)

        except Exception as e:
            print(
                f"[WARNING] Embedding generation failed with provider '{self.provider}': {e}. Using mock generator."
            )
            return self._generate_deterministic_mock_embedding(text)

    def _generate_deterministic_mock_embedding(self, text: str) -> list[float]:
        """
        Generates a deterministic unit vector based on the text hash.
        Used as a robust zero-cost, zero-dependency simulator.
        """
        text_hash = abs(hash(text))
        rng = np.random.default_rng(text_hash)
        vector = rng.standard_normal(self.dimension)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def get_dimension(self) -> int:
        """Returns the dimension of the embedding vectors."""
        return self.dimension


if __name__ == "__main__":
    manager = EmbeddingManager()
    test_text = "PostgreSQL is a relational database."
    vector = manager.get_embedding(test_text)
    print(f"Generated a {len(vector)}-dimensional vector successfully!")
