import json
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from db import get_connection, HAS_PGVECTOR


class RAGStorage:
    def __init__(self):
        # Verify that we can connect
        try:
            conn = get_connection()
            conn.close()
        except Exception as e:
            print(
                f"[RAGStorage] WARNING: Could not establish initial database connection: {e}"
            )

    def insert_document(
        self, content: str, embedding: list[float], metadata: dict = None
    ) -> int:
        """
        Inserts a single document chunk, its embedding, and metadata into the database.
        Returns the ID of the newly inserted row.
        """
        query = """
        INSERT INTO documents (content, metadata, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """

        metadata_json = json.dumps(metadata) if metadata else None

        # If pgvector is not available, we serialize the vector to a JSON string
        # to store it in a standard JSONB column.
        embedding_val = embedding if HAS_PGVECTOR else json.dumps(embedding)

        conn = get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(query, (content, metadata_json, embedding_val))
                    doc_id = cur.fetchone()[0]
                    return doc_id
        except Exception as e:
            print(f"[ERROR] Error inserting document: {e}")
            conn.rollback()
            raise e
        finally:
            conn.close()

    def insert_documents_batch(self, documents: list[dict]) -> int:
        """
        Inserts a batch of documents in a single transaction.
        Returns the number of successfully inserted documents.
        """
        query = """
        INSERT INTO documents (content, metadata, embedding)
        VALUES (%s, %s, %s);
        """

        conn = get_connection()
        inserted_count = 0
        try:
            with conn:
                with conn.cursor() as cur:
                    for doc in documents:
                        content = doc.get("content")
                        embedding = doc.get("embedding")
                        metadata = doc.get("metadata")

                        metadata_json = json.dumps(metadata) if metadata else None
                        embedding_val = (
                            embedding if HAS_PGVECTOR else json.dumps(embedding)
                        )

                        cur.execute(query, (content, metadata_json, embedding_val))
                        inserted_count += 1
            return inserted_count
        except Exception as e:
            print(f"[ERROR] Error inserting batch of documents: {e}")
            conn.rollback()
            raise e
        finally:
            conn.close()

    def search_similar_documents(
        self, query_embedding: list[float], limit: int = 3
    ) -> list[dict]:
        """
        Performs a vector cosine similarity search.

        If pgvector is installed:
          Executes Cosine Distance '<=>' in SQL.

        If pgvector is NOT installed:
          Fetches records and calculates Cosine Similarity in Python using NumPy.
        """
        if HAS_PGVECTOR:
            # High-performance SQL pgvector implementation
            query = """
            SELECT 
                id, 
                content, 
                metadata, 
                (1.0 - (embedding <=> %s)) AS similarity
            FROM documents
            ORDER BY embedding <=> %s ASC
            LIMIT %s;
            """

            conn = get_connection()
            try:
                with conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(query, (query_embedding, query_embedding, limit))
                        results = cur.fetchall()

                        formatted_results = []
                        for row in results:
                            formatted_results.append(
                                {
                                    "id": row["id"],
                                    "content": row["content"],
                                    "metadata": (
                                        row["metadata"] if row["metadata"] else {}
                                    ),
                                    "similarity": float(row["similarity"]),
                                }
                            )
                        return formatted_results
            except Exception as e:
                print(f"[ERROR] Error searching similar documents with pgvector: {e}")
                raise e
            finally:
                conn.close()
        else:
            # Standard SQL fallback with CPU-based Python vector calculation
            query = """
            SELECT id, content, metadata, embedding 
            FROM documents;
            """

            conn = get_connection()
            try:
                with conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(query)
                        rows = cur.fetchall()

                        # Prepare vectors and calculate similarity
                        formatted_results = []
                        q_vec = np.array(query_embedding)
                        q_norm = np.linalg.norm(q_vec)

                        for row in rows:
                            doc_id = row["id"]
                            content = row["content"]
                            metadata = row["metadata"] if row["metadata"] else {}

                            # psycopg2 automatically deserializes JSONB columns into Python lists
                            doc_emb = row["embedding"]
                            if isinstance(doc_emb, str):
                                doc_emb = json.loads(doc_emb)

                            d_vec = np.array(doc_emb)
                            d_norm = np.linalg.norm(d_vec)

                            # Calculate Cosine Similarity = dot(A, B) / (norm(A) * norm(B))
                            if q_norm > 0 and d_norm > 0:
                                similarity = np.dot(q_vec, d_vec) / (q_norm * d_norm)
                            else:
                                similarity = 0.0

                            formatted_results.append(
                                {
                                    "id": doc_id,
                                    "content": content,
                                    "metadata": metadata,
                                    "similarity": float(similarity),
                                }
                            )

                        # Sort by similarity score descending (highest similarity first)
                        formatted_results.sort(
                            key=lambda x: x["similarity"], reverse=True
                        )

                        # Return the top K items
                        return formatted_results[:limit]

            except Exception as e:
                print(f"[ERROR] Error performing fallback similarity search: {e}")
                raise e
            finally:
                conn.close()

    def get_document_count(self) -> int:
        """Returns the total number of documents in the database."""
        conn = get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents;")
                    return cur.fetchone()[0]
        except Exception:
            return 0
        finally:
            conn.close()

    def clear_all_documents(self):
        """Truncates the documents table. Useful for testing."""
        conn = get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
            print("[INFO] Cleared all documents from the database.")
        except Exception as e:
            print(f"[ERROR] Error clearing database: {e}")
        finally:
            conn.close()


# global class
if __name__ == "__main__":
    # Small test
    storage = RAGStorage()
    print("RAG Storage ready.")
