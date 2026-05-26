import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("[ERROR] DATABASE_URL not found in environment variables. Please check your .env file.")
    sys.exit(1)

# Globally check and store if pgvector is available on the Postgres server.
# This is set dynamically in verify_pgvector_availability().
HAS_PGVECTOR = True

def verify_pgvector_availability() -> bool:
    """
    Checks if the 'vector' extension is available in the PostgreSQL installation.
    """
    global HAS_PGVECTOR
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # Query if 'vector' is listed in available extensions
            cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
            result = cur.fetchone()
            HAS_PGVECTOR = (result is not None)
    except Exception:
        HAS_PGVECTOR = False
    return HAS_PGVECTOR

# Run initial check
verify_pgvector_availability()

def get_connection():
    """
    Establishes a connection to the PostgreSQL database.
    If pgvector is supported, it registers the pgvector type handler.
    Otherwise, it connects in standard SQL compatibility mode.
    """
    global HAS_PGVECTOR
    try:
        conn = psycopg2.connect(DATABASE_URL)
        
        # If pgvector is installed on the server, we register it
        if HAS_PGVECTOR:
            try:
                from pgvector.psycopg2 import register_vector
                register_vector(conn)
            except Exception as e:
                # Double-check if the extension is enabled in this database
                if "vector type not found" in str(e).lower():
                    try:
                        # Try to enable the extension dynamically
                        temp_conn = psycopg2.connect(DATABASE_URL)
                        temp_conn.autocommit = True
                        with temp_conn.cursor() as cur:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        temp_conn.close()
                        
                        from pgvector.psycopg2 import register_vector
                        register_vector(conn)
                    except Exception:
                        # If enabling fails, pgvector is not installed on the system
                        print("[INFO] pgvector extension not found on server. Falling back to Python Cosine Similarity.")
                        HAS_PGVECTOR = False
                else:
                    raise e
        
        return conn
    except psycopg2.OperationalError as e:
        print(f"\n[CONNECTION ERROR] Could not connect to PostgreSQL at {DATABASE_URL.split('@')[-1]}")
        print("Please verify that:")
        print("1. Your PostgreSQL server is running.")
        print("2. The database 'flaskdb' exists.")
        print(f"3. Credentials (username/password) are correct.\n")
        print(f"Details: {e}")
        raise e

def init_db(embedding_dimension=1536):
    """
    Initializes the database:
    1. If pgvector is present, creates 'documents' with the vector type and HNSW index.
    2. If pgvector is missing, creates 'documents' using standard JSONB for embedding storage.
    """
    global HAS_PGVECTOR
    # Re-verify availability
    verify_pgvector_availability()
    
    print(f"Initializing database... (pgvector support: {HAS_PGVECTOR})")
    try:
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'documents'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        # If it exists, let's verify if the schemas match
        if table_exists:
            cur.execute("""
                SELECT data_type FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'embedding';
            """)
            col_type_res = cur.fetchone()
            
            if col_type_res:
                col_type = col_type_res[0].lower()
                
                # Check for mismatch:
                # - If server has pgvector, but column is standard 'user-defined' (vector) or JSONB
                # - Or we changed dimensions
                mismatch = False
                if HAS_PGVECTOR and col_type != "user-defined":  # User-defined represents pgvector's 'vector' type
                    mismatch = True
                elif not HAS_PGVECTOR and col_type != "jsonb":
                    mismatch = True
                
                # Check dimension if using pgvector
                if HAS_PGVECTOR and not mismatch:
                    cur.execute("""
                        SELECT atttypmod FROM pg_attribute 
                        WHERE attrelid = 'documents'::regclass AND attname = 'embedding';
                    """)
                    result = cur.fetchone()
                    if result and result[0] != embedding_dimension:
                        mismatch = True
                
                if mismatch:
                    print(f"[WARNING] Database Schema Mismatch Detected! Dropping and recreating table...")
                    cur.execute("DROP TABLE IF EXISTS documents CASCADE;")
        
        # Create table depending on pgvector support
        if HAS_PGVECTOR:
            print(f"-> Creating 'documents' table using pgvector (dimension {embedding_dimension})...")
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector({embedding_dimension})
            );
            """
            cur.execute(create_table_query)
            
            # Create HNSW index
            print("-> Creating HNSW vector index for high-speed similarity searches...")
            create_index_query = """
            CREATE INDEX IF NOT EXISTS documents_embedding_cosine_idx 
            ON documents USING hnsw (embedding vector_cosine_ops);
            """
            cur.execute(create_index_query)
        else:
            print("-> Creating 'documents' table using standard JSONB fallback (No pgvector installed)...")
            create_table_query = """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding JSONB
            );
            """
            cur.execute(create_table_query)
        
        cur.close()
        conn.close()
        print("[SUCCESS] Database successfully initialized!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        return False

if __name__ == "__main__":
    init_db()
