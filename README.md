# 🧠 Learn RAG: PostgreSQL + pgvector RAG Pipeline (Full-Stack Web App)

Welcome! This is an educational, production-ready **Retrieval-Augmented Generation (RAG)** pipeline designed to teach you how to build modern AI semantic applications using **PostgreSQL** as a vector database.

This project is optimized to run **100% free** and supports both **cloud-free developer APIs** (like Google Gemini) and **completely offline local models** (Hugging Face embeddings + Ollama LLM) running directly on your CPU/GPU!

Now includes a full-stack web application with React frontend and FastAPI backend.

---

## 🏗️ Project Architecture

A RAG pipeline bridges the gap between **static LLM knowledge** and **dynamic, private, or real-time data stores**. It functions in two stages:

```
[ Ingestion Stage ]
1. Text Chunks ──> 2. Embedding Model (Local/Cloud) ──> 3. pgvector (PostgreSQL)

[ Retrieval & Generation Stage ]
Query ──> Embed Query ──> Cosine Distance Search ──> Context Chunks ──> LLM Prompt ──> Grounded Answer
```

1. **Ingestion**: Raw text is chunked into cohesive paragraphs, vectorized using an embedding model (like a local Hugging Face transformer or Gemini's free API), and inserted into a PostgreSQL table with a vector column.
2. **Retrieval**: The user asks a question. The pipeline vectorizes this question and uses a database query to select the top $K$ documents with the smallest **Cosine Distance** (highest semantic similarity) to the query.
3. **Generation**: The retrieved document chunks are formatted into a prompt as *context*. The prompt is sent to an LLM (e.g., local Ollama model or Google's free Gemini API) which extracts the answer solely based on the provided facts.

---

## 📂 Project Tour

Inside your `c:\GenAI-RAG` folder, you will find:

*   **`db.py`**: Handles connecting to PostgreSQL. It automatically:
    *   Enables the `vector` extension.
    *   Registers the pgvector data-type handler with Python's PostgreSQL driver (`psycopg2`).
    *   Creates a `documents` table storing text `content`, a JSONB `metadata` field, and a vector column of the correct dimension.
    *   **Dimension Protection**: Automatically detects if you change your embedding configuration in `.env` (e.g. from 1536 dims to 384 dims) and resets the database schema safely, preventing connection errors!
    *   Creates a **Hierarchical Navigable Small World (HNSW)** index to speed up similarity lookups.
*   **`embeddings.py`**: Supports dynamic embedding generation. You can configure:
    *   `local`: Uses Hugging Face's `sentence-transformers/BAAI/bge-base-en-v1.5` (384 dimensions, runs locally on your CPU/GPU, 100% free and offline).
    *   `gemini`: Uses Google Gemini's `text-embedding-004` (768 dimensions, 100% free developer tier).
    *   `openai`: Uses paid OpenAI `text-embedding-3-small` (1536 dimensions).
    *   `mock`: A zero-dependency offline vector math simulator (1536 dimensions).
*   **`rag_storage.py`**: Implements high-performance operations on your database, including batch insertions (`INSERT INTO ...`) and semantic searches using pgvector's cosine distance operator (`<=>`).
*   **`api/`**: Contains the FastAPI backend application:
    *   `api.py`: Main FastAPI app with CORS middleware and route definitions.
    *   `routes/chat.py`: Chat endpoint that handles RAG logic.
    *   `services/rag_service.py`: (If exists) Additional RAG service logic.
*   **`client/`**: Contains the React frontend application:
    *   `src/App.jsx`: Main chat interface component.
    *   `src/main.jsx`: Entry point for the React app.
    *   `package.json`: Defines frontend dependencies and scripts.
*   **`main.py`**: The interactive command-line orchestrator (original CLI version). Run this to ingest files, ask questions, perform database searches, see the constructed LLM prompt, and print grounded chat responses.
*   **`.env.example`**: Template for environment variables (copy to `.env` and fill in your secrets).
*   **`requirements.txt`**: List of Python package dependencies.
*   **`client/package.json`**: List of JavaScript package dependencies.

---

## ⚙️ Setup & Execution Guide

### 1. Prerequisite: Start PostgreSQL
Make sure you have PostgreSQL running. As per the default configuration, it should be running on `localhost:5432` with a database named `flaskdb` and login `postgres:admin`.
*   *Note: Ensure the `pgvector` extension is installed. If you installed PostgreSQL on Windows via StackBuilder or modern installer packages, it is already pre-packaged! If you're using Docker, you can run:*
    ```bash
    docker run --name pgvector-db -e POSTGRES_PASSWORD=admin -e POSTGRES_DB=flaskdb -p 5432:5432 -d pgvector/pgvector:pg16
    ```

### 2. Set Up Python Virtual Environment (for backend)
We recommend utilizing a virtual environment:
```powershell
# Create a virtual environment
py -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### 3. Install Backend Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```powershell
cd client
npm install
cd ..
```

### 5. Configure Environment Variables
Copy the example environment files and fill in your values:

#### Backend (api/.env)
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgres://user:password@host:port/dbname
EMBEDDING_PROVIDER=local  # options: local, openai, gemini
```

#### Frontend (client/.env)
```env
VITE_API_URL=http://localhost:8000  # Change to your deployed backend URL in production
```

> **Important**: The actual `.env` files are ignored by Git (see `.gitignore`). Never commit your real secrets. Use the `.env.example` files as templates.

### 6. Choose Your Free Architecture in `api/.env`

#### Option A: Free Cloud RAG (Highly Recommended - No Credit Card Needed)
Uses local CPU-based Hugging Face embeddings and Google Gemini's free API for text generation (which gives you 15 requests per minute completely free).
1. Go to [Google AI Studio](https://aistudio.google.com/) and click **Get API Key** to generate a key in 10 seconds.
2. Edit `api/.env` to set:
    ```env
    EMBEDDING_PROVIDER=local
    LLM_PROVIDER=gemini
    GEMINI_API_KEY=your_copied_gemini_api_key
    ```

#### Option B: 100% Offline RAG (No Internet/Zero Keys)
Runs both embeddings and generation entirely on your own local computer!
1. Download and install [Ollama](https://ollama.com/) (free open-source local LLM runner for Windows).
2. Open a command prompt and download a fast, lightweight model (like Gemma 2B):
    ```bash
    ollama run gemma2:2b
    ```
3. Edit `api/.env` to set:
    ```env
    EMBEDDING_PROVIDER=local
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=gemma2:2b
    ```

---

## ▶️ Running the Application Locally

### Start the Backend (FastAPI)
```powershell
# From the project root
uvicorn api.api:app --reload --port 8000
```
The backend will be available at `http://localhost:8000`

### Start the Frontend (React)
```powershell
# From the project root
cd client
npm run dev
```
The frontend will be available at `http://localhost:5173` (or another port if 5173 is in use)

### Using the CLI Version (Optional)
```powershell
python main.py
```

---

## 🚀 Deployment to Render

### Backend Deployment
1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Configure:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.api:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add:
   - `GEMINI_API_KEY`: (your actual key)
   - `DATABASE_URL`: (your PostgreSQL connection string from Render's managed PostgreSQL or external)
   - `EMBEDDING_PROVIDER`: `local` (or your preferred provider)
5. Deploy! Render will provide a URL like `https://your-service.onrender.com`

### Frontend Deployment
1. Create a new **Static Site** on Render
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
4. Under **Environment**, add:
   - `VITE_API_URL`: (your Render backend URL, e.g., `https://your-service.onrender.com`)
5. Deploy! Render will provide a URL for your frontend.

### Important Notes for Render
- The frontend makes requests to the backend using the `VITE_API_URL` environment variable
- Ensure your backend's CORS middleware allows the frontend's origin (Render's default CORS setup in `api.py` allows all origins with `"*"`, which is fine for development but consider restricting in production)
- The free tier of Render may spin down services after periods of inactivity - a "warm up" request may be needed

---

## 🔬 Mathematical Breakdown of pgvector Queries

In our search query inside `rag_storage.py`, you'll see:
```sql
SELECT content, (1.0 - (embedding <=> %s)) AS similarity
FROM documents
ORDER BY embedding <=> %s ASC;
```

*   **`<=>` Operator**: In pgvector, `<=>` denotes **Cosine Distance**, which is defined as:
    $$\text{Cosine Distance}(A, B) = 1.0 - \frac{A \cdot B}{\|A\| \|B\|}$$
*   **Cosine Similarity**: Since distance decreases as vectors get more similar, sorting by Cosine Distance in **ASCENDING** order (`ASC`) retrieves the most semantically related chunks.
*   **HNSW Index**: The Hierarchical Navigable Small World index performs Approximate Nearest Neighbor (ANN) search, forming a multi-layered graph of vectors. Searching this graph has $O(\log N)$ time complexity rather than performing an exhaustive $O(N)$ table scan, which is critical when scaling to millions of documents!

---

## 📚 Learning Resources
- [Understanding Retrieval-Augmented Generation (RAG)](https://lilianweng.github.io/posts/2020-10-/retrieval-augmented-generation/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Google Gemini API](https://ai.google.dev/)
- [Ollama - Local LLMs](https://ollama.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

*Happy coding! 🚀*