# GCP LangChain Testing

LangChain + LangGraph examples using Google Cloud Platform's Vertex AI and RAG Engine.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- GCP project with Vertex AI API enabled
- Service account with the following roles:
  - Vertex AI User (`roles/aiplatform.user`)
  - Storage Object User (`roles/storage.objectUser`)
- Service account key JSON file

## Installation

```bash
# Install dependencies
uv sync
```

## Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` with your values:

```env
# Required: GCP Authentication
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./path/to/service-account-key.json

# Required: Vertex AI Model Configuration
VERTEX_MODEL=gemini-2.0-flash-exp
VERTEX_TEMPERATURE=0.7
VERTEX_MAX_TOKENS=8192

# Optional: RAG Engine (only needed for simple_langgraph_test.py)
RAG_CORPUS=projects/PROJECT/locations/LOCATION/ragCorpora/ID
RAG_TOP_K=5
GCS_BUCKET=your-bucket-name
```

**Important**: Add your service account key to `.gitignore` or store it in a `keys/` directory (already gitignored).

## Usage

### Direct PDF Context Upload

Upload a PDF directly to the LLM context (no RAG required):

```bash
uv run src/upload_direct_into_context.py
```

This encodes a PDF as base64 and sends it directly to Gemini for analysis.

### RAG-based Query System

Query documents using Vertex AI RAG Engine:

```bash
uv run src/simple_langgraph_test.py
```

This demonstrates:
- Question type classification (corpus overview vs. specific query)
- RAG retrieval from Vertex AI corpus
- File upload and import to RAG corpus
- Structured responses using Pydantic models

## Project Structure

```
.
├── src/
│   ├── shared/
│   │   ├── gcp_rag_helpers.py    # RAG corpus management utilities
│   │   └── lc_llm.py              # LLM configuration helper
│   ├── simple_langgraph_test.py   # RAG-based LangGraph example
│   └── upload_direct_into_context.py  # Direct PDF upload example
├── files_to_upload/               # Sample PDF files
├── .env                           # Configuration (gitignored)
└── pyproject.toml                 # Dependencies
```

## GCP Setup

### Service Account Permissions

Your service account needs the following IAM roles:
- **Vertex AI User** (`roles/aiplatform.user`) - for Vertex AI model access
- **Storage Object User** (`roles/storage.objectUser`) - for GCS bucket access

### Create RAG Corpus (Optional)

For RAG functionality, create a corpus in the Vertex AI RAG Engine via the GCP Console and set the `RAG_CORPUS` variable to its full resource name.

## Troubleshooting

**"API key required" error**: Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account key file.

**"Permission denied" error**: Verify the service account has the Vertex AI User role.

**"API not enabled" error**: Enable the Vertex AI API using the `gcloud` command above.
