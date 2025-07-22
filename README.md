# PDF Document Analysis System

A system that analyzes PDF documents using semantic search and natural language processing to extract relevant information based on user queries.

## Features

- PDF text extraction with smart section detection
- Semantic search using SBERT embeddings
- Intelligent text chunking for better search results
- Document diversity in search results
- Smart title extraction and cleaning
- Multiple query file processing
- Docker support for easy deployment
- Configurable environment variables

## Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local development)
- At least 4GB RAM recommended
- PDF files to analyze

## Directory Structure

```
adobe/
├── PDFs/                  # Place your PDF documents here
├── input/                 # Place your query JSON files here
├── output/               # Generated results will be saved here
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── README.md          # This file
```

## Query JSON Format

Create your query files in the `input` directory using this format:

```json
{
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip..."
    },
    "documents": [
        {
            "filename": "document.pdf",
            "title": "Document Title"
        }
    ]
}
```

# PDF Document Analysis System

// ...existing code...

## Docker Setup

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and run
docker-compose up --build

# Stop container
docker-compose down
```

### Option 2: Using Docker Directly
```bash
# Build the image
docker build -t pdf-analyzer .

# Run the container
docker run \
  -v "$(pwd)/PDFs:/app/PDFs" \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -e MODEL_PATH=local_sbert_model \
  -e PDF_FOLDER=PDFs \
  -e QUERY_JSON=input/query1.json \
  -e OUTPUT_FOLDER=output \
  -e TOP_K=5 \
  -e MIN_SECTION_LENGTH=50 \
  -e MAX_SECTION_LENGTH=2000 \
  pdf-analyzer

# Stop the container
docker stop $(docker ps -q --filter ancestor=pdf-analyzer)
```

// ...existing code...