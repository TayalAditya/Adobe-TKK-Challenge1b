# Persona-Driven Document Intelligence System
## Adobe Challenge 1B: "Connect What Matters — For the User Who Matters"

A sophisticated document analysis system that extracts and prioritizes the most relevant sections from PDF collections based on specific personas and their job-to-be-done requirements. Built for the Adobe Challenge 1B with semantic search capabilities and intelligent content ranking.

## Challenge Overview

This system addresses the core challenge of **persona-driven document intelligence** by:
- Analyzing 3-10 related PDF documents
- Understanding specific user personas and their expertise areas
- Extracting relevant content based on concrete job-to-be-done tasks
- Providing ranked, actionable insights tailored to user needs

## Key Features

### Core Intelligence
- **Semantic Search**: SBERT-based embeddings for deep content understanding
- **Persona-Driven Analysis**: Tailored content extraction based on user role and expertise
- **Smart Section Detection**: Advanced PDF parsing with intelligent title extraction
- **Document Diversity**: Balanced representation across all input documents
- **Quality Ranking**: Relevance scoring and importance ranking of extracted sections

### Technical Excellence
- **CPU-Only Processing**: Optimized for environments without GPU access
- **Lightweight Model**: <1GB SBERT model for efficient processing
- **Fast Processing**: <60 seconds for typical document collections
- **Offline Capability**: No internet access required during execution
- **Docker Ready**: Containerized deployment for consistent execution

## Project Structure

```
Challenge_1b/
├── PDFs/                          # Input PDF documents (7 South of France guides)
│   ├── South of France - Cities.pdf
│   ├── South of France - Cuisine.pdf
│   └── ... (travel guide collection)
├── input/                         # Query JSON files
│   ├── query1.json               # Travel Planner persona
│   └── query2.json               # Additional test case
├── output/                        # Generated analysis results
│   ├── query1.json               # Ranked sections and analysis
│   └── query2.json               # Structured output format
├── local_sbert_model/            # Pre-trained SBERT model (<1GB)
├── nltk_data/                    # NLTK corpus for text processing
├── main.py                       # Core application logic
├── test_system.py               # System validation script
├── approach_explanation.md      # Methodology documentation (300-500 words)
├── EXECUTION_GUIDE.md          # Comprehensive deployment guide
├── Dockerfile                   # Container configuration
├── docker-compose.yml          # Orchestration setup
└── requirements.txt            # Python dependencies
```

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- PDF documents in the `PDFs/` directory
- Query JSON files in the `input/` directory

### Build and Run
```bash
# Build and run the container
docker-compose up --build

# Stop the container
docker-compose down
```

### Validate System
```bash
# Run system validation tests
python test_system.py
```

## Input Format

Create persona-driven queries in the `input/` directory:

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_002",
        "test_case_name": "travel_planner",
        "description": "France Travel"
    },
    "documents": [
        {
            "filename": "South of France - Cities.pdf",
            "title": "South of France - Cities"
        }
    ],
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }
}
```

## Output Format

The system generates structured analysis results:

```json
{
    "metadata": {
        "input_documents": ["doc1.pdf", "doc2.pdf"],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a trip of 4 days...",
        "processing_timestamp": "2025-01-XX..."
    },
    "extracted_sections": [
        {
            "document": "South of France - Cities.pdf",
            "section_title": "Comprehensive Guide to Major Cities",
            "importance_rank": 1,
            "page_number": 1,
            "relevance_score": 0.85
        }
    ],
    "subsection_analysis": [
        {
            "document": "South of France - Cities.pdf",
            "refined_text": "Detailed content analysis...",
            "page_number": 1,
            "relevance_score": 0.85
        }
    ]
}
```

## Sample Test Cases

### Test Case 1: Travel Planning
- **Documents**: 7 South of France travel guides
- **Persona**: Travel Planner
- **Job**: Plan a 4-day trip for 10 college friends
- **Expected Output**: Ranked sections on activities, accommodations, dining, and logistics

### Test Case 2: Academic Research
- **Documents**: Research papers on Graph Neural Networks
- **Persona**: PhD Researcher in Computational Biology
- **Job**: Prepare comprehensive literature review
- **Expected Output**: Methodologies, datasets, and performance benchmarks

### Test Case 3: Business Analysis
- **Documents**: Annual reports from tech companies
- **Persona**: Investment Analyst
- **Job**: Analyze revenue trends and market positioning
- **Expected Output**: Financial metrics, R&D investments, strategic insights

## Configuration

Environment variables (configurable in docker-compose.yml):

```yaml
environment:
  - MODEL_PATH=local_sbert_model     # Path to SBERT model
  - PDF_FOLDER=PDFs                  # PDF documents directory
  - OUTPUT_FOLDER=output             # Output directory
  - TOP_K=5                          # Number of top sections to extract
  - MIN_SECTION_LENGTH=50            # Minimum section length
  - MAX_SECTION_LENGTH=2000          # Maximum section length
```

## Advanced Usage

### Docker Commands

#### Option 1: Docker Compose (Recommended)
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

#### Option 2: Direct Docker Commands
```bash
# Build the image
docker build -t pdf-analyzer .

# Run the container (Windows)
docker run -v "%cd%\PDFs:/app/PDFs" -v "%cd%\input:/app/input" -v "%cd%\output:/app/output" pdf-analyzer

# Run the container (Linux/Mac)
docker run -v "$(pwd)/PDFs:/app/PDFs" -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" pdf-analyzer
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires NLTK data setup)
python main.py

# Run validation tests
python test_system.py
```

## System Architecture

### Processing Pipeline
1. **Document Ingestion**: PDF parsing with pdfplumber
2. **Text Processing**: Cleaning, chunking, and title extraction
3. **Semantic Encoding**: SBERT embeddings generation
4. **Query Enhancement**: Persona and task-aware query construction
5. **Relevance Matching**: Cosine similarity-based content ranking
6. **Diversity Selection**: Multi-document balanced representation
7. **Output Generation**: Structured JSON with metadata and analysis

### Key Algorithms
- **Document Diversity Algorithm**: Ensures balanced representation across all input documents
- **Smart Title Extraction**: Word-level analysis with model-assisted cleaning
- **Chunking Strategy**: Overlapping chunks (1500 chars, 200 char overlap) for context preservation
- **Quality Scoring**: Multi-factor relevance assessment with minimum thresholds

## Performance Metrics

- **Processing Speed**: 30-45 seconds for 7 PDFs
- **Memory Usage**: 2-3GB during processing
- **Model Size**: 384MB (SBERT model)
- **Accuracy**: Semantic relevance scores >0.15
- **Coverage**: Balanced document representation

## Troubleshooting

### Common Issues

1. **Docker build fails**
   - Ensure Docker is running and has sufficient resources
   - Check available disk space (>2GB recommended)

2. **No output generated**
   - Verify PDF files exist in `PDFs/` directory
   - Validate input JSON format with `python test_system.py`
   - Check container logs: `docker-compose logs`

3. **Low relevance scores**
   - Review persona and job_to_be_done descriptions for clarity
   - Ensure PDF content domain matches query requirements

4. **Memory issues**
   - Reduce `TOP_K` value in configuration
   - Decrease `MAX_SECTION_LENGTH` parameter
   - Ensure sufficient system RAM (4GB+ recommended)

### Validation Commands
```bash
# System validation
python test_system.py

# Docker health check
docker-compose ps

# View processing logs
docker-compose logs pdf-analyzer
```

## Technical Constraints

**Challenge Requirements Met**:
- CPU-only processing (no GPU dependencies)
- Model size ≤ 1GB (384MB SBERT model)
- Processing time ≤ 60 seconds for document collections
- No internet access required during execution
- Structured JSON output format compliance

## Success Indicators

After successful execution, you should see:
1. Processing logs for each PDF document
2. Embedding creation progress indicators
3. Output files generated in `output/` directory
4. Validation success messages
5. Structured JSON with persona-driven ranked sections

## Documentation

- **[approach_explanation.md](approach_explanation.md)**: Detailed methodology (300-500 words)


**Ready for evaluation and deployment!** 🚀