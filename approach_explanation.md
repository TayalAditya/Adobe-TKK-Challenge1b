# Persona-Driven Document Intelligence Approach

## Overview
This system implements a semantic search-based approach to extract and prioritize document sections based on persona requirements and job-to-be-done tasks. The solution uses sentence transformers for semantic understanding and implements intelligent text processing to deliver relevant content.

## Core Methodology

### 1. Document Processing Pipeline
- **PDF Extraction**: Uses pdfplumber for robust text extraction with proper handling of formatting and special characters
- **Section Detection**: Implements intelligent title extraction using word-level analysis rather than character-level to improve accuracy
- **Text Chunking**: Splits long sections into overlapping chunks (1500 chars with 200 char overlap) to maintain context while enabling granular search
- **Content Cleaning**: Applies comprehensive text normalization including whitespace handling, encoding fixes, and artifact removal

### 2. Semantic Understanding
- **SBERT Embeddings**: Utilizes a local sentence-transformer model (≤1GB) for generating semantic embeddings of document sections
- **Enhanced Query Construction**: Combines persona role and task description with domain-specific keywords to improve matching accuracy
- **Contextual Matching**: Employs cosine similarity between query and document embeddings to identify semantically relevant content

### 3. Intelligent Ranking and Selection
- **Diversity Algorithm**: Implements document-aware selection to ensure balanced representation across all input documents
- **Relevance Scoring**: Uses semantic similarity scores with minimum threshold (0.15) to filter low-quality matches
- **Multi-pass Selection**: First selects best content from each document, then fills remaining slots with highest-scoring content overall

### 4. Title Enhancement
- **Model-Assisted Cleaning**: Uses NLTK word corpus and custom algorithms to improve extracted section titles
- **Candidate Generation**: Creates multiple title variations through camelCase splitting, compound word detection, and prefix/suffix handling
- **Quality Scoring**: Evaluates title candidates based on readability, semantic coherence, length appropriateness, and word count optimization

### 5. Output Optimization
- **Structured Results**: Generates JSON output with metadata, ranked sections, and refined subsection analysis
- **Content Validation**: Implements quality checks to ensure meaningful content extraction and proper output structure
- **Scalable Processing**: Supports batch processing of multiple query files with individual result generation

## Technical Constraints Compliance
- **CPU-Only Operation**: Uses CPU-optimized sentence transformers without GPU dependencies
- **Model Size**: Local SBERT model under 1GB constraint
- **Performance**: Optimized for <60 second processing time through efficient chunking and caching
- **Offline Capability**: No internet access required during execution, all models and data local

This approach balances semantic understanding with computational efficiency while ensuring diverse, relevant results tailored to specific persona needs and job requirements.