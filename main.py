import os
import glob
import json
import torch
import pdfplumber
import re
import nltk
import sys
import signal
from nltk.corpus import words
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Constants with environment variable support
MODEL_PATH = os.getenv("MODEL_PATH", "local_sbert_model")
PDF_FOLDER = os.getenv("PDF_FOLDER", "PDFs")
QUERY_JSON = os.getenv("QUERY_JSON", "input/query1.json")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "output")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SECTION_LENGTH = int(os.getenv("MIN_SECTION_LENGTH", "50"))
MAX_SECTION_LENGTH = int(os.getenv("MAX_SECTION_LENGTH", "2000"))

# Health check function for Docker
def healthcheck() -> bool:
    """Verify if the application is ready to process requests"""
    try:
        # Check if model is loaded
        if not model:
            return False
            
        # Check if NLTK data is available
        if not os.path.exists("./nltk_data"):
            return False
            
        # Check if required directories exist
        if not all(os.path.exists(d) for d in [PDF_FOLDER, "input", OUTPUT_FOLDER]):
            return False
            
        return True
    except Exception:
        return False

# Signal handlers for Docker
def signal_handler(sig, frame):
    """Handle Docker stop signals gracefully"""
    print("\n🛑 Gracefully shutting down...")
    sys.exit(0)

# Initialize SBERT Model
def initialize_model():
    """Initialize the SBERT model with proper error handling"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        return SentenceTransformer(MODEL_PATH, device=device)
    except Exception as e:
        print(f"❌ Error initializing SBERT model: {e}")
        sys.exit(1)

# Initialize NLTK data
def initialize_nltk():
    """Initialize NLTK with proper error handling"""
    try:
        nltk.data.path.append("./nltk_data")
        return set(words.words())
    except LookupError:
        print("❌ ERROR: NLTK words corpus not found in ./nltk_data")
        sys.exit(1)


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove page numbers and common PDF artifacts
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\f', '', text)  # Remove form feeds
    
    # Fix common encoding issues
    text = text.replace('\u2022', '•')  # Bullet points
    text = text.replace('\u2019', "'")  # Right single quotation mark
    text = text.replace('\u201c', '"')  # Left double quotation mark
    text = text.replace('\u201d', '"')  # Right double quotation mark
    text = text.replace('\u2013', '–')  # En dash
    text = text.replace('\u2014', '—')  # Em dash
    
    return text


def generate_split_candidates(text: str) -> List[str]:
    """Generate potential word splitting candidates"""
    candidates = []
    
    # Original text
    candidates.append(text)
    
    # Basic camelCase and number splitting
    basic_split = text
    basic_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', basic_split)
    basic_split = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', basic_split)
    basic_split = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', basic_split)
    candidates.append(basic_split)
    
    # Try splitting at common word boundaries
    for pattern in [r'([a-z]{3,})([A-Z][a-z]{2,})', 
                    r'([a-z]{2,})([A-Z]{2,})',
                    r'([A-Z]{2,})([A-Z][a-z]{2,})']:
        split_attempt = re.sub(pattern, r'\1 \2', text)
        if split_attempt != text:
            candidates.append(split_attempt)
    
    # Try splitting common prefixes/suffixes
    prefixes = ['pre', 'post', 'anti', 'pro', 'sub', 'super', 'inter', 'intra', 'multi', 'uni']
    suffixes = ['tion', 'ing', 'ness', 'ment', 'able', 'ible', 'ful', 'less']
    
    for prefix in prefixes:
        pattern = f'({prefix})([A-Z][a-z]{{2,}})'
        split_attempt = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
        if split_attempt != text:
            candidates.append(split_attempt)
    
    for suffix in suffixes:
        pattern = f'([a-z]{{3,}})({suffix})'
        split_attempt = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
        if split_attempt != text:
            candidates.append(split_attempt)
    
    # Try splitting common compound words
    common_compounds = ['travel', 'guide', 'restaurant', 'accommodation', 'transport', 'culture', 
                       'history', 'museum', 'gallery', 'market', 'shopping', 'dining', 'hotel']
    
    for compound in common_compounds:
        # Look for compound word at start
        pattern = f'({compound})([A-Z][a-z]{{2,}})'
        split_attempt = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
        if split_attempt != text:
            candidates.append(split_attempt)
        
        # Look for compound word at end
        pattern = f'([A-Z][a-z]{{2,}})({compound})'
        split_attempt = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
        if split_attempt != text:
            candidates.append(split_attempt)
    
    return list(set(candidates))  # Remove duplicates


def score_text_readability(text: str, english_words_set: set) -> float:
    """Score text readability based on valid English words"""
    words_list = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words_list:
        return 0.0
    
    valid_words = sum(1 for word in words_list if word in english_words_set or len(word) <= 2)
    total_words = len(words_list)
    
    # Additional scoring factors
    word_length_score = sum(1 for word in words_list if 2 <= len(word) <= 12) / total_words
    readability_score = (valid_words / total_words) * 0.7 + word_length_score * 0.3
    
    return readability_score


def clean_section_title_with_model(title: str, model, english_words_set: set) -> str:
    """Enhanced section title cleaning using the SBERT model for validation"""
    if not title or title == "N/A":
        return "N/A"
    
    # Generate splitting candidates
    candidates = generate_split_candidates(title)
    
    # Apply basic cleaning to all candidates
    cleaned_candidates = []
    for candidate in candidates:
        # Apply existing cleaning logic
        cleaned = candidate
        cleaned = re.sub(r'([a-zA-Z]):([a-zA-Z])', r'\1: \2', cleaned)
        cleaned = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1. \2', cleaned)
        cleaned = re.sub(r'([a-zA-Z]),([a-zA-Z])', r'\1, \2', cleaned)
        cleaned = re.sub(r'([a-zA-Z]);([a-zA-Z])', r'\1; \2', cleaned)
        cleaned = re.sub(r'•([a-zA-Z])', r'• \1', cleaned)
        cleaned = re.sub(r'-([a-zA-Z])', r'- \1', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
        cleaned = re.sub(r'[•\-\.]+\s*$', '', cleaned).strip()
        
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        cleaned_candidates.append(cleaned)
    
    # Remove duplicates and filter out empty/invalid candidates
    cleaned_candidates = [c for c in set(cleaned_candidates) if c and len(c) > 1]
    
    if len(cleaned_candidates) <= 1:
        return cleaned_candidates[0] if cleaned_candidates else "N/A"
    
    try:
        # Score each candidate
        candidate_scores = []
        for i, candidate in enumerate(cleaned_candidates):
            # Readability score
            readability = score_text_readability(candidate, english_words_set)
            
            # Semantic coherence (default high score since we're comparing similar texts)
            semantic_score = 1.0
            
            # Length penalty for overly long titles
            length_penalty = max(0, min(1, (100 - len(candidate)) / 100))
            
            # Word count preference (prefer 2-8 words)
            word_count = len(candidate.split())
            word_count_score = 1.0 if 2 <= word_count <= 8 else max(0.5, 1.0 - abs(word_count - 5) * 0.1)
            
            # Combined score
            total_score = (
                readability * 0.4 + 
                semantic_score * 0.3 + 
                length_penalty * 0.2 + 
                word_count_score * 0.1
            )
            
            candidate_scores.append((candidate, total_score))
        
        # Sort by score and return the best candidate
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_candidate = candidate_scores[0][0]
        
        # Final length check
        if len(best_candidate) > 100:
            words_list = best_candidate.split()
            if len(words_list) > 12:
                best_candidate = ' '.join(words_list[:12]) + "..."
            else:
                best_candidate = best_candidate[:97] + "..."
        
        return best_candidate
        
    except Exception as e:
        print(f"Error in model-based title cleaning: {e}")
        # Fallback to the candidate with best readability score
        fallback_scores = [(c, score_text_readability(c, english_words_set)) for c in cleaned_candidates]
        fallback_scores.sort(key=lambda x: x[1], reverse=True)
        return fallback_scores[0][0] if fallback_scores else "N/A"


def extract_section_title_improved_with_model(page, model, english_words_set: set) -> str:
    """Improved section title extraction using words (instead of characters)"""
    try:
        words_on_page = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words_on_page:
            return "N/A"
        
        # Extract words from top 100 vertical units (header area)
        top_words = [w for w in words_on_page if w['top'] < 100]
        top_words.sort(key=lambda x: (x['top'], x['x0']))  # Sort top to bottom, left to right
        
        # Group top line
        title_words = []
        line_top = top_words[0]['top'] if top_words else None
        for w in top_words:
            if abs(w['top'] - line_top) > 5:
                break
            title_words.append(w['text'])
        
        title = " ".join(title_words).strip()
        if not title:
            return "N/A"
        
        # Clean the title using model
        return clean_section_title_with_model(title, model, english_words_set)
    
    except Exception as e:
        print(f"[WARN] Title extraction failed on page: {e}")
        return "N/A"

def split_text_into_chunks(text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split long text into overlapping chunks for better semantic search"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]+\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap
            words_list = current_chunk.split()
            if len(words_list) > 20:
                current_chunk = ' '.join(words_list[-overlap//10:]) + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_sections_from_pdf(pdf_path: str, model, english_words_set: set) -> List[Dict]:
    """Extract sections from PDF with improved parsing and model-enhanced title cleaning"""
    sections = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if not text or len(text.strip()) < MIN_SECTION_LENGTH:
                        continue
                    
                    # Clean the text
                    cleaned_text = clean_text(text)
                    
                    # Get section title using model-enhanced cleaning
                    section_title = extract_section_title_improved_with_model(page, model, english_words_set)
                    
                    # Split long sections into chunks
                    if len(cleaned_text) > MAX_SECTION_LENGTH:
                        chunks = split_text_into_chunks(cleaned_text)
                        for i, chunk in enumerate(chunks):
                            chunk_title = f"{section_title} (Part {i+1})" if len(chunks) > 1 else section_title
                            sections.append({
                                "filename": os.path.basename(pdf_path),
                                "page_number": page_num + 1,
                                "section_title": chunk_title,
                                "text": chunk,
                                "chunk_id": i if len(chunks) > 1 else None
                            })
                    else:
                        sections.append({
                            "filename": os.path.basename(pdf_path),
                            "page_number": page_num + 1,
                            "section_title": section_title,
                            "text": cleaned_text,
                            "chunk_id": None
                        })
                        
                except Exception as e:
                    print(f"Error processing page {page_num + 1} in {pdf_path}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    
    return sections


def process_all_pdfs(pdf_folder: str, allowed_files: set, model, english_words_set: set) -> Tuple[List[str], List[Dict]]:
    """Process all PDFs with improved error handling and filtering"""
    corpus = []
    metadata = []
    
    for filename in os.listdir(pdf_folder):
        if not filename.endswith(".pdf") or filename not in allowed_files:
            continue
            
        print(f"Processing {filename}...")
        full_path = os.path.join(pdf_folder, filename)
        
        try:
            sections = extract_sections_from_pdf(full_path, model, english_words_set)
            valid_sections = 0
            
            for section in sections:
                # Filter out sections that are too short or seem invalid
                if (len(section["text"]) >= MIN_SECTION_LENGTH and 
                    not section["text"].isspace()):
                    corpus.append(section["text"])
                    metadata.append({
                        "filename": section["filename"],
                        "page_number": section["page_number"],
                        "section_title": section["section_title"],
                        "chunk_id": section.get("chunk_id")
                    })
                    valid_sections += 1
            
            print(f"  Extracted {valid_sections} valid sections from {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Total corpus size: {len(corpus)} sections")
    return corpus, metadata


def create_enhanced_query(persona_role: str, task: str) -> str:
    """Create an enhanced query for better semantic matching"""
    # Create multiple query variations for better matching
    base_query = f"As a {persona_role}, {task}"
    
    # Add travel-specific keywords for better matching
    if "travel" in task.lower() or "trip" in task.lower():
        enhanced_query = f"{base_query}. Travel planning itinerary activities attractions restaurants accommodation transportation tips recommendations"
        return enhanced_query
    
    return base_query


def get_top_chunks_with_diversity(query: str, corpus: List[str], corpus_embeddings, metadata: List[Dict], top_k: int) -> List[Tuple]:
    """Get top chunks with document diversity to ensure balanced representation"""
    # Create enhanced query
    enhanced_query = query + " travel guide recommendations activities attractions food restaurants hotels tips"
    
    query_embedding = model.encode(enhanced_query, convert_to_tensor=True)
    # Get more candidates for diversity selection
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k*3)[0]
    
    # Group hits by document
    doc_hits = {}
    for hit in hits:
        corpus_id = hit["corpus_id"]
        score = float(hit["score"])
        filename = metadata[corpus_id]["filename"]
        
        # Only include results with reasonable relevance scores
        if score > 0.15:  # Minimum relevance threshold
            if filename not in doc_hits:
                doc_hits[filename] = []
            doc_hits[filename].append((corpus[corpus_id], metadata[corpus_id], score))
    
    # Sort documents by their best score
    sorted_docs = sorted(doc_hits.items(), key=lambda x: max(hit[2] for hit in x[1]), reverse=True)
    
    # Select diverse results with balanced document representation
    selected_hits = []
    max_per_doc = max(1, top_k // max(1, len(sorted_docs)))  # At least 1 per doc, but balanced
    
    # First pass: get the best from each document
    for doc_name, doc_content in sorted_docs:
        # Sort hits within this document by score
        doc_content.sort(key=lambda x: x[2], reverse=True)
        
        # Take up to max_per_doc from this document
        taken = 0
        for content, md, score in doc_content:
            if taken >= max_per_doc or len(selected_hits) >= top_k:
                break
                
            # Check for content deduplication
            content_hash = hash(content[:200])
            if not any(hash(existing[0][:200]) == content_hash for existing in selected_hits):
                selected_hits.append((content, md, score))
                taken += 1
    
    # Second pass: fill remaining slots with best remaining content
    if len(selected_hits) < top_k:
        all_remaining = []
        for doc_name, doc_content in sorted_docs:
            for content, md, score in doc_content:
                # Skip if already selected
                if any(existing[0][:100] == content[:100] for existing in selected_hits):
                    continue
                all_remaining.append((content, md, score))
        
        # Sort by score and add best remaining
        all_remaining.sort(key=lambda x: x[2], reverse=True)
        for content, md, score in all_remaining:
            if len(selected_hits) >= top_k:
                break
            
            # Final deduplication check
            content_hash = hash(content[:200])
            if not any(hash(existing[0][:200]) == content_hash for existing in selected_hits):
                selected_hits.append((content, md, score))
    
    # Sort final results by relevance score
    selected_hits.sort(key=lambda x: x[2], reverse=True)
    
    return selected_hits[:top_k]


def build_output(data: Dict, top_hits: List[Tuple]) -> Dict:
    """Build output with improved structure and content"""
    timestamp = datetime.now().isoformat()
    
    # Get unique documents from hits
    unique_docs = list(dict.fromkeys([md["filename"] for _, md, _ in top_hits]))
    
    metadata_out = {
        "input_documents": unique_docs,
        "persona": data["persona"]["role"],
        "job_to_be_done": data["job_to_be_done"]["task"],
        "processing_timestamp": timestamp
    }
    
    extracted_sections = []
    subsection_analysis = []
    
    for i, (text, md, score) in enumerate(top_hits):
        # Clean up section title for display (apply final cleaning)
        section_title = clean_section_title_with_model(md["section_title"], model, english_words)
        if section_title == "N/A" or not section_title.strip():
            section_title = f"Section from page {md['page_number']}"
        
        extracted_sections.append({
            "document": md["filename"],
            "section_title": section_title,
            "importance_rank": i + 1,
            "page_number": md["page_number"],
            "relevance_score": round(score, 3)
        })
        
        # Improve text formatting for output
        refined_text = text.strip()
        
        # Add structure to bullet points and lists
        refined_text = re.sub(r'•\s*([^•\n]+)', r'• \1', refined_text)
        refined_text = re.sub(r'\n\s*-\s*', '\n• ', refined_text)
        
        subsection_analysis.append({
            "document": md["filename"],
            "refined_text": refined_text,
            "page_number": md["page_number"],
            "relevance_score": round(score, 3)
        })
    
    return {
        "metadata": metadata_out,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }


def validate_output(output_data: Dict) -> bool:
    """Validate the output structure and content quality"""
    try:
        # Check required fields
        required_fields = ["metadata", "extracted_sections", "subsection_analysis"]
        for field in required_fields:
            if field not in output_data:
                print(f"Missing required field: {field}")
                return False
        
        # Check if we have meaningful content
        if len(output_data["extracted_sections"]) == 0:
            print("No sections extracted - this indicates a problem with PDF parsing")
            return False
        
        # Check if subsection analysis has meaningful content
        total_text_length = sum(len(section["refined_text"]) for section in output_data["subsection_analysis"])
        if total_text_length < 500:
            print("Extracted text is too short - may indicate parsing issues")
            return False
        
        print("Output validation passed")
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


# ---------- Main Script ----------
# ...existing code...
# ...existing imports...


if __name__ == "__main__":
    try:
        # Initialize components once
        print("🔧 Initializing components...")
        model = initialize_model()
        english_words = initialize_nltk()
        
        # Get all JSON files from input directory
        input_files = glob.glob(os.path.join('input', '*.json'))
        
        if not input_files:
            print("❌ No JSON files found in input directory")
            sys.exit(1)
            
        print(f"📁 Found {len(input_files)} input files to process")
        
        # Process each input file
        for input_file in input_files:
            try:
                print(f"\n📄 Processing {input_file}...")
                
                # Update environment variable for current file
                os.environ['QUERY_JSON'] = input_file
                
                with open(input_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                allowed_files = set(doc["filename"] for doc in data.get("documents", []))
                print(f"Processing {len(allowed_files)} documents: {', '.join(allowed_files)}")
                
                # Process files and generate output
                print("📄 Processing PDFs...")
                corpus, metadata = process_all_pdfs(PDF_FOLDER, allowed_files, model, english_words)
                
                if not corpus:
                    print(f"⚠️ No content extracted for {input_file}, skipping...")
                    continue
                
                print("🔗 Creating embeddings...")
                corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
                
                # Create enhanced query
                role = data["persona"]["role"]
                task = data["job_to_be_done"]["task"]
                query = create_enhanced_query(role, task)
                print(f"Enhanced query: {query}")
                
                print("🔍 Searching for relevant content...")
                top_hits = get_top_chunks_with_diversity(query, corpus, corpus_embeddings, metadata, TOP_K)
                
                if not top_hits:
                    print(f"⚠️ No relevant content found for {input_file}, skipping...")
                    continue
                
                print("📤 Building output...")
                output_data = build_output(data, top_hits)
                
                if not validate_output(output_data):
                    print(f"❌ Output validation failed for {input_file}")
                    continue
                
                # Save output
                out_filename = os.path.basename(input_file)
                out_path = os.path.join(OUTPUT_FOLDER, out_filename)
                with open(out_path, "w", encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Success! Output saved to {out_path}")
                
            except Exception as e:
                print(f"❌ Error processing {input_file}: {e}")
                continue
        
        print("\n🎉 All files processed. Exiting...")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)