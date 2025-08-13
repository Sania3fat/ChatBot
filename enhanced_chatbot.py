import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import os
import json
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict, Optional
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import time
import requests
import glob
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ask Scottie - Maryville College Academic Assistant",
    page_icon="🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Configure for embedding - hide Streamlit UI elements when embedded and add Maryville College styling
maryville_embedded_style = """
<style>
    /* Import Maryville College colors and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
    
    /* Maryville College Official Color Palette */
    :root {
        --maryville-maroon: #5B0F1B;
        --maryville-orange: #EC5E1A;
        --maryville-white: #FFFFFF;
        --maryville-light-gray: #F5F5F5;
        --maryville-dark-gray: #666666;
        --maryville-accent-blue: #1976D2;
        --maryville-success-green: #4CAF50;
        --maryville-warning-gold: #FFC107;
    }
    
    /* Hide the Streamlit header and menu */
    header[data-testid="stHeader"] {
        height: 0px;
        visibility: hidden;
    }
    
    /* Hide the footer */
    footer[data-testid="stFooter"] {
        visibility: hidden;
    }
    
    /* Remove default padding and margins for embedded mode */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
        background: linear-gradient(135deg, var(--maryville-light-gray) 0%, var(--maryville-white) 100%);
        border: 3px solid var(--maryville-maroon);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(91, 15, 27, 0.1);
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Ensure no horizontal overflow */
    .main {
        overflow-x: hidden;
    }
    
    /* Hide sidebar toggle button when collapsed */
    button[data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Hide the main menu hamburger */
    #MainMenu {
        visibility: hidden;
    }
    
    /* Remove "Made with Streamlit" footer */
    footer:after {
        content: "";
        visibility: hidden;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
    
    /* Prevent scrollbars on the main container */
    .stApp {
        overflow: hidden;
    }
    
    /* Set fixed height for chat container to prevent vertical scrolling */
    .stChatFloatingInputContainer {
        position: sticky;
        bottom: 0;
        background: var(--maryville-white);
        border-top: 1px solid var(--maryville-dark-gray);
        padding: 10px 0;
        z-index: 1000;
    }
    
    /* Chat messages container with controlled height */
    [data-testid="stChatMessageContainer"] {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    /* Custom scrollbar for chat messages */
    [data-testid="stChatMessageContainer"]::-webkit-scrollbar {
        width: 8px;
    }
    
    [data-testid="stChatMessageContainer"]::-webkit-scrollbar-track {
        background: var(--maryville-light-gray);
        border-radius: 4px;
    }
    
    [data-testid="stChatMessageContainer"]::-webkit-scrollbar-thumb {
        background: var(--maryville-maroon);
        border-radius: 4px;
    }
    
    [data-testid="stChatMessageContainer"]::-webkit-scrollbar-thumb:hover {
        background: var(--maryville-orange);
    }
    
    /* Style the main title with Maryville branding */
    .main h1 {
        color: var(--maryville-maroon);
        font-family: 'Open Sans', sans-serif;
        font-weight: 700;
        text-align: center;
        border-bottom: 3px solid var(--maryville-orange);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Style subheaders */
    .main h2, .main h3 {
        color: var(--maryville-maroon);
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        border: 1px solid var(--maryville-dark-gray);
        background-color: var(--maryville-white);
        margin-bottom: 10px;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, var(--maryville-maroon), var(--maryville-orange));
        color: white;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, var(--maryville-white), var(--maryville-light-gray));
        border-left: 4px solid var(--maryville-orange);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--maryville-maroon), var(--maryville-orange));
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--maryville-orange), var(--maryville-maroon));
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(91, 15, 27, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid var(--maryville-dark-gray);
        border-radius: 8px;
        font-family: 'Open Sans', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--maryville-maroon);
        box-shadow: 0 0 0 2px rgba(91, 15, 27, 0.2);
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-left: 4px solid var(--maryville-accent-blue);
        border-radius: 8px;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
        border-left: 4px solid var(--maryville-success-green);
        border-radius: 8px;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(135deg, #FFF8E1, #FFECB3);
        border-left: 4px solid var(--maryville-warning-gold);
        border-radius: 8px;
    }
    
    /* Error boxes */
    .stError {
        background: linear-gradient(135deg, #FFF5F5, #FFEBEE);
        border-left: 4px solid var(--maryville-maroon);
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--maryville-maroon), var(--maryville-orange));
        color: white;
    }
    
    /* Metrics styling */
    .metric-container {
        background: var(--maryville-white);
        border: 1px solid var(--maryville-dark-gray);
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Make sample questions more compact for embedding */
    .sample-questions-container {
        max-height: 200px;
        overflow-y: auto;
    }
    
    /* Compact header for embedded mode */
    .embedded-header {
        margin-bottom: 15px;
    }
    
    .embedded-header h1 {
        font-size: 1.8em;
        margin-bottom: 5px;
    }
    
    .embedded-header p {
        font-size: 0.9em;
        margin: 2px 0;
    }
</style>
"""

# Check if running in embedded mode - apply Maryville College styling
try:
    query_params = st.query_params
    if query_params.get("embed") or query_params.get("embedded"):
        st.markdown(maryville_embedded_style, unsafe_allow_html=True)
except AttributeError:
    # Fallback for older Streamlit versions
    try:
        query_params = st.experimental_get_query_params()
        if query_params.get("embed") or query_params.get("embedded"):
            st.markdown(maryville_embedded_style, unsafe_allow_html=True)
    except:
        # Apply Maryville styling by default for embedded use
        st.markdown(maryville_embedded_style, unsafe_allow_html=True)

# --- INSTITUTIONAL CONFIGURATION ---
class Config:
    """Optimized configuration for institutional deployment."""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150  # Your preferred setting
    SEARCH_RESULTS = 13  # Your preferred setting
    MODEL_NAME = 'all-MiniLM-L6-v2'
    CACHE_DURATION_DAYS = 90
    BATCH_SIZE = 32  # Larger batch size for efficiency
    MAX_RETRIES = 2  # Reduced retries for faster response
    
    # Institutional PDF directory and web sources
    INSTITUTIONAL_PDF_DIR = "institutional_pdfs"
    INSTITUTIONAL_WEB_SOURCES = "institutional_web_sources.json"  # JSON file with URLs
    CACHE_DIR = "institutional_cache"
    
    # Context-aware settings
    MAX_CONVERSATION_TOKENS = 2000  # Max tokens to use for conversation history
    CONTEXT_WINDOW_SIZE = 10  # Number of previous messages to consider

# --- HELPER FUNCTIONS & CLASSES ---

class InstitutionalPDFChatbot:
    """Optimized PDF chatbot for institutional deployment with pre-loaded documents."""
    
    def __init__(self):
        self.pdf_contents: Dict[str, str] = {}
        self.web_contents: Dict[str, str] = {}  # Store web page content
        self.text_chunks: List[Dict] = []
        self.chunk_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Initialize directories
        os.makedirs(Config.INSTITUTIONAL_PDF_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Load the sentence transformer model with optimized settings
        self.embedding_model = self._load_embedding_model()
        if self.embedding_model:
            self.model_device = self.embedding_model.device
        else:
            st.error("Could not load embedding model.")
            st.stop()

    @st.cache_resource
    def _load_embedding_model(_self):
        """Load embedding model with optimized retry logic."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                # Determine best device
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
                
                # Model cache directory
                cache_folder = os.path.join(os.getcwd(), "model_cache")
                os.makedirs(cache_folder, exist_ok=True)
                
                # Load model with settings consistent with first chatbot
                model = SentenceTransformer(
                    Config.MODEL_NAME, 
                    device=device,
                    cache_folder=cache_folder
                )
                
                return model
                
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    wait_time = (2 ** attempt) * 3  # Reduced wait time
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(wait_time)
                else:
                    break
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    # Try fallback model
                    try:
                        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
                    except:
                        return None
                else:
                    time.sleep(2 ** attempt)
        
        return None

    def get_cache_path(self, identifier: str) -> str:
        """Generates cache path for institutional documents."""
        cache_hash = hashlib.md5(identifier.encode('utf-8')).hexdigest()
        return os.path.join(Config.CACHE_DIR, f"institutional_{cache_hash}")

    def is_cache_valid(self, cache_path: str) -> bool:
        """Check if institutional cache is valid."""
        meta_file = f"{cache_path}_meta.json"
        if not os.path.exists(meta_file):
            return False
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            cached_at_str = metadata.get('cached_at')
            if not cached_at_str: 
                return False
            
            cached_date = datetime.fromisoformat(cached_at_str)
            return datetime.now() - cached_date < timedelta(days=Config.CACHE_DURATION_DAYS)
        except:
            return False

    def save_to_cache(self, identifier: str) -> None:
        """Save processed institutional data to cache."""
        cache_path = self.get_cache_path(identifier)
        
        try:
            metadata = {
                'identifier': identifier,
                'cached_at': datetime.now().isoformat(),
                'files_count': len(self.pdf_contents) + len(self.web_contents),
                'chunks_count': len(self.text_chunks),
                'config': {
                    'chunk_size': Config.CHUNK_SIZE,
                    'chunk_overlap': Config.CHUNK_OVERLAP,
                    'search_results': Config.SEARCH_RESULTS
                }
            }
            
            with open(f"{cache_path}_meta.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
                
            with open(f"{cache_path}_chunks.json", 'w', encoding='utf-8') as f:
                json.dump(self.text_chunks, f)
                
            if self.chunk_embeddings is not None:
                np.save(f"{cache_path}_embeddings.npy", self.chunk_embeddings.cpu().numpy())
            
            # Save TF-IDF components
            if self.tfidf_vectorizer is not None:
                import pickle
                with open(f"{cache_path}_tfidf_vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                np.save(f"{cache_path}_tfidf_matrix.npy", self.tfidf_matrix.toarray())
                
        except Exception as e:
            st.error(f"Error saving to cache: {e}")

    def load_from_cache(self, identifier: str) -> bool:
        """Load institutional data from cache."""
        cache_path = self.get_cache_path(identifier)
        if not self.is_cache_valid(cache_path):
            return False
            
        try:
            with open(f"{cache_path}_chunks.json", 'r', encoding='utf-8') as f:
                self.text_chunks = json.load(f)
            
            loaded_embeddings = np.load(f"{cache_path}_embeddings.npy")
            self.chunk_embeddings = torch.from_numpy(loaded_embeddings).to(self.model_device)

            # Load TF-IDF components
            import pickle
            try:
                with open(f"{cache_path}_tfidf_vectorizer.pkl", 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                self.tfidf_matrix = np.load(f"{cache_path}_tfidf_matrix.npy")
            except FileNotFoundError:
                self._create_tfidf_index()

            return True
        except Exception as e:
            return False

    def extract_text_from_pdf(self, pdf_path: str, filename: str) -> None:
        """Enhanced PDF text extraction with better cleaning - consistent with first chatbot."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Better text cleaning - same as first chatbot
                # Remove excessive whitespace
                page_text = re.sub(r'\s+', ' ', page_text)
                # Fix hyphenated words across lines
                page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page_text)
                # Remove header/footer patterns (basic)
                page_text = re.sub(r'^\d+\s*$', '', page_text, flags=re.MULTILINE)
                
                text += page_text + "\n\n"  # Double newline between pages
            
            doc.close()
            
            # Final cleaning
            text = text.strip()
            # Remove multiple consecutive newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            if len(text) > 100:
                self.pdf_contents[filename] = text
                st.success(f"✅ Extracted {len(text):,} characters from {filename}")
            else:
                st.warning(f"⚠️ Minimal content extracted from {filename}")
            
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {e}")

    def extract_text_from_webpage(self, url: str) -> None:
        """Extract text content from a webpage."""
        try:
            # Check robots.txt compliance (basic check)
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Set headers to identify as a bot
            headers = {
                'User-Agent': 'Ask Scottie Academic Chatbot (Educational Use)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            # Fetch the webpage
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from main content areas
            text_content = ""
            
            # Try to find main content areas first
            main_selectors = ['main', 'article', '.content', '.main-content', '#content', '#main']
            main_content = None
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text_content = main_content.get_text()
            else:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    text_content = body.get_text()
                else:
                    text_content = soup.get_text()
            
            # Clean the text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Additional cleaning
            text_content = re.sub(r'\s+', ' ', text_content)  # Multiple spaces to single
            text_content = re.sub(r'\n+', '\n', text_content)  # Multiple newlines to single
            
            if len(text_content) > 200:  # Minimum content threshold
                # Use domain name as identifier
                domain = urlparse(url).netloc
                page_title = soup.find('title')
                if page_title:
                    identifier = f"{domain} - {page_title.get_text().strip()}"
                else:
                    identifier = domain
                
                self.web_contents[identifier] = text_content
                st.success(f"✅ Extracted {len(text_content):,} characters from {identifier}")
            else:
                st.warning(f"⚠️ Minimal content extracted from {url}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error fetching {url}: {e}")
        except Exception as e:
            st.error(f"❌ Error processing {url}: {e}")

    def load_web_sources(self) -> bool:
        """Load web pages from the configured sources file."""
        web_sources_file = Config.INSTITUTIONAL_WEB_SOURCES
        
        if not os.path.exists(web_sources_file):
            return False
        
        try:
            with open(web_sources_file, 'r', encoding='utf-8') as f:
                web_sources = json.load(f)
            
            if not web_sources or 'urls' not in web_sources:
                return False
            
            urls = web_sources['urls']
            if not urls:
                return False
            
            progress_bar = st.progress(0, text="Processing web sources...")
            
            for i, url in enumerate(urls):
                st.info(f"Processing: {url}")
                self.extract_text_from_webpage(url)
                
                progress_bar.progress((i + 1) / len(urls), 
                                    text=f"Processing web source ({i+1}/{len(urls)})")
            
            progress_bar.empty()
            return len(self.web_contents) > 0
            
        except Exception as e:
            st.error(f"Error loading web sources: {e}")
            return False
        """Load all PDFs from the institutional directory."""
        self.pdf_contents = {}
        
        # Look for PDF files in the institutional directory
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        
        if not pdf_files:
            return False
        
        progress_bar = st.progress(0, text="Processing files...")
        
        for i, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)
            self.extract_text_from_pdf(pdf_path, filename)
            
            progress_bar.progress((i + 1) / len(pdf_files), 
                                text=f"Processing {filename} ({i+1}/{len(pdf_files)})")
        
        progress_bar.empty()
        
    def load_institutional_pdfs(self) -> bool:
        """Load all PDFs from the institutional directory."""
        self.pdf_contents = {}
        
        # Look for PDF files in the institutional directory
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        
        if not pdf_files:
            return False
        
        progress_bar = st.progress(0, text="Processing PDF files...")
        
        for i, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)
            self.extract_text_from_pdf(pdf_path, filename)
            
            progress_bar.progress((i + 1) / len(pdf_files), 
                                text=f"Processing {filename} ({i+1}/{len(pdf_files)})")
        
        progress_bar.empty()
        
        return len(self.pdf_contents) > 0

    def load_all_sources(self) -> bool:
        """Load both PDFs and web sources."""
        pdf_success = self.load_institutional_pdfs()
        web_success = self.load_web_sources()
        
        return pdf_success or web_success

    def smart_chunk_text(self, text: str, source: str) -> List[Dict]:
        """
        Enhanced chunking that preserves academic requirement structures and semantic boundaries.
        """
        chunks = []
        
        # Academic requirement patterns that need larger chunks
        requirement_patterns = [
            r'General Education Requirements',
            r'Category [IVX]+ Domains?',
            r'Foundations?\s*\(\d+\s*Credit Hours?\)',
            r'Required courses? include:?',
            r'Completion of one',
            r'The Maryville Curriculum',
            r'Undergraduate Degree Requirements',
            r'Domain[s]? of Knowledge',
            r'Empirical Study',
            r'Scientific Reasoning',
            r'Mathematical Reasoning',
            r'Creative Arts',
            r'Ethical Citizenship'
        ]
        
        # Check if this text contains academic requirements
        is_requirements_section = any(re.search(pattern, text, re.IGNORECASE) 
                                     for pattern in requirement_patterns)
        
        # Adjust chunking parameters based on content type
        if is_requirements_section:
            chunk_size = 1400  # Much larger for requirements to keep sections together
            overlap_size = 300   # Larger overlap to ensure continuity
        else:
            chunk_size = Config.CHUNK_SIZE
            overlap_size = Config.CHUNK_OVERLAP
        
        # Split into sentences first
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback if NLTK fails
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                # Add overlap from previous chunk if it exists
                if chunks and overlap_size > 0:
                    prev_chunk_words = chunks[-1]['text'].split()
                    overlap_words = prev_chunk_words[-min(overlap_size//5, len(prev_chunk_words)):]
                    chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
                
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': len(chunks),
                    'is_requirements': is_requirements_section
                })
                
                # Start new chunk with overlap
                if overlap_size > 0:
                    overlap_sentences = current_chunk[-min(3, len(current_chunk)):]
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunks and overlap_size > 0:
                prev_chunk_words = chunks[-1]['text'].split()
                overlap_words = prev_chunk_words[-min(overlap_size//5, len(prev_chunk_words)):]
                chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
            
            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': len(chunks),
                'is_requirements': is_requirements_section
            })
        
        return chunks

    def _create_tfidf_index(self):
        """Create TF-IDF index for keyword-based search - consistent with first chatbot."""
        if not self.text_chunks:
            return
            
        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)

    def create_chunks_and_embeddings(self) -> bool:
        """Process PDF and web content into chunks and generate embeddings."""
        self.text_chunks = []
        if not self.pdf_contents and not self.web_contents:
            st.error("No PDF or web content available.")
            return False

        # Create chunks from PDFs
        for filename, content in self.pdf_contents.items():
            file_chunks = self.smart_chunk_text(content, f"📄 {filename}")
            self.text_chunks.extend(file_chunks)
        
        # Create chunks from web sources
        for source_name, content in self.web_contents.items():
            web_chunks = self.smart_chunk_text(content, f"🌐 {source_name}")
            self.text_chunks.extend(web_chunks)
        
        if not self.text_chunks:
            st.error("No chunks created from sources.")
            return False

        st.info(f"Creating embeddings for {len(self.text_chunks)} chunks...")

        # Generate semantic embeddings - consistent with first chatbot
        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        try:
            self.chunk_embeddings = self.embedding_model.encode(
                chunk_texts, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=32  # Process in batches for better memory usage
                # Removed normalize_embeddings=True to match first chatbot
            )
            
            # Create TF-IDF index
            self._create_tfidf_index()
            
            st.success("✅ Embeddings and indexes created!")
            return True
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            self.chunk_embeddings = None
            return False

    def extract_context_keywords(self, conversation_history: List[Dict]) -> List[str]:
        """Extract important keywords from recent conversation for enhanced search."""
        if not conversation_history:
            return []
        
        # Get recent messages (last 4-6 messages)
        recent_messages = conversation_history[-6:]
        
        # Extract nouns and important terms from recent conversation
        keywords = []
        for msg in recent_messages:
            if msg['role'] == 'assistant':
                # Extract quoted terms, document names, and emphasized terms
                quotes = re.findall(r'"([^"]*)"', msg['content'])
                keywords.extend(quotes)
                
                # Extract document references
                doc_refs = re.findall(r'\[([^\]]+\.pdf)\]', msg['content'])
                keywords.extend(doc_refs)
                
                # Extract bolded terms
                bold_terms = re.findall(r'\*\*([^*]+)\*\*', msg['content'])
                keywords.extend(bold_terms)
        
        return list(set(keywords))  # Remove duplicates

    def get_related_requirement_chunks(self, initial_chunks: List[Dict], question: str) -> List[Dict]:
        """Find and include related requirement chunks for comprehensive queries."""
        related_chunks = list(initial_chunks)
        
        # Comprehensive requirement keywords
        comprehensive_keywords = [
            'general education', 'curriculum requirements', 'degree requirements',
            'all requirements', 'what do i need', 'complete requirements'
        ]
        
        is_comprehensive = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        if is_comprehensive:
            # Find requirement sources from initial chunks
            requirement_sources = set()
            for chunk in initial_chunks:
                if chunk.get('is_requirements', False) or any(keyword in chunk['text'].lower() for keyword in 
                       ['requirements', 'category', 'foundations', 'credit hours', 'domain']):
                    requirement_sources.add(chunk['source'])
            
            if requirement_sources:
                # Add ALL requirement-related chunks from the same sources
                requirement_keywords = [
                    'category', 'foundations', 'domain', 'credit hours', 'completion of',
                    'required courses', 'general education', 'curriculum', 'empirical study',
                    'scientific reasoning', 'mathematical reasoning', 'creative arts',
                    'ethical citizenship', 'literary studies', 'historical reasoning',
                    'second language', 'pluralism'
                ]
                
                for chunk in self.text_chunks:
                    if (chunk['source'] in requirement_sources and 
                        chunk not in related_chunks and
                        any(keyword in chunk['text'].lower() for keyword in requirement_keywords)):
                        
                        # Calculate a simple relevance score
                        relevance_score = sum(1 for keyword in requirement_keywords 
                                            if keyword in chunk['text'].lower())
                        chunk['requirement_relevance'] = relevance_score
                        related_chunks.append(chunk)
                
                # Sort by relevance and limit
                related_chunks.sort(key=lambda x: x.get('requirement_relevance', 0), reverse=True)
                return related_chunks[:25]  # Increase limit for comprehensive queries
        
        return related_chunks[:15]  # Standard limit

    def context_aware_search(self, question: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """
        Enhanced search that considers conversation context and handles comprehensive requirement queries.
        """
        if self.chunk_embeddings is None or len(self.text_chunks) == 0:
            st.warning("No embeddings available for search.")
            return []
        
        # Detect comprehensive requirement queries
        comprehensive_keywords = [
            'general education requirements', 'curriculum requirements', 'degree requirements',
            'all requirements', 'what do i need to take', 'complete requirements',
            'graduation requirements', 'core curriculum'
        ]
        
        is_comprehensive_query = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        # Extract context keywords from conversation
        context_keywords = []
        if conversation_history:
            context_keywords = self.extract_context_keywords(conversation_history)
        
        # Enhance the question with context and comprehensive search terms
        enhanced_question = question
        if is_comprehensive_query:
            # Add comprehensive search terms
            enhanced_question = f"{question} foundations category domains credit hours courses curriculum general education requirements"
        elif context_keywords and any(word in question.lower() for word in ['that', 'this', 'it', 'those', 'these', 'more', 'else']):
            # This seems to be a follow-up question
            enhanced_question = f"{question} {' '.join(context_keywords[:3])}"
        
        try:
            # Semantic search with enhanced question
            question_embedding = self.embedding_model.encode(enhanced_question, convert_to_tensor=True)
            semantic_scores = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
            
            # Keyword search (TF-IDF)
            keyword_scores = np.zeros(len(self.text_chunks))
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                question_tfidf = self.tfidf_vectorizer.transform([enhanced_question])
                keyword_similarities = cosine_similarity(question_tfidf, self.tfidf_matrix)[0]
                keyword_scores = keyword_similarities
            
            # Boost scores for requirement chunks when asking comprehensive questions
            if is_comprehensive_query:
                for i, chunk in enumerate(self.text_chunks):
                    if chunk.get('is_requirements', False):
                        semantic_scores[i] = semantic_scores[i] * 1.5  # 50% boost for requirement chunks
                    
                    # Additional boost for chunks containing key requirement terms
                    requirement_terms = ['category', 'foundations', 'domain', 'credit hours']
                    term_count = sum(1 for term in requirement_terms if term in chunk['text'].lower())
                    if term_count > 0:
                        semantic_scores[i] = semantic_scores[i] * (1.0 + 0.1 * term_count)
            
            # Boost scores for chunks mentioned in recent conversation
            if conversation_history and len(conversation_history) > 0:
                recent_sources = set()
                for msg in conversation_history[-4:]:  # Look at last 4 messages
                    if msg['role'] == 'assistant':
                        # Extract source references
                        sources = re.findall(r'\[([^\]]+\.pdf)\]', msg['content'])
                        recent_sources.update(sources)
                
                # Boost chunks from recently discussed sources
                for i, chunk in enumerate(self.text_chunks):
                    if chunk['source'] in recent_sources:
                        semantic_scores[i] = semantic_scores[i] * 1.2  # 20% boost
            
            # Combine scores (weighted average)
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            # Normalize scores to 0-1 range
            semantic_scores_norm = (semantic_scores.cpu().numpy() + 1) / 2
            keyword_scores_norm = keyword_scores
            
            combined_scores = (semantic_weight * semantic_scores_norm + 
                             keyword_weight * keyword_scores_norm)
            
            # Determine number of results based on query type
            if is_comprehensive_query:
                num_results = min(25, len(self.text_chunks))  # Get more results for comprehensive queries
            else:
                num_results = Config.SEARCH_RESULTS
            
            # Get top results
            top_indices = np.argsort(combined_scores)[-num_results:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.text_chunks[idx].copy()
                chunk['semantic_score'] = semantic_scores[idx].item()
                chunk['keyword_score'] = keyword_scores[idx]
                chunk['combined_score'] = combined_scores[idx]
                relevant_chunks.append(chunk)
            
            # Filter out very low scores (lower threshold for comprehensive queries)
            min_score = 0.05 if is_comprehensive_query else 0.1
            relevant_chunks = [chunk for chunk in relevant_chunks 
                             if chunk['combined_score'] > min_score]
            
            # Get related requirement chunks for comprehensive queries
            if is_comprehensive_query and relevant_chunks:
                relevant_chunks = self.get_related_requirement_chunks(relevant_chunks, question)
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error in context-aware search: {e}")
            return []

    def summarize_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Create a summary of the conversation context for the prompt."""
        if not conversation_history:
            return ""
        
        # Take last few exchanges
        recent_history = conversation_history[-Config.CONTEXT_WINDOW_SIZE:]
        
        context_summary = "Previous conversation:\n"
        for msg in recent_history:
            if msg['role'] == 'user':
                context_summary += f"User asked: {msg['content']}\n"
            else:
                # Summarize assistant responses to save tokens
                response = msg['content']
                if len(response) > 200:
                    # Extract key points
                    first_sentence = response.split('.')[0] + '.'
                    context_summary += f"Assistant explained: {first_sentence}...\n"
                else:
                    context_summary += f"Assistant explained: {response}\n"
        
        return context_summary

    def generate_answer(self, question: str, context_chunks: List[Dict], client: OpenAI, 
                       conversation_history: List[Dict] = None) -> str:
        """Generate answer with conversation context awareness and enhanced requirement handling."""
        if not context_chunks:
            return "I couldn't find relevant information in the PDF documents to answer your question."
        
        # Detect if this is a comprehensive requirements question
        comprehensive_keywords = [
            'general education requirements', 'curriculum requirements', 'degree requirements',
            'all requirements', 'what do i need', 'graduation requirements'
        ]
        is_comprehensive_query = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        # Prepare document context
        context_str = ""
        sources = set()
        
        for i, chunk in enumerate(context_chunks):
            context_str += f"=== Context {i+1} (from {chunk['source']}) ===\n"
            context_str += f"{chunk['text']}\n\n"
            sources.add(chunk['source'])
        
        # Prepare conversation context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_context = self.summarize_conversation_context(conversation_history)
        
        # Enhanced context-aware prompt with special handling for comprehensive queries
        base_prompt = f"""You are an expert document analyst with access to institutional PDF documents. You are having an ongoing conversation with a user.

{conversation_context}

CURRENT QUESTION: {question}

CONTEXT FROM DOCUMENTS:
{context_str}

INSTRUCTIONS:
1. Answer the current question based on the provided document context
2. Consider the conversation history to understand what the user is referring to
3. If the current question refers to something discussed earlier (like "that", "it", "those"), make the connection clear
4. Maintain consistency with your previous answers
5. If the context doesn't contain sufficient information, clearly state what's missing
6. Cite sources by mentioning the document name in brackets, e.g., [document.pdf]
7. Be specific and detailed in your response
8. If this is a follow-up question, acknowledge the connection to the previous discussion"""

        # Add special instructions for comprehensive requirement queries
        if is_comprehensive_query:
            comprehensive_instruction = """

SPECIAL INSTRUCTIONS FOR COMPREHENSIVE REQUIREMENTS:
This question asks for comprehensive general education requirements. Please provide ALL categories and requirements found in the context, including:
- Foundations requirements (with credit hours)
- All Category I, II, III, and IV domains (with credit hours for each)
- Specific course options within each category
- Any special conditions, notes, or placement exam exemptions
- US Pluralism requirements
- Second Language requirements
- Any other degree requirements mentioned

Structure your response clearly with headers for each major category (Foundations, Category I, Category II, etc.) and list all course options and credit hour requirements. Be comprehensive and don't omit any categories or requirements that appear in the context."""
            
            base_prompt += comprehensive_instruction

        prompt = base_prompt + "\n\nAnswer:"

        try:
            # Build message history for better context
            messages = [
                {"role": "system", "content": "You are 'Ask Scottie', Maryville College's helpful academic assistant. You ONLY answer questions about information found in the Maryville College Academic Catalog. You maintain context across conversations and provide accurate information based on the provided PDF documents. When users ask questions outside the academic catalog scope, politely redirect them to contact the appropriate Maryville College department. Always maintain a friendly, helpful tone while being clear about your limitations to academic catalog information only."}
            ]
            
            # Include a few recent exchanges for additional context (if they exist)
            if conversation_history and len(conversation_history) > 2:
                # Add last 2-3 exchanges to provide context to the model
                recent_exchanges = conversation_history[-(min(6, len(conversation_history))):]
                for msg in recent_exchanges:
                    if msg['role'] == 'user':
                        messages.append({"role": "user", "content": msg['content']})
                    else:
                        # Truncate long responses to save tokens
                        content = msg['content']
                        if len(content) > 500:
                            content = content[:500] + "..."
                        messages.append({"role": "assistant", "content": content})
            
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Adjust max_tokens for comprehensive queries
            max_tokens = 3000 if is_comprehensive_query else 2000
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"

    # Keep the original hybrid_search method for backward compatibility
    def hybrid_search(self, question: str) -> List[Dict]:
        """Original hybrid search method - now calls context_aware_search."""
        return self.context_aware_search(question, conversation_history=None)


# --- STREAMLIT UI ---

def get_openai_client():
    """Get OpenAI client with API key."""
    api_key = st.secrets.get("OPENAI_API_KEY") 
    
    if not api_key:
        st.warning("🔑 OpenAI API key not found in Streamlit secrets.")
        
        with st.expander("🔧 API Key Setup Instructions", expanded=True):
            st.markdown("""
            **To set up your OpenAI API key:**
            
            1. **For local development:** Create a `.streamlit/secrets.toml` file:
               ```toml
               OPENAI_API_KEY = "your-api-key-here"
               ```
            
            2. **For Streamlit Cloud:** Add the key in your app's secrets section
            
            3. **For other deployments:** Set the environment variable `OPENAI_API_KEY`
            
            **Get your API key from:** https://platform.openai.com/api-keys
            """)
        
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI: {e}")
        return None

def initialize_chatbot():
    """Initialize chatbot with institutional documents."""
    if 'chatbot_initialized' not in st.session_state:
        chatbot = InstitutionalPDFChatbot()
        
        # Create identifier for institutional documents (including web sources)
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        web_sources_file = Config.INSTITUTIONAL_WEB_SOURCES
        
        if not pdf_files and not os.path.exists(web_sources_file):
            # Show setup instructions instead of stopping
            st.warning(f"⚠️ No PDF files or web sources found.")
            
            with st.expander("📋 Setup Instructions", expanded=True):
                st.markdown("""
                **To set up Ask Scottie's Academic Catalog knowledge base:**
                
                **Option 1: PDF Files**
                1. Create a folder named `institutional_pdfs` in the same directory as this app
                2. Add your **Maryville College Academic Catalog PDF documents** to this folder
                
                **Option 2: Web Sources**
                1. Create a file named `institutional_web_sources.json` in the same directory as this app
                2. Add URLs to scrape in this format:
                ```json
                {
                  "urls": [
                    "https://www.maryvillecollege.edu/academics/catalog/",
                    "https://www.maryvillecollege.edu/admissions/requirements/",
                    "https://www.maryvillecollege.edu/academics/programs/"
                  ]
                }
                ```
                
                **Option 3: Both PDFs and Web Sources**
                - Set up both options above for comprehensive coverage
                
                **Example folder structure:**
                ```
                ask_scottie/
                ├── institutional_pdfs/
                │   ├── maryville_academic_catalog_2025-2026.pdf
                │   └── student_handbook.pdf
                ├── institutional_web_sources.json
                └── ask_scottie.py
                ```
                
                **Note:** Ask Scottie only works with Academic Catalog information.
                """)
            
            # Create the directory if it doesn't exist
            os.makedirs(Config.INSTITUTIONAL_PDF_DIR, exist_ok=True)
            
            # Return a dummy chatbot to allow the app to continue
            st.session_state.chatbot = None
            st.session_state.chatbot_initialized = False
            return None
        
        with st.spinner("Initializing knowledge base..."):
            # Create cache identifier including both PDFs and web sources
            file_stats = []
            
            # Add PDF file stats
            for pdf_path in sorted(pdf_files):
                stat = os.stat(pdf_path)
                file_stats.append(f"pdf-{os.path.basename(pdf_path)}-{stat.st_size}-{stat.st_mtime}")
            
            # Add web sources file stat if it exists
            if os.path.exists(web_sources_file):
                stat = os.stat(web_sources_file)
                file_stats.append(f"web-sources-{stat.st_size}-{stat.st_mtime}")
            
            identifier = "institutional_" + hashlib.sha256("|".join(file_stats).encode('utf-8')).hexdigest()
            
            # Try to load from cache first
            if chatbot.load_from_cache(identifier):
                st.success("✅ Knowledge base loaded from cache")
            else:
                # Load and process documents
                if chatbot.load_all_sources():  # This loads both PDFs and web sources
                    success = chatbot.create_chunks_and_embeddings()
                    if success:
                        chatbot.save_to_cache(identifier)
                        st.success("✅ Knowledge base initialized and cached")
                    else:
                        st.error("Failed to create embeddings")
                        return None
                else:
                    st.error("Failed to load any sources")
                    return None
            
            st.session_state.chatbot = chatbot
            st.session_state.chatbot_initialized = True
            st.session_state.messages = []
            
    return st.session_state.chatbot

def main():
    # Add Maryville College header with Scottish Terrier mascot (compact for embedding)
    st.markdown("""
    <div class="embedded-header" style="text-align: center; margin-bottom: 20px;">
        <div style="display: inline-block; background: linear-gradient(135deg, #5B0F1B, #EC5E1A); 
                    color: white; padding: 15px 30px; border-radius: 12px; margin-bottom: 15px;
                    box-shadow: 0 4px 15px rgba(91, 15, 27, 0.3);">
            <h1 style="margin: 0; font-size: 1.8em; font-weight: 700;">🐕‍🦺 Ask Scottie</h1>
            <p style="margin: 3px 0; font-size: 0.9em; opacity: 0.9;">Scottish Terrier Academic Assistant</p>
            <p style="margin: 5px 0 0 0; font-size: 1.0em; opacity: 0.9;">Maryville College Academic Catalog</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add scope limitation notice (compact)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #E3F2FD, #BBDEFB); 
                border: 2px solid #1976D2; border-radius: 8px; padding: 10px; margin: 15px 0;
                text-align: center;">
        <p style="margin: 0; color: #5B0F1B; font-weight: 600; font-size: 0.9em;">
            📚 Academic Catalog questions only. For other inquiries, contact the appropriate department.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize OpenAI client
    openai_client = get_openai_client()
    if openai_client is None:
        st.stop()
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    if chatbot is None:
        # Show a helpful message instead of stopping
        st.info("👆 Please follow the setup instructions above to add Academic Catalog PDF documents.")
        
        # Show some demo content
        st.markdown("""
        <div style="background: white; border: 2px solid #FFC107; border-radius: 10px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #5B0F1B; text-align: center; margin-top: 0;">🔧 Ask Scottie System Status</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Application loaded successfully")
        with col2:
            st.success("✅ OpenAI client initialized")
        with col3:
            st.error("❌ No sources found")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add refresh button
        if st.button("🔄 Refresh After Adding Sources"):
            st.rerun()
        
        return
    
    # Display system info in collapsed sidebar with Maryville styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #5B0F1B; margin: 0;">🐕‍🦺 Ask Scottie</h2>
            <p style="color: #5B0F1B; margin: 5px 0 0 0;">Academic Catalog Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("📊 System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(chatbot.pdf_contents))
            st.metric("Knowledge Chunks", len(chatbot.text_chunks))
        with col2:
            st.metric("Web Sources", len(chatbot.web_contents))
            st.metric("Chat Messages", len(st.session_state.messages))
        
        st.markdown("---")
        st.markdown("**Configuration:**")
        st.text(f"Chunk Size: {Config.CHUNK_SIZE}")
        st.text(f"Overlap: {Config.CHUNK_OVERLAP}")
        st.text(f"Model: {Config.MODEL_NAME}")
        st.text(f"Context Window: {Config.CONTEXT_WINDOW_SIZE} messages")
        
        if st.session_state.messages:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("🆕 New Topic"):
                    # Add a separator message
                    st.session_state.messages.append({
                        "role": "system", 
                        "content": "--- New Topic Started ---"
                    })
                    st.rerun()
    
    # Display available documents with Maryville styling (compact for embedding)
    if st.session_state.messages == []:
        st.markdown("""
        <div style="background: white; border: 2px solid #EC5E1A; border-radius: 8px; padding: 15px; margin: 15px 0;">
            <h3 style="color: #5B0F1B; text-align: center; margin-top: 0; font-size: 1.1em;">📚 Available Sources</h3>
        """, unsafe_allow_html=True)
        
        # Show PDF documents
        pdf_names = list(chatbot.pdf_contents.keys())
        web_names = list(chatbot.web_contents.keys())
        
        if pdf_names:
            st.markdown("<p style='color: #5B0F1B; font-weight: 600; margin: 10px 0 5px 0;'>📄 PDF Documents:</p>", unsafe_allow_html=True)
            for doc_name in pdf_names:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #E3F2FD, #BBDEFB); 
                            border: 1px solid #1976D2; border-radius: 6px; 
                            padding: 8px; margin: 5px 0; text-align: center; font-size: 0.9em;">
                    <strong>📄 {doc_name}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        if web_names:
            st.markdown("<p style='color: #5B0F1B; font-weight: 600; margin: 10px 0 5px 0;'>🌐 Web Sources:</p>", unsafe_allow_html=True)
            for web_name in web_names:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #E8F5E8, #C8E6C9); 
                            border: 1px solid #4CAF50; border-radius: 6px; 
                            padding: 8px; margin: 5px 0; text-align: center; font-size: 0.9em;">
                    <strong>🌐 {web_name}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sample-questions-container" style="background: white; border: 2px solid #4CAF50; border-radius: 8px; padding: 15px; margin: 15px 0;">
            <h3 style="color: #5B0F1B; text-align: center; margin-top: 0; font-size: 1.1em;">💡 Sample Questions</h3>
        """, unsafe_allow_html=True)
        
        sample_questions = [
            "What are the general education requirements?",
            "What is the grading policy?",
            "What are the admission requirements?",
            "What are the graduation requirements?",
            "Tell me about the Maryville Curriculum"
        ]
        
        cols = st.columns(1)  # Use single column for embedding to save space
        for i, question in enumerate(sample_questions):
            with cols[0]:
                if st.button(question, key=f"sample_{i}"):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # Generate response immediately
                    with st.spinner("🐕‍🦺 Scottie is searching the Academic Catalog..."):
                        # Use context-aware search
                        relevant_chunks = chatbot.context_aware_search(
                            question, 
                            conversation_history=st.session_state.messages
                        )
                        
                        # Generate answer with conversation history
                        answer = chatbot.generate_answer(
                            question, 
                            relevant_chunks, 
                            openai_client,
                            conversation_history=st.session_state.messages
                        )
                    
                    # Add assistant response
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Rerun to show the conversation
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "system":
            # Display system messages differently
            st.markdown(f"<div style='text-align: center; color: gray; margin: 20px 0;'>{message['content']}</div>", 
                       unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input with Maryville styling
    if question := st.chat_input("Ask Scottie about the Academic Catalog..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response with context awareness
        with st.chat_message("assistant"):
            with st.spinner("🐕‍🦺 Scottie is searching the Academic Catalog..."):
                # Use context-aware search
                relevant_chunks = chatbot.context_aware_search(
                    question, 
                    conversation_history=st.session_state.messages
                )
                
                # Generate answer with conversation history
                answer = chatbot.generate_answer(
                    question, 
                    relevant_chunks, 
                    openai_client,
                    conversation_history=st.session_state.messages
                )
                st.markdown(answer)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()