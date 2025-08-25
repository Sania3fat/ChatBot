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
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ask Scottie - Maryville College Academic Assistant",
    page_icon="🐕‍🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple styling that definitely works
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #5B0F1B, #EC5E1A) !important;
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .info-box {
        background: #E8E8E8 !important;
        color: #333333 !important;
        border: 1px solid #CCCCCC !important;
        border-left: 4px solid #5B0F1B !important;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: normal !important;
    }
    
    .info-box strong {
        color: #222222 !important;
        font-weight: bold !important;
    }
    
    .source-item {
        background: #F8F8F8 !important;
        color: #333333 !important;
        border: 1px solid #DDDDDD !important;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    /* Ensure text visibility across all themes */
    .stMarkdown p {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)


# --- INSTITUTIONAL CONFIGURATION ---
class Config:
    """Configuration for institutional deployment."""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    SEARCH_RESULTS = 13
    MODEL_NAME = 'all-MiniLM-L6-v2'
    CACHE_DURATION_DAYS = 90
    BATCH_SIZE = 32
    MAX_RETRIES = 2
    
    # Institutional PDF directory and web sources
    INSTITUTIONAL_PDF_DIR = "institutional_pdfs"
    INSTITUTIONAL_WEB_SOURCES = "institutional_web_sources.json"
    CACHE_DIR = "institutional_cache"
    
    # Context-aware settings
    MAX_CONVERSATION_TOKENS = 2000
    CONTEXT_WINDOW_SIZE = 10

class InstitutionalPDFChatbot:
    """PDF chatbot for institutional deployment."""
    
    def __init__(self):
        self.pdf_contents: Dict[str, str] = {}
        self.web_contents: Dict[str, str] = {}
        self.text_chunks: List[Dict] = []
        self.chunk_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Initialize directories
        os.makedirs(Config.INSTITUTIONAL_PDF_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Load the sentence transformer model
        self.embedding_model = self._load_embedding_model()
        if self.embedding_model:
            self.model_device = self.embedding_model.device
        else:
            st.error("Could not load embedding model.")
            st.stop()

    @st.cache_resource
    def _load_embedding_model(_self):
        """Load embedding model."""
        try:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            
            cache_folder = os.path.join(os.getcwd(), "model_cache")
            os.makedirs(cache_folder, exist_ok=True)
            
            model = SentenceTransformer(
                Config.MODEL_NAME, 
                device=device,
                cache_folder=cache_folder
            )
            
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
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
        """Extract text from PDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page_text)
                text += page_text + "\n\n"
            
            doc.close()
            
            text = text.strip()
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
            headers = {
                'User-Agent': 'Ask Scottie Academic Chatbot (Educational Use)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from main content areas
            main_selectors = ['main', 'article', '.content', '.main-content', '#content', '#main']
            main_content = None
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text_content = main_content.get_text()
            else:
                body = soup.find('body')
                if body:
                    text_content = body.get_text()
                else:
                    text_content = soup.get_text()
            
            # Clean the text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            text_content = re.sub(r'\s+', ' ', text_content)
            
            if len(text_content) > 200:
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

    def load_institutional_pdfs(self) -> bool:
        """Load all PDFs from the institutional directory."""
        self.pdf_contents = {}
        
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
        """Enhanced chunking that preserves academic requirement structures."""
        chunks = []
        
        # Academic requirement patterns
        requirement_patterns = [
            r'General Education Requirements',
            r'Category [IVX]+ Domains?',
            r'Foundations?\s*\(\d+\s*Credit Hours?\)',
            r'Required courses? include:?',
            r'Completion of one',
            r'The Maryville Curriculum',
            r'Undergraduate Degree Requirements'
        ]
        
        is_requirements_section = any(re.search(pattern, text, re.IGNORECASE) 
                                     for pattern in requirement_patterns)
        
        if is_requirements_section:
            chunk_size = 1400
            overlap_size = 300
        else:
            chunk_size = Config.CHUNK_SIZE
            overlap_size = Config.CHUNK_OVERLAP
        
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
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
        """Create TF-IDF index for keyword-based search."""
        if not self.text_chunks:
            return
            
        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
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

        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        try:
            self.chunk_embeddings = self.embedding_model.encode(
                chunk_texts, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=32
            )
            
            self._create_tfidf_index()
            
            st.success("✅ Embeddings and indexes created!")
            return True
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            self.chunk_embeddings = None
            return False

    def context_aware_search(self, question: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """Enhanced search for comprehensive requirement queries."""
        if self.chunk_embeddings is None or len(self.text_chunks) == 0:
            st.warning("No embeddings available for search.")
            return []
        
        comprehensive_keywords = [
            'general education requirements', 'curriculum requirements', 'degree requirements',
            'all requirements', 'what do i need to take', 'complete requirements',
            'graduation requirements', 'core curriculum'
        ]
        
        is_comprehensive_query = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        enhanced_question = question
        if is_comprehensive_query:
            enhanced_question = f"{question} foundations category domains credit hours courses curriculum general education requirements"
        
        try:
            question_embedding = self.embedding_model.encode(enhanced_question, convert_to_tensor=True)
            semantic_scores = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
            
            keyword_scores = np.zeros(len(self.text_chunks))
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                question_tfidf = self.tfidf_vectorizer.transform([enhanced_question])
                keyword_similarities = cosine_similarity(question_tfidf, self.tfidf_matrix)[0]
                keyword_scores = keyword_similarities
            
            if is_comprehensive_query:
                for i, chunk in enumerate(self.text_chunks):
                    if chunk.get('is_requirements', False):
                        semantic_scores[i] = semantic_scores[i] * 1.5
            
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            semantic_scores_norm = (semantic_scores.cpu().numpy() + 1) / 2
            keyword_scores_norm = keyword_scores
            
            combined_scores = (semantic_weight * semantic_scores_norm + 
                             keyword_weight * keyword_scores_norm)
            
            if is_comprehensive_query:
                num_results = min(25, len(self.text_chunks))
            else:
                num_results = Config.SEARCH_RESULTS
            
            top_indices = np.argsort(combined_scores)[-num_results:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.text_chunks[idx].copy()
                chunk['semantic_score'] = semantic_scores[idx].item()
                chunk['keyword_score'] = keyword_scores[idx]
                chunk['combined_score'] = combined_scores[idx]
                relevant_chunks.append(chunk)
            
            min_score = 0.05 if is_comprehensive_query else 0.1
            relevant_chunks = [chunk for chunk in relevant_chunks 
                             if chunk['combined_score'] > min_score]
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error in search: {e}")
            return []

    def generate_answer(self, question: str, context_chunks: List[Dict], client: OpenAI, 
                       conversation_history: List[Dict] = None) -> str:
        """Generate answer with enhanced requirement handling."""
        if not context_chunks:
            return "I couldn't find relevant information in the documents to answer your question."
        
        comprehensive_keywords = [
            'general education requirements', 'curriculum requirements', 'degree requirements',
            'all requirements', 'what do i need', 'graduation requirements'
        ]
        is_comprehensive_query = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            context_str += f"=== Context {i+1} (from {chunk['source']}) ===\n"
            context_str += f"{chunk['text']}\n\n"
        
        base_prompt = f"""You are 'Ask Scottie', Maryville College's helpful academic assistant. You ONLY answer questions about information found in the Maryville College Academic Catalog.

CURRENT QUESTION: {question}

CONTEXT FROM DOCUMENTS:
{context_str}

INSTRUCTIONS:
1. Answer the current question based on the provided document context
2. If the context doesn't contain sufficient information, clearly state what's missing
3. Cite sources by mentioning the document name in brackets, e.g., [document.pdf]
4. Be specific and detailed in your response"""

        if is_comprehensive_query:
            comprehensive_instruction = """

SPECIAL INSTRUCTIONS FOR COMPREHENSIVE REQUIREMENTS:
This question asks for comprehensive general education requirements. Please provide ALL categories and requirements found in the context, including:
- Foundations requirements (with credit hours)
- All Category I, II, III, and IV domains (with credit hours for each)
- Specific course options within each category
- Any special conditions, notes, or placement exam exemptions

Structure your response clearly with headers for each major category and list all course options and credit hour requirements."""
            
            base_prompt += comprehensive_instruction

        prompt = base_prompt + "\n\nAnswer:"

        try:
            max_tokens = 3000 if is_comprehensive_query else 2000
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"

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
        
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        web_sources_file = Config.INSTITUTIONAL_WEB_SOURCES
        
        if not pdf_files and not os.path.exists(web_sources_file):
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
                    "https://www.maryvillecollege.edu/admissions/requirements/"
                  ]
                }
                ```
                """)
            
            os.makedirs(Config.INSTITUTIONAL_PDF_DIR, exist_ok=True)
            
            st.session_state.chatbot = None
            st.session_state.chatbot_initialized = False
            return None
        
        with st.spinner("Initializing knowledge base..."):
            file_stats = []
            
            for pdf_path in sorted(pdf_files):
                stat = os.stat(pdf_path)
                file_stats.append(f"pdf-{os.path.basename(pdf_path)}-{stat.st_size}-{stat.st_mtime}")
            
            if os.path.exists(web_sources_file):
                stat = os.stat(web_sources_file)
                file_stats.append(f"web-sources-{stat.st_size}-{stat.st_mtime}")
            
            identifier = "institutional_" + hashlib.sha256("|".join(file_stats).encode('utf-8')).hexdigest()
            
            if chatbot.load_from_cache(identifier):
                st.success("✅ Knowledge base loaded from cache")
            else:
                if chatbot.load_all_sources():
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

# Add this near the top of your main() function, after the header
def keep_alive():
    """Add auto-refresh to prevent sleeping"""
    st.markdown("""
    <script>
        // Auto-refresh every 25 minutes to prevent sleeping
        setTimeout(function(){
            window.location.reload();
        }, 300000); // 5 minutes in milliseconds
    </script>
    """, unsafe_allow_html=True)


def main():
    # Header with your custom logo
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <img src="data:image/png;base64,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" 
                 alt="Scottie Logo" 
                 style="width: 60px; height: 60px; border-radius: 50%; background: white; padding: 5px;">
            <div>
                <h1 style="margin: 0; font-size: 2.5em;">Ask Scottie</h1>
                <p style="margin: 1px 0 0 0;">Academic Catalog Assistant</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Scope limitation notice
    st.markdown("""
    <div class="info-box">
        <strong>📚 Ask Scottie can only answer questions about the Maryville College Academic Catalog.</strong><br>
        For other inquiries, please contact the appropriate college department.
    </div>
    """, unsafe_allow_html=True)

    # Initialize OpenAI client
    openai_client = get_openai_client()
    if openai_client is None:
        st.stop()
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    if chatbot is None:
        st.info("Please add Academic Catalog sources to get started.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ App Ready")
        with col2:
            st.success("✅ OpenAI OK")
        with col3:
            st.error("❌ No Sources")
        
        if st.button("🔄 Refresh After Adding Sources"):
            st.rerun()
        
        return
    
    # Display system info in sidebar
    
    keep_alive()


    # Display initial content if no messages
    if st.session_state.messages == []:
        # Available sources
        if chatbot.pdf_contents or chatbot.web_contents:
            st.subheader("📚 Available Sources")
            
            # Show PDF documents
            if chatbot.pdf_contents:
                st.markdown("**📄 PDF Documents:**")
                for doc_name in chatbot.pdf_contents.keys():
                    st.markdown(f'<div class="source-item">📄 {doc_name}</div>', unsafe_allow_html=True)
            
            # Show web sources
            if chatbot.web_contents:
                st.markdown("**🌐 Web Sources:**")
                for web_name in chatbot.web_contents.keys():
                    st.markdown(f'<div class="source-item">🌐 {web_name}</div>', unsafe_allow_html=True)
        
        # Sample questions
        st.subheader("💡 Try These Sample Questions")
        
        sample_questions = [
            "What are the general education requirements?",
            "What is the grading policy?",
            "Tell me about admission requirements",
            "What are the graduation requirements?",
            "Tell me about the Maryville Curriculum"
        ]
        
        # Two columns for sample questions
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    with st.spinner("🔍 Searching the Academic Catalog..."):
                        relevant_chunks = chatbot.context_aware_search(
                            question, 
                            conversation_history=st.session_state.messages
                        )
                        
                        answer = chatbot.generate_answer(
                            question, 
                            relevant_chunks, 
                            openai_client,
                            conversation_history=st.session_state.messages
                        )
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "system":
            st.markdown(f'<div style="text-align: center; color: #666; margin: 10px 0; font-style: italic;">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask Scottie about the Academic Catalog..."):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching the Academic Catalog..."):
                relevant_chunks = chatbot.context_aware_search(
                    question, 
                    conversation_history=st.session_state.messages
                )
                
                answer = chatbot.generate_answer(
                    question, 
                    relevant_chunks, 
                    openai_client,
                    conversation_history=st.session_state.messages
                )
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()