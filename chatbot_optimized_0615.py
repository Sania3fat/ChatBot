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
from typing import List, Dict
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import time
import requests
import glob
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

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
    
    # Institutional PDF directory
    INSTITUTIONAL_PDF_DIR = "institutional_pdfs"
    CACHE_DIR = "institutional_cache"

# --- HELPER FUNCTIONS & CLASSES ---

class InstitutionalPDFChatbot:
    """Optimized PDF chatbot for institutional deployment with pre-loaded documents."""
    
    def __init__(self):
        self.pdf_contents: Dict[str, str] = {}
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
                
                # Load model with optimized settings
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
                'files_count': len(self.pdf_contents),
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
        """Optimized PDF text extraction."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Optimized text cleaning
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page_text)
                page_text = re.sub(r'^\d+\s*$', '', page_text, flags=re.MULTILINE)
                
                text += page_text + "\n\n"
            
            doc.close()
            
            # Final cleaning
            text = text.strip()
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            if len(text) > 100:
                self.pdf_contents[filename] = text
            
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")

    def load_institutional_pdfs(self) -> bool:
        """Load all PDFs from the institutional directory."""
        self.pdf_contents = {}
        
        # Look for PDF files in the institutional directory
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        
        if not pdf_files:
            return False
        
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            self.extract_text_from_pdf(pdf_path, filename)
        
        return len(self.pdf_contents) > 0

    def smart_chunk_text(self, text: str, source: str) -> List[Dict]:
        """Optimized text chunking."""
        chunks = []
        
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
            
            if current_length + sentence_length > Config.CHUNK_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                # Add overlap
                if chunks and Config.CHUNK_OVERLAP > 0:
                    prev_chunk_words = chunks[-1]['text'].split()
                    overlap_words = prev_chunk_words[-min(Config.CHUNK_OVERLAP//5, len(prev_chunk_words)):]
                    chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
                
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': len(chunks)
                })
                
                # Start new chunk with overlap
                if Config.CHUNK_OVERLAP > 0:
                    overlap_sentences = current_chunk[-min(2, len(current_chunk)):]
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
            if chunks and Config.CHUNK_OVERLAP > 0:
                prev_chunk_words = chunks[-1]['text'].split()
                overlap_words = prev_chunk_words[-min(Config.CHUNK_OVERLAP//5, len(prev_chunk_words)):]
                chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
            
            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': len(chunks)
            })
        
        return chunks

    def _create_tfidf_index(self):
        """Create optimized TF-IDF index."""
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

    def create_chunks_and_embeddings(self) -> None:
        """Optimized chunk and embedding creation."""
        self.text_chunks = []
        if not self.pdf_contents:
            return False

        # Create chunks
        for filename, content in self.pdf_contents.items():
            file_chunks = self.smart_chunk_text(content, filename)
            self.text_chunks.extend(file_chunks)
        
        if not self.text_chunks:
            return False

        # Generate embeddings with optimized batching
        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        try:
            embeddings_list = []
            
            for i in range(0, len(chunk_texts), Config.BATCH_SIZE):
                batch = chunk_texts[i:i + Config.BATCH_SIZE]
                
                # Optimized retry logic
                for attempt in range(Config.MAX_RETRIES):
                    try:
                        batch_embeddings = self.embedding_model.encode(
                            batch, 
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            batch_size=len(batch),
                            normalize_embeddings=True  # Normalize for better similarity
                        )
                        embeddings_list.append(batch_embeddings)
                        break
                    except Exception as e:
                        if attempt == Config.MAX_RETRIES - 1:
                            raise e
                        else:
                            time.sleep(2 ** attempt)
            
            # Combine all embeddings
            self.chunk_embeddings = torch.cat(embeddings_list, dim=0)
            
            # Create TF-IDF index
            self._create_tfidf_index()
            
            return True
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return False

    def hybrid_search(self, question: str) -> List[Dict]:
        """Optimized hybrid search with fixed parameters."""
        if self.chunk_embeddings is None or len(self.text_chunks) == 0:
            return []
            
        try:
            # Encode question with retry
            question_embedding = None
            for attempt in range(Config.MAX_RETRIES):
                try:
                    question_embedding = self.embedding_model.encode(
                        question, 
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                    break
                except Exception as e:
                    if attempt == Config.MAX_RETRIES - 1:
                        return []
                    time.sleep(2 ** attempt)
            
            # Semantic similarity
            semantic_scores = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
            
            # Keyword search
            keyword_scores = np.zeros(len(self.text_chunks))
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                question_tfidf = self.tfidf_vectorizer.transform([question])
                keyword_similarities = cosine_similarity(question_tfidf, self.tfidf_matrix)[0]
                keyword_scores = keyword_similarities
            
            # Optimized score combination
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            semantic_scores_norm = (semantic_scores.cpu().numpy() + 1) / 2
            keyword_scores_norm = keyword_scores
            
            combined_scores = (semantic_weight * semantic_scores_norm + 
                             keyword_weight * keyword_scores_norm)
            
            # Get top results with fixed count
            top_indices = np.argsort(combined_scores)[-Config.SEARCH_RESULTS:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.text_chunks[idx].copy()
                chunk['semantic_score'] = semantic_scores[idx].item()
                chunk['keyword_score'] = keyword_scores[idx]
                chunk['combined_score'] = combined_scores[idx]
                relevant_chunks.append(chunk)
            
            # Filter low scores
            relevant_chunks = [chunk for chunk in relevant_chunks 
                             if chunk['combined_score'] > 0.15]  # Slightly higher threshold
            
            return relevant_chunks
            
        except Exception as e:
            return []

    def generate_answer(self, question: str, context_chunks: List[Dict], client: OpenAI) -> str:
        """Optimized answer generation."""
        if not context_chunks:
            return "I couldn't find relevant information in the institutional documents to answer your question."
        
        # Prepare context efficiently
        context_str = ""
        sources = set()
        
        for i, chunk in enumerate(context_chunks[:8]):  # Limit context for efficiency
            context_str += f"=== Source {i+1} ({chunk['source']}) ===\n"
            context_str += f"{chunk['text']}\n\n"
            sources.add(chunk['source'])
        
        # Optimized prompt
        prompt = f"""Answer the question based on the provided institutional documents.

CONTEXT:
{context_str}

QUESTION: {question}

Provide a comprehensive answer using only the information above. Cite sources in brackets [document.pdf]. If insufficient information is available, state this clearly.

Answer:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an institutional knowledge assistant. Answer questions accurately based on provided documents and always cite sources."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,  # Reduced for efficiency
                temperature=0.1,
                top_p=0.95,
                frequency_penalty=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"


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
        
        # Create identifier for institutional documents
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        if not pdf_files:
            # Show setup instructions instead of stopping
            st.warning(f"⚠️ No PDF files found in `{Config.INSTITUTIONAL_PDF_DIR}/` directory.")
            
            with st.expander("📋 Setup Instructions", expanded=True):
                st.markdown("""
                **To set up the institutional knowledge base:**
                
                1. Create a folder named `institutional_pdfs` in the same directory as this app
                2. Add your PDF documents to this folder
                3. Refresh the page
                
                **Example folder structure:**
                ```
                your_app/
                ├── institutional_pdfs/
                │   ├── student_handbook.pdf
                │   ├── academic_policies.pdf
                │   └── course_catalog.pdf
                └── your_streamlit_app.py
                ```
                """)
            
            # Create the directory if it doesn't exist
            os.makedirs(Config.INSTITUTIONAL_PDF_DIR, exist_ok=True)
            
            # Return a dummy chatbot to allow the app to continue
            st.session_state.chatbot = None
            st.session_state.chatbot_initialized = False
            return None
        
        with st.spinner("Initializing knowledge base..."):
            # Create cache identifier
            file_stats = []
            for pdf_path in sorted(pdf_files):
                stat = os.stat(pdf_path)
                file_stats.append(f"{os.path.basename(pdf_path)}-{stat.st_size}-{stat.st_mtime}")
            
            identifier = "institutional_" + hashlib.sha256("|".join(file_stats).encode('utf-8')).hexdigest()
            
            # Try to load from cache first
            if chatbot.load_from_cache(identifier):
                st.success("✅ Knowledge base loaded from cache")
            else:
                # Load and process documents
                if chatbot.load_institutional_pdfs():
                    if chatbot.create_chunks_and_embeddings():
                        chatbot.save_to_cache(identifier)
                        st.success("✅ Knowledge base initialized and cached")
                    else:
                        st.error("Failed to create embeddings")
                        return None
                else:
                    st.error("Failed to load institutional documents")
                    return None
            
            st.session_state.chatbot = chatbot
            st.session_state.chatbot_initialized = True
            st.session_state.messages = []
            
    return st.session_state.chatbot

def main():
    st.title("🎓 Institutional Knowledge Assistant")
    st.markdown("*Ask questions about institutional documents and policies*")

    # Initialize OpenAI client
    openai_client = get_openai_client()
    if openai_client is None:
        st.stop()
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    if chatbot is None:
        # Show a helpful message instead of stopping
        st.info("👆 Please follow the setup instructions above to add PDF documents.")
        
        # Show some demo content
        st.subheader("🔧 System Status")
        st.write("✅ Application loaded successfully")
        st.write("✅ OpenAI client initialized")
        st.write("❌ No institutional documents found")
        
        # Add refresh button
        if st.button("🔄 Refresh After Adding PDFs"):
            st.rerun()
        
        return
    
    # Display system info in collapsed sidebar
    with st.sidebar:
        st.header("📊 System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(chatbot.pdf_contents))
            st.metric("Knowledge Chunks", len(chatbot.text_chunks))
        with col2:
            st.metric("Chat Messages", len(st.session_state.messages))
            st.metric("Search Results", Config.SEARCH_RESULTS)
        
        st.markdown("---")
        st.markdown("**Configuration:**")
        st.text(f"Chunk Size: {Config.CHUNK_SIZE}")
        st.text(f"Overlap: {Config.CHUNK_OVERLAP}")
        st.text(f"Model: {Config.MODEL_NAME}")
        
        if st.session_state.messages:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    # Display available documents
    if st.session_state.messages == []:
        st.subheader("📚 Available Documents")
        doc_names = list(chatbot.pdf_contents.keys())
        if doc_names:
            cols = st.columns(min(3, len(doc_names)))
            for i, doc_name in enumerate(doc_names):
                with cols[i % 3]:
                    st.info(f"📄 {doc_name}")
        
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the admission requirements?",
            "What is the grading policy?",
            "How do I apply for financial aid?",
            "What are the graduation requirements?",
            "What support services are available?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}"):
                    # Simulate user asking the question
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask about institutional documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                relevant_chunks = chatbot.hybrid_search(question)
                answer = chatbot.generate_answer(question, relevant_chunks, openai_client)
                st.markdown(answer)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()