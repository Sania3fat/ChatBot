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
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Configure for embedding - hide Streamlit UI elements when embedded
hide_streamlit_style = """
<style>
    /* Hide the Streamlit header and menu */
    header[data-testid="stHeader"] {
        height: 0px;
        visibility: hidden;
    }
    
    /* Hide the footer */
    footer[data-testid="stFooter"] {
        visibility: hidden;
    }
    
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Hide sidebar toggle button when collapsed */
    button[data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Optional: Hide the main menu hamburger */
    #MainMenu {
        visibility: hidden;
    }
    
    /* Optional: Remove "Made with Streamlit" footer */
    footer:after {
        content: "";
        visibility: hidden;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
</style>
"""

# Check if running in embedded mode
try:
    query_params = st.query_params
    if query_params.get("embed") or query_params.get("embedded"):
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
except AttributeError:
    # Fallback for older Streamlit versions
    try:
        query_params = st.experimental_get_query_params()
        if query_params.get("embed") or query_params.get("embedded"):
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    except:
        # If query params don't work, apply styles anyway for embedded use
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
    
    # Context-aware settings
    MAX_CONVERSATION_TOKENS = 2000  # Max tokens to use for conversation history
    CONTEXT_WINDOW_SIZE = 10  # Number of previous messages to consider

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

    def load_institutional_pdfs(self) -> bool:
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
        
        return len(self.pdf_contents) > 0

    def smart_chunk_text(self, text: str, source: str) -> List[Dict]:
        """
        Improved chunking that preserves semantic boundaries - consistent with first chatbot.
        """
        chunks = []
        
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
            if current_length + sentence_length > Config.CHUNK_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                # Add overlap from previous chunk if it exists
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
        """Process PDF content into chunks and generate embeddings - consistent with first chatbot."""
        self.text_chunks = []
        if not self.pdf_contents:
            st.error("No PDF content available.")
            return False

        # Create chunks
        for filename, content in self.pdf_contents.items():
            file_chunks = self.smart_chunk_text(content, filename)
            self.text_chunks.extend(file_chunks)
        
        if not self.text_chunks:
            st.error("No chunks created from PDFs.")
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

    def context_aware_search(self, question: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """
        Enhanced search that considers conversation context.
        """
        if self.chunk_embeddings is None or len(self.text_chunks) == 0:
            st.warning("No embeddings available for search.")
            return []
        
        # Extract context keywords from conversation
        context_keywords = []
        if conversation_history:
            context_keywords = self.extract_context_keywords(conversation_history)
        
        # Enhance the question with context if it's a follow-up
        enhanced_question = question
        if context_keywords and any(word in question.lower() for word in ['that', 'this', 'it', 'those', 'these', 'more', 'else']):
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
            
            # Get top results
            top_indices = np.argsort(combined_scores)[-Config.SEARCH_RESULTS:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.text_chunks[idx].copy()
                chunk['semantic_score'] = semantic_scores[idx].item()
                chunk['keyword_score'] = keyword_scores[idx]
                chunk['combined_score'] = combined_scores[idx]
                relevant_chunks.append(chunk)
            
            # Filter out very low scores
            relevant_chunks = [chunk for chunk in relevant_chunks 
                             if chunk['combined_score'] > 0.1]
            
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
        """Generate answer with conversation context awareness."""
        if not context_chunks:
            return "I couldn't find relevant information in the PDF documents to answer your question."
        
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
        
        # Enhanced context-aware prompt
        prompt = f"""You are an expert document analyst with access to institutional PDF documents. You are having an ongoing conversation with a user.

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
8. If this is a follow-up question, acknowledge the connection to the previous discussion

Answer:"""

        try:
            # Build message history for better context
            messages = [
                {"role": "system", "content": "You are a helpful institutional knowledge assistant that maintains context across conversations. You answer questions based on provided PDF documents while keeping track of the conversation flow."}
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
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2000,
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
                    success = chatbot.create_chunks_and_embeddings()
                    if success:
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
        if message["role"] == "system":
            # Display system messages differently
            st.markdown(f"<div style='text-align: center; color: gray; margin: 20px 0;'>{message['content']}</div>", 
                       unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask about institutional documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response with context awareness
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
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