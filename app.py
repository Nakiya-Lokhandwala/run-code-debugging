import streamlit as st
import os
import faiss
import pandas as pd
import numpy as np
import subprocess
import tempfile
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import dotenv
from google import genai
from google.genai import types

# Load environment variables from the .env file (for local deployment)
dotenv.load_dotenv()

# --- CONFIGURATION & PATHS ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "deployment" / "data"
MODELS_DIR = BASE_DIR / "deployment" / "models" / "sentence_embedder" 

@st.cache_resource
def load_assets():
    """Loads FAISS index, dataset, and Sentence Transformer model."""
    try:
        if not DATA_DIR.exists() or not MODELS_DIR.exists():
            st.error("Deployment files not found. Ensure the 'deployment' folder is present and correctly structured.")
            return None, None, None
            
        index = faiss.read_index(str(DATA_DIR / "faiss_index.index"))
        df = pd.read_csv(DATA_DIR / "cleaned_stackoverflow.csv")
        embedder = SentenceTransformer(str(MODELS_DIR))
        return index, df, embedder
    except Exception as e:
        st.error(f"Error loading assets. Check 'deployment' folder structure and file integrity. Error: {e}")
        return None, None, None

# Load all RAG assets
with st.spinner("Initializing Knowledge Base and AI Assistant..."): 
    FAISS_INDEX, DATASET, EMBEDDER = load_assets()

if FAISS_INDEX is None:
    st.stop() 

# Initialize Gemini client
try:
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    if not api_key: raise ValueError("API key not set.")
    CLIENT = genai.Client(api_key=api_key)
except Exception as e:
    st.error("AI Assistant Key Error: The key is invalid or not found.")
    st.warning("Please ensure your API Key is set in the `.env` file or Streamlit secrets.")
    st.stop()

# --- CORE FUNCTIONS ---

def execute_code(code: str) -> str:
    """Execute user code safely and capture errors. (Includes Windows VENV Fix)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp:
        temp.write(code)
        temp_path = temp.name
    try:
        python_executable = sys.executable 
        result = subprocess.run(
            [python_executable, temp_path], 
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            return "Code ran successfully!\n" + result.stdout
        else:
            return "Error occurred:\n" + result.stderr
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def retrieve_similar(error_msg: str, top_k=3):
    """Retrieve similar Q&A from dataset using FAISS."""
    query_vec = EMBEDDER.encode([error_msg], convert_to_numpy=True).astype('float32')
    _, indices = FAISS_INDEX.search(query_vec, top_k)
    results = []
    for i in indices[0]:
        q = DATASET.iloc[i]['input_text']
        a = DATASET.iloc[i]['output_text']
        results.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(results)

def generate_fix(prompt: str):
    """Generates content using the AI model."""
    content = types.Content(role='user', parts=[types.Part.from_text(text=prompt)])
    response = CLIENT.models.generate_content(
        model="gemini-2.5-flash", contents=content
    )
    return response.text

def debug_code(user_code: str):
    """Main debugging logic combining code execution, RAG, and AI generation."""
    st.info("Executing Code...") 
    exec_result = execute_code(user_code)
    st.markdown("---")
    st.subheader("Execution Result (stdout/stderr)")
    st.code(exec_result, language="text")
    st.markdown("---")

    if "Code ran successfully!" not in exec_result:
        # RAG Retrieval
        error_lines = exec_result.split("Error occurred:\n")[-1]
        retrieved_context = retrieve_similar(error_lines)
        
        # Generation Prompt
        prompt = f"""
You are an expert Python debugging assistant. Your goal is to provide a single, unified, highly readable response using the following strict structure. Use a conversational tone like a human assistant.

**IMPORTANT:** Use the provided Stack Overflow context to inform your explanation and solution.

1.  **Error Explanation:** Start immediately by explaining the error clearly and stating the fundamental Python rule being violated.
2.  **Fix:** Explain the necessary fix (e.g., type conversion). Use an example snippet if helpful.
3.  **Corrected Code:** Present the complete, runnable corrected code in a Python code block (```python ... ```).
4.  **Brief Trace:** Explain the trace of the corrected code and state the final expected output.

User code:
{user_code}

Execution Error:
{exec_result}

Stack Overflow Context (Use this for reliable fixes):
{retrieved_context}
"""
        # Spinner rephrased to emphasize Knowledge Base/Context
        with st.spinner("Searching Knowledge Base and Generating Fix..."):
            result = generate_fix(prompt)
        return result

    else:
        # Code ran successfully path
        prompt = f"You are an expert Python assistant. The following code ran successfully: {user_code}. The program output was: {exec_result.replace('Code ran successfully!', '').strip()}. Explain what this code does, step by step."
        
        # Spinner rephrased for explanation
        with st.spinner("Generating Code Explanation..."):
            result = generate_fix(prompt)
        return result

# --- STREAMLIT UI ---

st.title("Python Code Debugging Assistant")
st.markdown("Paste your Python code below to debug errors or get an explanation.")
st.markdown("This assistant combines **AI reasoning** with a **Vast Code Knowledge Base**.")

# Default code to show a typical error
example_code = """
age = "25"
print(age + 5)
"""

user_code = st.text_area(
    "Paste your Python code here:",
    height=250,
    value=example_code,
    key="code_input"
)

if st.button("Debug / Explain Code", type="primary", use_container_width=True):
    if user_code.strip() == "":
        st.warning("Please enter some Python code to proceed.")
    else:
        with st.container():
            st.subheader("Assistant Analysis")
            # The final result is generated without explicit mention of 'Gemini' in the status
            gemini_response = debug_code(user_code)
            st.markdown(gemini_response)

st.sidebar.header("Knowledge Base Info")
if DATASET is not None:
    st.sidebar.markdown(f"**Total Q&A Pairs:** {len(DATASET):,}")
if FAISS_INDEX:
    st.sidebar.markdown(f"**FAISS Index Size:** {FAISS_INDEX.ntotal:,}")