import os
import json
from pathlib import Path
import torch # Import torch to check for CUDA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sympy.physics.units import temperature

# --- Pre-run Checks ---
# 1. Ensure dependencies are installed in your virtual environment:
#    pip install langchain langchain-community langchain-huggingface faiss-cpu # or faiss-gpu if you have CUDA set up
#    pip install unstructured markdown # For document loading
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Or cu118 - INSTALL CORRECT PYTORCH VERSION!
#    pip install llama-cpp-python # Potentially reinstall with CUDA flags if needed (see above)
# 2. Ensure your embedding model files are in LLM_DIR
# 3. Ensure your GGUF model file is in LLM_DIR
# 4. Ensure your .md documents are in DOCS_DIR

# Project directory setup
PROJECT_DIR = Path.cwd()
LLM_DIR = PROJECT_DIR / "llm"
DOCS_DIR = PROJECT_DIR / "docs"
FAISS_INDEX_DIR = PROJECT_DIR / "faiss_index"

# Ensure directories exist
os.makedirs(LLM_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# --- CUDA Check ---
is_cuda_available = torch.cuda.is_available()
print("-" * 30)
print(f"CUDA Available: {is_cuda_available}")
if is_cuda_available:
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
else:
    print("CUDA not available or PyTorch was installed without CUDA support.")
    print("Will use CPU for embeddings and potentially LLM.")
print("-" * 30)


# --- Embedding Model Loading ---
# --- Embedding Model Loading (Relying on Hugging Face Cache) ---
embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Определяем устройство
is_cuda_available = torch.cuda.is_available()
embedding_device = "cuda" if is_cuda_available else "cpu"
print(f"Using device: {embedding_device} for embeddings")

print(f"Loading embedding model '{embedding_model_name}' using Hugging Face cache...")

try:
    # Просто загружаем по имени. Langchain/HuggingFace сами найдут модель
    # в кэше или скачают ее туда, если ее там нет.
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": embedding_device}
    )
    print("Embedding model loaded successfully (from cache or downloaded).")
except Exception as e:
    print(f"\n------\nERROR: Failed to load or download embedding model '{embedding_model_name}'.")
    print(f"Error details: {e}")
    print("\nTroubleshooting tips:")
    print("- Check your internet connection.")
    print("- Ensure you have sufficient disk space in your user profile's cache directory (usually C:\\Users\\YourUsername\\.cache\\huggingface).")
    print("- Check firewall/antivirus settings.")
    print("- Try clearing the Hugging Face cache manually (delete the folder above) and run again.")
    print("------\n")
    raise e


# --- Document Loading ---
# Requires: pip install unstructured markdown
print(f"\nLoading documents from: {DOCS_DIR}")
loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True,
    use_multithreading=True, # Can speed up loading
    # Optional: Add specific unstructured kwargs if needed
    # loader_kwargs={"mode": "elements"}
)

try:
    documents = loader.load()
except ImportError as e:
     print("\n------\nERROR: Missing dependencies for document loading.")
     print("Please ensure 'unstructured' and 'markdown' are installed:")
     print("pip install unstructured markdown")
     print(f"Original error: {e}")
     print("------\n")
     raise SystemExit(1) # Exit script
except Exception as e:
     print(f"An error occurred during document loading: {e}")
     # Handle other potential loading errors (e.g., file permissions, malformed files)
     raise e

if not documents:
    # Check if the directory is empty or if files were filtered out
    md_files_found = list(DOCS_DIR.glob("**/*.md"))
    if not md_files_found:
        raise ValueError(f"No .md files found in the directory tree under {DOCS_DIR}. Please add Markdown files.")
    else:
         # This case might mean files exist but couldn't be loaded/parsed.
         # The loader usually raises an error earlier if parsing fails.
         print(f"Warning: Found {len(md_files_found)} .md files but loader returned no documents. Check file contents, permissions, and potential parsing errors shown above.")
         raise ValueError("Document loading resulted in an empty list, though .md files seem to exist.")

print(f"Loaded {len(documents)} documents.")

# --- Text Splitting ---
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]
# text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# chunks = text_splitter.split_text(documents)


print(f"Split into {len(chunks)} chunks.")

# --- Vector Store ---
print("\nCreating or loading FAISS vector store...")
faiss_index_path_str = str(FAISS_INDEX_DIR) # Use string representation

# Check if index exists and is not empty
index_exists = FAISS_INDEX_DIR.exists() and \
               (FAISS_INDEX_DIR / "index.faiss").exists() and \
               (FAISS_INDEX_DIR / "index.pkl").exists()

if index_exists:
    print(f"Loading existing FAISS index from {faiss_index_path_str}")
    try:
        vectorstore = FAISS.load_local(
            faiss_index_path_str,
            embeddings,
            allow_dangerous_deserialization=True # Be cautious with this flag
        )
        print("FAISS index loaded.")
    except Exception as e:
        print(f"Error loading existing FAISS index: {e}. Recreating index.")
        index_exists = False # Force recreation

if not index_exists:
    print("Creating new FAISS index...")
    if not chunks:
         raise ValueError("Cannot create FAISS index: No document chunks were generated.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS index created. Saving...")
    vectorstore.save_local(faiss_index_path_str)
    print(f"FAISS index saved to {faiss_index_path_str}")

# --- LLM Loading ---
gguf_model_name = "Llama-3-Groq-8B-Tool-Use-Q4_K_M.gguf" # Your GGUF file name
gguf_path = LLM_DIR / gguf_model_name
print(f"\nChecking for GGUF model at: {gguf_path}")

if not gguf_path.exists():
    # Provide a more helpful error message
    print("\n------\nERROR: GGUF Model Not Found!")
    print(f"The required model file '{gguf_model_name}' was not found in the directory:")
    print(f"{LLM_DIR}")
    print("\nPlease ensure the GGUF model file is present in that location.")
    print("------\n")
    raise FileNotFoundError(f"GGUF model not found at {gguf_path}")

print("Loading LlamaCpp model...")

# Determine GPU layers based on CUDA availability AND user preference
# Set to -1 to use GPU if available (offload all possible layers), 0 for CPU only
n_gpu_layers = -1 if is_cuda_available else 0
# You can manually override this:
# n_gpu_layers = 0 # Force CPU
# n_gpu_layers = 20 # Use 20 layers on GPU if available

if n_gpu_layers > 0 and is_cuda_available:
    print(f"Attempting to offload {n_gpu_layers} layers to GPU.")
elif n_gpu_layers == -1 and is_cuda_available:
     print(f"Attempting to offload all possible layers to GPU (-1).")
else:
    if n_gpu_layers > 0:
         print(f"GPU offload requested ({n_gpu_layers} layers), but CUDA is not available. Using CPU.")
    else:
         print("Running LlamaCpp on CPU (n_gpu_layers=0).")
    n_gpu_layers = 0 # Ensure it's 0 if CUDA not available

try:
    llm = LlamaCpp(
        model_path=str(gguf_path),
        n_ctx=8192,      # Model's context window size
        n_gpu_layers=n_gpu_layers, # Set based on CUDA check and preference
        n_batch=1024,     # Adjust based on VRAM/RAM. Try reducing if OOM errors occur.
        verbose=True,    # Logs details during setup and generation
        # Optional generation parameters:
        temperature=0.1,
        top_p=0.9,
        # stop=["\nHuman:", "\nAI:"] # Example stop tokens
    )
    print("LlamaCpp model loaded successfully.")
except Exception as e:
    print(f"\n------\nERROR: Failed to load LlamaCpp model.")
    print(f"Error details: {e}")
    print(f"Model path used: {gguf_path}")
    print(f"GPU layers attempted: {n_gpu_layers} (Set based on CUDA availability)")
    print("\nTroubleshooting tips:")
    print("- Ensure 'llama-cpp-python' is installed correctly.")
    print("- If using GPU (n_gpu_layers > 0):")
    print("  - Verify CUDA-enabled PyTorch is installed and working (`torch.cuda.is_available()`).")
    print("  - Consider reinstalling `llama-cpp-python` with CUDA flags (see notes at top of script).")
    print("  - Check VRAM usage; reduce `n_gpu_layers` or `n_batch` if necessary.")
    print("- Verify the GGUF file is not corrupted and is compatible with your `llama-cpp-python` version.")
    print("- Ensure 'n_ctx' does not exceed the model's maximum context size.")
    print("------\n")
    raise e


from langchain.prompts import PromptTemplate

# Создаем шаблон промпта на русском языке
prompt_template = """Используя следующий контекст, подробно ответь на вопрос в конце.
Отвечай ТОЛЬКО на русском языке. Не используй другие языки.
Если контекст не содержит ответа на вопрос, честно скажи, что информация не найдена в предоставленных документах.
Не придумывай информацию, которой нет в контексте.

Контекст:
{context}

Вопрос: {question}

Ответ на русском языке:"""

# Создаем объект PromptTemplate
RUSSIAN_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Передаем кастомный промпт в chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    # retriever=vectorstore.as_retriever(search_kwargs={"k": 6}), # Оставляем k=3 или настраиваем
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 20} # fetch_k - сколько изначально выбрать, k - сколько вернуть после MMR
    ),
    # temperature = 0.3,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": RUSSIAN_PROMPT}, # <--- Вот здесь добавляем промпт
    verbose=True # Можно установить True для отладки промптов
)
print("Chain created with custom Russian prompt.")




# --- RAG Chain ---
# print("\nCreating Conversational Retrieval Chain...")
# chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 3 chunks
#     return_source_documents=True,
#     # Optional: Customize how chat history is condensed or how docs are combined
#     # condense_question_prompt=...,
#     # combine_docs_chain_kwargs={"prompt": ...},
#     verbose=False # Set to True for detailed chain logging
# )
print("Chain created.")

# --- Interactive Query Loop ---
print("\n--- RAG Chatbot Ready ---")
chat_history = []
while True:
    try:
        query = input("Введите ваш запрос на русском (или 'выход' для завершения): ")
        query = query.strip() # Remove leading/trailing whitespace

        if query.lower() == "выход":
            print("Завершение работы.")
            break
        if not query:
            print("Пожалуйста, введите запрос.")
            continue

        print("Processing query...")
        # Run the chain
        result = chain.invoke({"question": query, "chat_history": chat_history}) # Use invoke for newer Langchain

        # Print the response
        print("\nОтвет:", result["answer"])

        # Print source document snippets
        if result.get("source_documents"):
            print("\nИсточники:")
            seen_sources = set()
            source_count = 0
            for doc in result["source_documents"]:
                 source_path = doc.metadata.get('source', 'Unknown Source')
                 source_name = Path(source_path).name # Get only filename
                 if source_name not in seen_sources: # Avoid printing same source multiple times if k > 1 retrieves from same doc
                      source_count += 1
                      # Limit snippet length for readability
                      snippet = doc.page_content.replace('\n', ' ').strip()[:250] + "..."
                      print(f" [{source_count}] From '{source_name}': {snippet}")
                      seen_sources.add(source_name)

        else:
            print("\nИсточники: Не найдены (возможно, ответ сгенерирован без извлеченных документов).")


        # Update chat history (important: Use the actual question and answer)
        chat_history.append((query, result["answer"]))

        # Optional: Limit chat history length to prevent excessive context/token usage
        # max_history_length = 5 # Keep last 5 pairs
        # if len(chat_history) > max_history_length:
        #     chat_history = chat_history[-max_history_length:]

    except KeyboardInterrupt:
        print("\nЗавершение работы по запросу пользователя (Ctrl+C).")
        break
    except Exception as e:
        print(f"\n------\nПроизошла ошибка во время обработки запроса:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("------\nПродолжение работы...")
        # Optionally decide whether to continue or break the loop on error
        # break