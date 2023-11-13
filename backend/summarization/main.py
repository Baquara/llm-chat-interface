import glob
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from accelerate import init_empty_weights
from chromadb.config import Settings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from optimum.gptq import load_quantized_model
from rich import print
from rich.markup import escape
from rich.panel import Panel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from typing_extensions import Annotated
from exl2 import ExllamaV2
from langchain.prompts import PromptTemplate

logger = logging.getLogger("chatdocs")


def get_embeddings(config: Dict[str, Any]) -> Embeddings:
    config = {**config["embeddings"]}
    config["model_name"] = config.pop("model")
    if config["model_name"].startswith("hkunlp/"):
        Provider = HuggingFaceInstructEmbeddings
    else:
        Provider = HuggingFaceEmbeddings
    return Provider(**config)


def get_vectorstore(config: Dict[str, Any]) -> VectorStore:
    embeddings = get_embeddings(config)
    config = config["chroma"]
    return Chroma(
        persist_directory=config["persist_directory"],
        embedding_function=embeddings,
        client_settings=Settings(**config),
    )


def get_vectorstore_from_documents(
    config: Dict[str, Any],
    documents: List[Document],
) -> VectorStore:
    embeddings = get_embeddings(config)
    config = config["chroma"]
    return Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=config["persist_directory"],
        client_settings=Settings(**config),
    )

def get_exllama_llm(config: Dict[str, Any]):

    # Example values for mandatory attributes
    client_value = "SomeClientObjectOrValue"
    model_path_value = './model/'

    # Create an instance of ExllamaV2 with mandatory parameters
    exllama_llm = ExllamaV2(
        model_path=model_path_value,
        temperature=0.70,
        top_k=20,
        top_p=0.9,
        typical=1.0,
        token_repetition_penalty=1.15,
        streaming=True
    )
    #https://github.com/turboderp/exllamav2/discussions/94


    return exllama_llm





def get_gptq_llm(config: Dict[str, Any]) -> LLM:
    logger.warning(
        "Using `llm: gptq` in `chatdocs.yml` is deprecated. "
        "Please use `llm: huggingface` instead as "
        "🤗 Transformers supports GPTQ models."
    )
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except ImportError:
        raise ImportError(
            "Could not import `auto_gptq` package. "
            "Please install it with `pip install chatdocs[gptq]`"
        )

    from transformers import (
        AutoTokenizer,
        TextGenerationPipeline,
        MODEL_FOR_CAUSAL_LM_MAPPING,
    )

    local_files_only = not config["download"]
    config = {**config["gptq"]}
    model_name_or_path = config.pop("model")
    model_file = config.pop("model_file", None)
    pipeline_kwargs = config.pop("pipeline_kwargs", None) or {}

    model_basename = None
    use_safetensors = False
    if model_file:
        model_basename = Path(model_file).stem
        use_safetensors = model_file.endswith(".safetensors")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        **config,
    )
    MODEL_FOR_CAUSAL_LM_MAPPING.register("chatdocs-gptq", model.__class__)
    pipeline = TextGenerationPipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs,
    )
    return HuggingFacePipeline(pipeline=pipeline)





def get_llm(
    config: Dict[str, Any],
    *,
    callback: Optional[Callable[[str], None]] = None,
) -> LLM:
    class CallbackHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            callback(token)
    #llm = get_gptq_llm(config)
    llm = get_exllama_llm(config)

    return llm





def get_retrieval_qa(
    config: Dict[str, Any],
    *,
    callback: Optional[Callable[[str], None]] = None,
) -> RetrievalQA:
    db = get_vectorstore(config)
    retriever = db.as_retriever(**config["retriever"])
    llm = get_llm(config, callback=callback)
    chain_type_kwargs = {}

    # Prepare and pass custom prompt if provided
    if "retriever" in config and "custom_prompt" in config["retriever"]:
        custom_prompt = config["retriever"]["custom_prompt"]

        chain_type_kwargs["prompt"] = PromptTemplate(
            template=custom_prompt, input_variables=["context", "question"]
        )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )



def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    result = d1.copy()
    for key, value in d2.items():
        if isinstance(value, dict):
            result[key] = merge_dicts(result.get(key, {}), value)
        else:
            result[key] = value
    return result

def get_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    if custom_config is None:
        return DEFAULT_CONFIG
    return merge_dicts(DEFAULT_CONFIG, custom_config)

DEFAULT_CONFIG = {
    "embeddings": {
        "model": "hkunlp/instructor-large"
    },
    "llm": "gptq",
    "gptq": {
        "model": "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ",
        "model_file": "model.safetensors",
        "pipeline_kwargs": {
            "max_new_tokens": 256
        }
    },
    "download": False,
    "host": "localhost",
    "port": 5000,
    "auth": False,
    "chroma": {
        "persist_directory": "db",
        "chroma_db_impl": "duckdb+parquet",
        "anonymized_telemetry": False
    },
    "retriever": {
        "search_kwargs": {
            "k": 6
        },
        "custom_prompt": """
        Use the following pieces of context to answer the question at the end. If the answer is not available in the provided context, just say that you don't know. Always start your responses saying "From the provided context,". Always try to provide long and elaborate answers (no less than 300 words), unless the user tells you otherwise.

        {context}

        Question: {question}
        Helpful Answer:
        """
    }
}


def print_answer(text: str) -> None:
    print(f"[bright_cyan]{escape(text)}", end="", flush=True)


def chat(config: Dict[str, Any], query: Optional[str] = None) -> None:
    qa = get_retrieval_qa(config, callback=print_answer)

    interactive = not query
    print()
    if interactive:
        print("Type your query below and press Enter.")
        print("Type 'exit' or 'quit' or 'q' to exit the application.\n")
    while True:
        print("[bold]Q: ", end="", flush=True)
        if interactive:
            query = input()
        else:
            print(escape(query))
        print()
        if query.strip() in ["exit", "quit", "q"]:
            print("Exiting...\n")
            break
        print("[bold]A:", end="", flush=True)

        res = qa(query)
        if config["llm"] != "ctransformers":
            print_answer(res["result"])

        print()
        for doc in res["source_documents"]:
            source, content = doc.metadata["source"], doc.page_content
            print(
                Panel(
                    f"[bright_blue]{escape(source)}[/bright_blue]\n\n{escape(content)}"
                )
            )
        print()

        if not interactive:
            break








# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf8"}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, docs in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(
    source_directory: str, ignored_files: List[str] = []
) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, "index")):
        if os.path.exists(
            os.path.join(persist_directory, "chroma-collections.parquet")
        ) and os.path.exists(
            os.path.join(persist_directory, "chroma-embeddings.parquet")
        ):
            list_index_files = glob.glob(os.path.join(persist_directory, "index/*.bin"))
            list_index_files += glob.glob(
                os.path.join(persist_directory, "index/*.pkl")
            )
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def add(config: Dict[str, Any], source_directory: str) -> None:
    persist_directory = config["chroma"]["persist_directory"]
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = get_vectorstore(config)
        collection = db.get()
        texts = process_documents(
            source_directory,
            [metadata["source"] for metadata in collection["metadatas"]],
        )
        print(f"Creating embeddings. May take a few minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents(source_directory)
        print(f"Creating embeddings. May take a few minutes...")
        db = get_vectorstore_from_documents(config, texts)
    db.persist()
    db = None



def add_documents():

    config = get_config()
    add(config=config, source_directory='./documents')

def download(config: Dict[str, Any]) -> None:
    config = {**config, "download": True}
    get_embeddings(config)
    get_llm(config)


def download_files():
    config = get_config()
    download(config=config)



config = get_config()

#download(config=config)


# Ensure the documents directory exists
os.makedirs('./documents', exist_ok=True)



add(config=config, source_directory='./documents')
chat(config=config)
