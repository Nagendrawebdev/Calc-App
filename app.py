import os
import uuid
import re
from typing import Optional, List, Dict

# Optional heavy deps: import if available; keep functions robust if missing.
try:
    import fitz  # pymupdf
except Exception:
    fitz = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import pandas as pd
except Exception:
    pd = None

# Optional summarizer
summarizer = None
try:
    from transformers import pipeline
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception:
        summarizer = None
except Exception:
    summarizer = None

from fastapi import FastAPI, File, UploadFile, HTTPException

APP_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(APP_DIR, "data", "uploads")

# Configure external tool paths from environment if provided
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
POPPLER_PATH = os.environ.get("POPPLER_PATH")
if pytesseract and TESSERACT_CMD:
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    except Exception:
        pass


def extract_text_from_pdf(path: str) -> List[Dict]:
    if fitz is None:
        raise RuntimeError("pymupdf (fitz) is not installed")
    doc = fitz.open(path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def ocr_pdf(path: str, dpi: int = 300) -> List[Dict]:
    if convert_from_path is None or pytesseract is None:
        raise RuntimeError("pdf2image and/or pytesseract are not installed/configured")
    # Pass poppler_path if configured
    convert_args = {"dpi": dpi}
    if POPPLER_PATH:
        convert_args["poppler_path"] = POPPLER_PATH
    images = convert_from_path(path, **convert_args)
    pages = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        pages.append({"page": i + 1, "text": text})
    return pages


def extract_tables(path: str) -> Dict[int, List]:
    if pdfplumber is None or pd is None:
        raise RuntimeError("pdfplumber and/or pandas are not installed")
    tables_by_page = {}
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                tables_by_page[i + 1] = [pd.DataFrame(t[1:], columns=t[0]) for t in tables]
    return tables_by_page


def heuristics_split_into_sections(pages: List[Dict]) -> List[Dict]:
    sections = []
    for p in pages:
        text = p.get("text", "")
        chunks = re.split(r"\n(?=[A-Z0-9 \-]{5,}\n)", text)
        for chunk in chunks:
            clean = chunk.strip()
            if clean:
                sections.append({"page": p.get("page"), "content": clean[:5000]})
    return sections


def create_chunks(sections: List[Dict], chunk_size: int = 500) -> List[Dict]:
    chunks = []
    for s in sections:
        text = s.get("content", "")
        for i in range(0, len(text), chunk_size):
            chunks.append({"page": s.get("page"), "text": text[i:i + chunk_size]})
    return chunks


def summarize_text(text: str, max_length: int = 150, min_length: int = 40) -> Optional[str]:
    if summarizer is None:
        return None
    if len(text) > 2000:
        text = text[:2000]
    try:
        result = summarizer(text, max_length=max_length, min_length=min_length)
        return result[0].get("summary_text") if result else None
    except Exception:
        return None


app = FastAPI()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    doc_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    try:
        with open(filepath, "wb") as f:
            f.write(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"id": doc_id, "status": "uploaded"}


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    filepath = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Document not found")
    stat = os.stat(filepath)
    return {
        "id": doc_id,
        "found": True,
        "size": stat.st_size,
        "path": filepath,
        "text_extracted": False,
        "tables_extracted": False,
    }
# requirements: pymupdf
import fitz  # pip install pymupdf

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text})
    doc.close()
    return pages

# requirements: pytesseract, pdf2image, pillow
# ensure tesseract is installed on the machine
from pdf2image import convert_from_path
import pytesseract

def ocr_pdf(path, dpi=300):
    images = convert_from_path(path, dpi=dpi)
    pages = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        pages.append({"page": i+1, "text": text})
    return pages

# requirements: pdfplumber, pandas
import pdfplumber
import pandas as pd

def extract_tables(path):
    tables_by_page = {}
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                tables_by_page[i+1] = [pd.DataFrame(t[1:], columns=t[0]) for t in tables]
    return tables_by_page

import re

def heuristics_split_into_sections(pages):
    sections = []
    for p in pages:
        text = p["text"]
        # Very simple: split on headings like lines in ALL CAPS or lines ending with ':'
        chunks = re.split(r'\n(?=[A-Z0-9 \-]{5,}\n)', text)  # crude heading detection
        for chunk in chunks:
            clean = chunk.strip()
            if clean:
                sections.append({"page": p["page"], "content": clean[:5000]})  # limit size
    return sections

# requirements: transformers, torch
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # example

def summarize_text(text, max_length=150, min_length=40):
    # chunk if needed
    if len(text) > 2000:
        text = text[:2000]
    summary = summarizer(text, max_length=max_length, min_length=min_length)[0]['summary_text']
    # requirements: pymupdf
    import fitz  # pip install pymupdf
    from pdf2image import convert_from_path
    import pytesseract
    import pdfplumber
    import pandas as pd
    import re
    from fastapi import FastAPI, File, UploadFile
    import uuid, os
    from typing import Optional

    # Try to import transformers and initialize summarizer optionally.
    # If transformers/torch aren't installed or initialization fails, keep summarizer None.
    summarizer = None
    try:
        from transformers import pipeline
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception:
            # large model download/initialization may fail in constrained environments
            summarizer = None
    except Exception:
        summarizer = None


    def extract_text_from_pdf(path: str):
        doc = fitz.open(path)
        pages = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append({"page": i + 1, "text": text})
        doc.close()
        return pages


    def ocr_pdf(path: str, dpi: int = 300):
        images = convert_from_path(path, dpi=dpi)
        pages = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            pages.append({"page": i + 1, "text": text})
        return pages


    def extract_tables(path: str):
        tables_by_page = {}
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    tables_by_page[i + 1] = [pd.DataFrame(t[1:], columns=t[0]) for t in tables]
        return tables_by_page


    def heuristics_split_into_sections(pages):
        sections = []
        for p in pages:
            text = p["text"]
            # Very simple: split on headings like lines in ALL CAPS or lines ending with ':'
            chunks = re.split(r'\n(?=[A-Z0-9 \-]{5,}\n)', text)  # crude heading detection
            for chunk in chunks:
                clean = chunk.strip()
                if clean:
                    sections.append({"page": p["page"], "content": clean[:5000]})
        return sections


    def summarize_text(text: str, max_length: int = 150, min_length: int = 40) -> Optional[str]:
        if summarizer is None:
            return None
        # chunk if needed
        if len(text) > 2000:
            text = text[:2000]
        try:
            summary = summarizer(text, max_length=max_length, min_length=min_length)[0]["summary_text"]
            return summary
        except Exception:
            return None


    def create_chunks(sections, chunk_size: int = 500):
        chunks = []
        for s in sections:
            text = s["content"]
            for i in range(0, len(text), chunk_size):
                chunks.append({"page": s["page"], "text": text[i:i + chunk_size]})
        return chunks


    app = FastAPI()

    # Use a project-local uploads folder (works on Windows and Unix).
    UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")


    @app.post("/upload")
    async def upload_pdf(file: UploadFile = File(...)):
        doc_id = str(uuid.uuid4())
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        filepath = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        with open(filepath, "wb") as f:
            f.write(await file.read())
        # Kick off background processing (extract text, OCR if needed, create embeddings)
        # For MVP you can call processing synchronously or dispatch background tasks
        return {"id": doc_id, "status": "uploaded"}


    @app.get("/document/{doc_id}")
    def get_document(doc_id: str):
        # Return basic metadata and presence of the uploaded file.
        filepath = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        if not os.path.exists(filepath):
            return {"id": doc_id, "found": False}
        stat = os.stat(filepath)
        return {
            "id": doc_id,
            "found": True,
            "size": stat.st_size,
            "path": filepath,
            # placeholders for future structured data
            "text_extracted": False,
            "tables_extracted": False,
        }
