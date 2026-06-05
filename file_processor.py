import os
import fitz
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1200, overlap=200):
    """Splits text using a smarter, recursive method."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""] # Prioritizes splitting on paragraphs and sentences
    )
    return text_splitter.split_text(text)

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_epub(file_path):
    """Extracts text from an EPUB file."""
    book = epub.read_epub(file_path)
    text = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        text += soup.get_text() + "\n\n"
    return text

def extract_text_from_txt(file_path):
    """Extracts text from a plain text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_file(file_path):
    """A helper to call the correct extraction function based on file extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at the specified path: {file_path}")
    
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.epub'):
        return extract_text_from_epub(file_path)
    elif file_path.lower().endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type. Please use PDF, EPUB, or TXT.")