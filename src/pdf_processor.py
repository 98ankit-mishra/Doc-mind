"""
src/pdf_processor.py
====================
Automated PDF text extraction and recursive document chunking.

Stage 1: PDF Extraction
- Page-by-page extraction using PyPDF.
- Preserves 1-based page numbers.
- Handles empty, malformed, encrypted, and scanned PDFs with clear diagnostic errors.
- Explicitly reports when scanned documents require OCR.

Stage 2: Document Chunking
- Recursive sliding-window text splitter using semantic hierarchy: paragraphs (\\n\\n),
  lines (\\n), sentences (. ), words ( ), and characters.
- Default: 1000 characters chunk size, 200 characters overlap.
- Guarantees every chunk is strictly <= chunk_size.
- Preserves semantic continuity across chunk boundaries with word-level overlap.
- Preserves page-level metadata for every chunk.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class Document:
    """
    Represents an extracted text chunk along with its provenance metadata.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_pdf_file(file_path: str) -> None:
    """
    Validates file existence, non-emptiness, and PDF magic header bytes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("The uploaded file is empty (0 bytes).")

    # Verify %PDF- header in the first 1024 bytes
    with open(file_path, "rb") as f:
        header = f.read(1024)
        if b"%PDF-" not in header:
            raise ValueError(
                "Invalid PDF format: File header is missing the standard '%PDF-' magic bytes."
            )


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Sliding-window recursive text splitter.
    Splits text hierarchically using semantic boundaries:
    paragraphs -> lines -> sentences -> words -> characters.

    Guarantees:
    - Every chunk length is strictly <= chunk_size.
    - Consecutive chunks retain semantic context with word-boundary overlap.
    - Chunks are trimmed and non-empty.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Look for preferred natural boundary in the overlap window
            search_start = max(start, end - chunk_overlap)
            window = text[search_start:end]

            split_idx = -1
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = window.rfind(sep)
                if idx != -1:
                    split_idx = search_start + idx + len(sep)
                    break

            if split_idx != -1 and split_idx > start:
                end = split_idx

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Calculate next starting position with overlap
        next_start = max(start + 1, end - chunk_overlap)

        # Advance next_start to next word boundary if in the middle of a word
        if (
            next_start < text_len
            and text[next_start] not in (" ", "\n")
            and next_start > 0
            and text[next_start - 1] not in (" ", "\n")
        ):
            next_space = text.find(" ", next_start)
            if next_space != -1 and next_space < end:
                next_start = next_space + 1

        start = next_start

    return chunks


def process_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Extracts text from a PDF page-by-page and splits it into chunks with metadata.

    Args:
        file_path: Local path to the PDF file.
        chunk_size: Maximum characters per chunk (default: 1000).
        chunk_overlap: Characters of overlap between chunks (default: 200).

    Returns:
        List of Document objects containing page_content and metadata:
        - page: 1-based page number
        - source: sanitized filename
        - chunk_index: sequential chunk index within the document
        - total_pages: total pages in the PDF

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty, corrupted, password-protected, or scanned (OCR needed).
    """
    validate_pdf_file(file_path)

    try:
        reader = PdfReader(file_path)
    except PdfReadError as e:
        raise ValueError(f"Corrupted or unreadable PDF: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to open PDF file: {str(e)}")

    if reader.is_encrypted:
        try:
            # Some PDFs have an empty password
            decrypted = reader.decrypt("")
            if decrypted == 0:
                raise ValueError("PDF is password-protected and cannot be read without credentials.")
        except Exception:
            raise ValueError("PDF is password-protected and cannot be read without credentials.")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("The PDF contains no pages.")

    all_chunks: List[Document] = []
    source_filename = os.path.basename(file_path)
    pages_with_text = 0

    chunk_global_idx = 0
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()
        if not text:
            continue

        pages_with_text += 1
        page_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_str in page_chunks:
            all_chunks.append(
                Document(
                    page_content=chunk_str,
                    metadata={
                        "page": page_num,
                        "source": source_filename,
                        "chunk_index": chunk_global_idx,
                        "total_pages": total_pages
                    }
                )
            )
            chunk_global_idx += 1

    if not all_chunks or pages_with_text == 0:
        raise ValueError(
            "No extractable text found in the PDF. The document appears to be scanned or image-based, "
            "which requires OCR (Optical Character Recognition). OCR is not currently supported."
        )

    return all_chunks
