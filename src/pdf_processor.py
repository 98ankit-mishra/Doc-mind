import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pypdf import PdfReader

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Pure Python recursive text splitter without LangChain.
    Splits text by double newlines, single newlines, spaces, or characters.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", " ", ""]

    def _split(text_segment: str, sep_idx: int) -> List[str]:
        if len(text_segment) <= chunk_size:
            return [text_segment] if text_segment.strip() else []
        if sep_idx >= len(separators):
            # Hard character split fallback
            step = max(1, chunk_size - chunk_overlap)
            return [text_segment[i:i + chunk_size] for i in range(0, len(text_segment), step) if text_segment[i:i + chunk_size].strip()]

        separator = separators[sep_idx]
        if separator == "":
            step = max(1, chunk_size - chunk_overlap)
            return [text_segment[i:i + chunk_size] for i in range(0, len(text_segment), step) if text_segment[i:i + chunk_size].strip()]

        splits = text_segment.split(separator)
        chunks = []
        current_chunk = ""

        for piece in splits:
            candidate = f"{current_chunk}{separator}{piece}" if current_chunk else piece
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(piece) > chunk_size:
                    # Recursive split with deeper separator
                    deeper_chunks = _split(piece, sep_idx + 1)
                    chunks.extend(deeper_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    raw_chunks = _split(text, 0)

    # Apply overlap smoothly between consecutive chunks
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        final_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and len(raw_chunks[i - 1]) >= chunk_overlap:
                overlap_prefix = raw_chunks[i - 1][-chunk_overlap:]
                final_chunks.append(f"{overlap_prefix} {chunk}")
            else:
                final_chunks.append(chunk)
        return final_chunks

    return raw_chunks

def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Loads a PDF using pypdf and splits it into chunks without LangChain.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    if len(reader.pages) == 0:
        raise ValueError("The PDF is empty or could not be parsed.")

    all_chunks: List[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        page_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in page_chunks:
            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_num, "source": file_path}
                )
            )

    if not all_chunks:
        raise ValueError("No extractable text found in the PDF.")

    return all_chunks
