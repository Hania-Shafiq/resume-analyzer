"""Resume parser module for extracting raw text and structure from PDF and DOCX files."""

import io
import os
import re
from pathlib import Path
from typing import Union, BinaryIO

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF fallback
import docx


def extract_text(file_path_or_bytes: Union[str, Path, bytes, bytearray, BinaryIO], filename: str = "") -> str:
    """Extract and clean text from a PDF or DOCX file.

    Parameters
    ----------
    file_path_or_bytes : str, Path, bytes, bytearray, or BinaryIO
        File system path to the document, or raw bytes / byte-like stream.
    filename : str, optional
        Original filename (e.g. 'resume.pdf'). Required if passing raw bytes
        or if the path does not contain a recognizable file extension.

    Returns
    -------
    str
        Cleaned, extracted plain text content.

    Raises
    ------
    FileNotFoundError
        If a file path was provided but does not exist.
    ValueError
        If the file format is unsupported, the input is empty,
        or no text could be extracted.
    """
    # Step 1: Identify filename and file extension
    target_name = filename
    if not target_name and isinstance(file_path_or_bytes, (str, Path)):
        target_name = str(file_path_or_bytes)

    if not target_name:
        raise ValueError("Filename with extension must be provided when passing raw bytes.")

    ext = Path(target_name).suffix.lower()

    # Step 2: Validate supported extensions
    if ext not in [".pdf", ".docx"]:
        raise ValueError(
            f"Unsupported file format '{ext}'. Only .pdf and .docx files are supported."
        )

    # Step 3: Check if file path exists or read bytes
    raw_bytes: bytes = b""
    is_path = isinstance(file_path_or_bytes, (str, Path))

    if is_path:
        path = Path(file_path_or_bytes)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError("The provided file is empty (0 bytes).")
        raw_bytes = path.read_bytes()
    elif isinstance(file_path_or_bytes, (bytes, bytearray)):
        raw_bytes = bytes(file_path_or_bytes)
        if len(raw_bytes) == 0:
            raise ValueError("The provided byte stream is empty (0 bytes).")
    elif hasattr(file_path_or_bytes, "read"):
        raw_bytes = file_path_or_bytes.read()
        if len(raw_bytes) == 0:
            raise ValueError("The provided file-like object is empty (0 bytes).")
    else:
        raise ValueError("Unsupported input type for file_path_or_bytes.")

    extracted_chunks = []

    # Step 4: Extract text based on file format
    if ext == ".pdf":
        # Open PDF from byte stream using PyMuPDF (fitz)
        try:
            with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
                if doc.page_count == 0:
                    raise ValueError("The PDF document contains no pages.")
                # Read text page by page
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text")
                    if page_text:
                        extracted_chunks.append(page_text)
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse PDF document: {e}") from e

    elif ext == ".docx":
        # Open DOCX from byte stream using python-docx
        try:
            doc = docx.Document(io.BytesIO(raw_bytes))

            # 4a. Extract regular paragraph text
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    extracted_chunks.append(text)

            # 4b. Extract table text (rows and cells)
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    # Deduplicate repeated adjacent cells (common in merged cells)
                    deduped_cells = []
                    for cell_text in row_cells:
                        if not deduped_cells or cell_text != deduped_cells[-1]:
                            deduped_cells.append(cell_text)
                    if deduped_cells:
                        extracted_chunks.append(" | ".join(deduped_cells))
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse DOCX document: {e}") from e

    # Step 5: Clean and normalize the extracted text
    combined_text = "\n".join(extracted_chunks)

    # Normalize horizontal whitespace (tabs, multiple spaces)
    cleaned_lines = []
    for line in combined_text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    result_text = "\n".join(cleaned_lines).strip()

    # Step 6: Verify extracted text is not empty
    if not result_text:
        raise ValueError("The document contains no extractable text.")

    return result_text
