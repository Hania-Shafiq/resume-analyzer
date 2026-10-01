"""Unit tests for the resume parser module (app/parser.py)."""

import io
import sys
import unittest
from pathlib import Path

# Add project root to sys.path so app can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parser import extract_text


class TestParser(unittest.TestCase):
    """Test suite for extract_text functionality."""

    def test_unsupported_file_extension(self):
        """Ensure an unsupported extension raises a ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_text(b"some content", filename="resume.txt")
        self.assertIn("Unsupported file format", str(ctx.exception))

    def test_empty_bytes_input(self):
        """Ensure an empty byte stream raises a ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_text(b"", filename="resume.pdf")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_docx_extraction_with_table(self):
        """Ensure DOCX paragraphs and tables are extracted accurately."""
        import docx

        doc = docx.Document()
        doc.add_heading("Hania Shafiq - Resume", level=1)
        doc.add_paragraph("Experienced Software Engineer specializing in Python and NLP.")

        # Add a table to test table text extraction
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Skill"
        table.cell(0, 1).text = "Proficiency"
        table.cell(1, 0).text = "Python"
        table.cell(1, 1).text = "Advanced"

        # Save to byte stream
        stream = io.BytesIO()
        doc.save(stream)
        docx_bytes = stream.getvalue()

        # Extract text
        extracted = extract_text(docx_bytes, filename="sample.docx")

        self.assertIn("Hania Shafiq - Resume", extracted)
        self.assertIn("Experienced Software Engineer", extracted)
        self.assertIn("Python | Advanced", extracted)

    def test_pdf_extraction(self):
        """Ensure PDF text extraction works correctly."""
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        # Create an in-memory PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Jane Doe - Data Scientist")
        page.insert_text((50, 80), "Machine Learning, FastAPI, PyTorch")
        pdf_bytes = doc.write()
        doc.close()

        # Extract text
        extracted = extract_text(pdf_bytes, filename="sample.pdf")

        self.assertIn("Jane Doe - Data Scientist", extracted)
        self.assertIn("Machine Learning, FastAPI, PyTorch", extracted)


def main():
    """Run tests or parse a user-supplied resume file if provided."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sample_path = Path(sys.argv[1])
        print(f"--- Parsing Sample Resume: {sample_path} ---")
        try:
            text = extract_text(sample_path)
            print("\n[Extracted Text Output]:\n")
            print(text)
            print(f"\n--- Success ({len(text)} characters extracted) ---")
        except Exception as e:
            print(f"Error parsing {sample_path}: {e}")
    else:
        unittest.main()


if __name__ == "__main__":
    main()
