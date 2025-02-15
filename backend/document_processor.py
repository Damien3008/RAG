"""
Document Processing Module
Handles PDF document processing, text extraction, and chunking.
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

class DocumentProcessor:
    """
    Processes PDF documents for RAG system.
    
    Attributes:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between consecutive chunks
        min_chunk_size: Minimum size for valid chunks
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = 100

    def process_docs_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            
        Returns:
            List of processed document chunks with metadata
        """
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        all_chunks = []
        current_id = 0

        print(f"\nProcessing {len(pdf_files)} PDF files...")
        
        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                file_path = os.path.join(folder_path, filename)
                chunks = self.process_pdf(file_path, current_id)
                all_chunks.extend(chunks)
                current_id = max(chunk['id'] for chunk in chunks) + 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        return all_chunks

    def process_pdf(self, file_path: str, start_chunk_id: int) -> List[Dict[str, Any]]:
        """
        Process a single PDF file.
        
        Args:
            file_path: Path to PDF file
            start_chunk_id: Starting ID for chunks
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        reader = PdfReader(file_path)
        
        text_buffer = ""
        current_page = 1
        
        for page in reader.pages:
            page_text = page.extract_text()
            if not page_text.strip():
                continue
                
            text_buffer += f" {self._clean_text(page_text)}"
            
            if len(text_buffer) > self.chunk_size * 2:
                new_chunks = self._create_chunks(
                    text_buffer,
                    current_page,
                    Path(file_path).name,
                    start_chunk_id + len(chunks)
                )
                chunks.extend(new_chunks)
                text_buffer = ""
            
            current_page += 1
        
        # Process remaining text
        if text_buffer:
            final_chunks = self._create_chunks(
                text_buffer,
                current_page - 1,
                Path(file_path).name,
                start_chunk_id + len(chunks)
            )
            chunks.extend(final_chunks)
        
        return chunks

    def _create_chunks(self, text: str, page: int, source: str, start_id: int) -> List[Dict[str, Any]]:
        """Create chunks from text with proper overlap."""
        chunks = []
        start = 0
        chunk_id = start_id
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Adjust chunk end to nearest sentence boundary
            if end < len(text):
                end = self._find_sentence_boundary(text, end)
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        'source': source,
                        'page': page,
                        'chunk_number': len(chunks) + 1
                    }
                })
                chunk_id += 1
            
            start = end - self.chunk_overlap
        
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        return ' '.join(text.split())

    @staticmethod
    def _find_sentence_boundary(text: str, position: int) -> int:
        """Find the nearest sentence boundary after position."""
        for i in range(position, min(position + 100, len(text))):
            if text[i] in '.!?':
                return i + 1
        return position 