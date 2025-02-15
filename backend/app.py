"""
Flask Application for Document Q&A System
Handles document uploads, processing, and question answering endpoints.
"""

import os
from pathlib import Path
from typing import Tuple

from flask import Flask, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Response

from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend')

# Initialize processors with optimized parameters
doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
rag_pipeline = RAGPipeline()

# Configure upload settings
BASE_DIR = Path(__file__).parent.parent
UPLOAD_FOLDER = BASE_DIR / "docs"
app.config.update(
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
)

def init_documents() -> None:
    """Initialize system by processing existing documents."""
    try:
        UPLOAD_FOLDER.mkdir(exist_ok=True)
        
        if not any(UPLOAD_FOLDER.iterdir()):
            return
        
        print("Starting document processing...")
        chunks = doc_processor.process_docs_folder(str(UPLOAD_FOLDER))
        print(f"Processing complete! Total chunks: {len(chunks)}")
        
        rag_pipeline.add_documents(chunks)
        print("System ready for queries!")
    except Exception as e:
        print(f"Error during initialization: {e}")

@app.route('/')
def serve_frontend() -> Response:
    """Serve the frontend application."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question() -> Tuple[Response, int]:
    """
    Handle question answering requests.
    
    Returns:
        Tuple of (response, status_code)
    """
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        result = rag_pipeline.query(question)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file() -> Tuple[str, int]:
    """
    Handle file uploads and processing.
    
    Returns:
        Tuple of (message, status_code)
    """
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if not file or not file.filename:
        return 'No selected file', 400
    
    if not file.filename.lower().endswith('.pdf'):
        return 'Invalid file type', 400
    
    try:
        # Save and process file
        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file.save(str(file_path))
        
        # Process document
        chunks = doc_processor.process_pdf(str(file_path), 0)
        rag_pipeline.add_documents(chunks)
        
        return f'File processed successfully: {len(chunks)} chunks extracted', 200
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        return f'Error processing file: {str(e)}', 500

@app.route('/api/stats', methods=['GET'])
def get_stats() -> Tuple[Response, int]:
    """
    Get document analysis statistics.
    
    Returns:
        Tuple of (response, status_code)
    """
    try:
        insights = rag_pipeline.ai_analyzer.get_key_insights(rag_pipeline.documents)
        return jsonify({'insights': insights}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize system on startup
init_documents()

if __name__ == '__main__':
    app.run(port=5000) 