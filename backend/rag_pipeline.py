"""
RAG (Retrieval Augmented Generation) Pipeline
This module implements a RAG system using Google's Gemini AI and FAISS for document retrieval.
"""

import re
import random
from typing import List, Dict, Any

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

from ai_analyzer import AIAnalyzer

# Load and validate environment
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)

class RAGPipeline:
    """
    Implements RAG pipeline for document querying and analysis.
    
    Attributes:
        model: Gemini AI model for text generation
        embeddings: Vector embeddings for document similarity
        vector_store: FAISS index for document storage and retrieval
        ai_analyzer: Component for additional AI analysis
        documents: List of processed documents
    """

    def __init__(self, 
                 model_name: str = 'gemini-pro',
                 embedding_model: str = "models/embedding-001",
                 relevance_threshold: float = 0.3):
        """Initialize RAG pipeline with specified models and parameters."""
        self.model = genai.GenerativeModel(model_name)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model=embedding_model
        )
        self.vector_store = None
        self.ai_analyzer = AIAnalyzer()
        self.documents = []
        self.relevance_threshold = relevance_threshold

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add new documents to the vector store and update document collection.
        
        Args:
            documents: List of document dictionaries containing text and metadata
        """
        texts = [doc['text'] for doc in documents]
        metadatas = [{
            'source': doc['metadata']['source'],
            'page': doc['metadata'].get('page', 'Unknown'),
            'chunk_id': doc['id']
        } for doc in documents]
        
        # Initialize or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.vector_store.add_texts(texts, metadatas=metadatas)

        # Update document collection and analyze
        self.documents.extend(documents)
        self._analyze_new_documents(documents)

    def query(self, question: str, k: int = 8) -> Dict[str, Any]:
        """
        Process a question and generate an answer using RAG.
        
        Args:
            question: User's question
            k: Number of relevant documents to retrieve

        Returns:
            Dictionary containing answer, citations, and analysis
        """
        if not self.vector_store:
            raise ValueError("No documents have been added to the pipeline yet")

        # Retrieve and filter relevant documents
        relevant_docs = self._retrieve_relevant_documents(question, k)
        
        # Prepare context and generate answer
        context = self._prepare_context(relevant_docs)
        prompt = self._create_prompt(question, context)
        response = self.model.generate_content(prompt)
        
        # Process and format response
        processed_response = self._post_process_response(response.text)
        
        return {
            'answer': processed_response,
            'citations': self._create_citations(relevant_docs, question),
            'analysis': self._analyze_response(processed_response)
        }

    def _retrieve_relevant_documents(self, question: str, k: int) -> List[Any]:
        """Retrieve and filter relevant documents based on similarity."""
        enhanced_question = self._enhance_question(question)
        docs_with_scores = self.vector_store.similarity_search_with_score(enhanced_question, k=k)
        return [doc for doc, score in docs_with_scores if score > self.relevance_threshold]

    def _prepare_context(self, docs: List[Any]) -> str:
        """Format document contents into structured context."""
        context_parts = []
        for doc in docs:
            context_parts.append(
                f"[Extract from {doc.metadata['source']}, Page {doc.metadata['page']}]\n"
                f"{doc.page_content}\n"
            )
        return "\n\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create a structured prompt for the AI model."""
        return f"""You are a precise and knowledgeable assistant. Answer the following question based on the provided extracts.

Question: {question}

Relevant Document Extracts:
{context}

Instructions:
1. Answer only based on the provided extracts
2. If information is not available, say so
3. Cite sources and page numbers
4. Use bullet points for multiple points
5. Highlight key findings or numbers
6. Use clear, professional language

Answer:"""

    def _post_process_response(self, text: str) -> str:
        """Clean and format the model's response."""
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation brackets
        text = re.sub(r'(?m)^[-*•] ', '• ', text)  # Standardize bullet points
        text = re.sub(r'(\d+(?:\.\d+)?(?:\s*%)?)', r'**\1**', text)  # Highlight numbers
        return text.strip()

    def _create_citations(self, docs: List[Any], question: str) -> List[Dict[str, Any]]:
        """
        Create citations from relevant documents.
        
        Args:
            docs: List of relevant documents
            question: Original question for context matching
            
        Returns:
            List of citations with metadata and relevant text
        """
        citations = []
        for i, doc in enumerate(docs, 1):
            # Extract relevant snippet around matching terms
            text = doc.page_content
            max_snippet_length = 300
            
            if len(text) > max_snippet_length:
                # Find relevant section based on question terms
                question_terms = question.lower().split()
                lowest_index = len(text)
                highest_index = 0
                
                for term in question_terms:
                    if term in text.lower():
                        term_index = text.lower().find(term)
                        lowest_index = min(lowest_index, term_index)
                        highest_index = max(highest_index, term_index + len(term))
                
                # Create window around relevant terms
                start = max(0, lowest_index - 100)
                end = min(len(text), highest_index + 100)
                
                # Add ellipsis for truncated text
                text = ('...' if start > 0 else '') + \
                       text[start:end] + \
                       ('...' if end < len(text) else '')
            
            citations.append({
                'source_number': i,
                'source': doc.metadata['source'],
                'page': doc.metadata['page'],
                'text': text,
                'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
            })
        
        return citations

    def _analyze_new_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Analyze newly added documents."""
        topics = self.ai_analyzer.extract_topics(documents)
        print(f"Extracted {len(topics)} main topics from new documents")

    def _analyze_response(self, response: str) -> Dict[str, Any]:
        """Perform analysis on the generated response."""
        return {
            'sentiment': self.ai_analyzer.analyze_sentiment(response),
            'topics': self.ai_analyzer.extract_topics(self.documents)
        }

    @staticmethod
    def _enhance_question(question: str) -> str:
        """Enhance question for better document retrieval."""
        enhancements = [
            "relevant information about",
            "specific details regarding",
            "key points about",
            "facts and data about"
        ]
        return f"{random.choice(enhancements)} {question}" 