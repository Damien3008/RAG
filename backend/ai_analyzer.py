"""
AI Analysis Module
Provides advanced text analysis features including sentiment analysis, 
topic modeling, and key insight extraction.
"""

import warnings
from typing import List, Dict, Any
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize

# Configure warnings and NLTK
warnings.filterwarnings('ignore', category=ConvergenceWarning)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class AIAnalyzer:
    """
    Provides AI-powered text analysis capabilities.
    
    Attributes:
        vectorizer: TF-IDF vectorizer for text processing
        topic_model: NMF model for topic extraction
        max_topics: Maximum number of topics to extract
        sentiment_threshold: Threshold for sentiment classification
    """

    def __init__(self, 
                 max_topics: int = 5,
                 sentiment_threshold: float = 0.1,
                 max_features: int = 1000):
        """
        Initialize the analyzer with customizable parameters.
        
        Args:
            max_topics: Maximum number of topics to extract
            sentiment_threshold: Threshold for sentiment classification
            max_features: Maximum number of features for TF-IDF
        """
        self.max_topics = max_topics
        self.sentiment_threshold = sentiment_threshold
        
        # Initialize text processing models
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        
        self.topic_model = NMF(
            n_components=max_topics,
            random_state=42,
            max_iter=500,
            tol=1e-4,
            init='nndsvd'
        )

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        # Determine sentiment category
        if sentiment > self.sentiment_threshold:
            category = "Positive"
        elif sentiment < -self.sentiment_threshold:
            category = "Negative"
        else:
            category = "Neutral"
        
        # Analyze significant sentences
        sentences = sent_tokenize(text)
        significant_sentiments = []
        
        for sentence in sentences:
            sent_blob = TextBlob(sentence)
            sent_sentiment = sent_blob.sentiment.polarity
            
            if abs(sent_sentiment) > self.sentiment_threshold * 2:
                significant_sentiments.append({
                    'text': sentence,
                    'sentiment': sent_sentiment,
                    'category': "Positive" if sent_sentiment > 0 else "Negative"
                })
        
        return {
            'overall_sentiment': sentiment,
            'category': category,
            'significant_sentences': significant_sentiments
        }

    def extract_topics(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract main topics from documents using NMF.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of topics with relevant words and documents
        """
        if not documents:
            return []
            
        # Prepare document texts
        texts = [doc['text'] for doc in documents]
        
        # Create and fit document-term matrix
        dtm = self.vectorizer.fit_transform(texts)
        topic_matrix = self.topic_model.fit_transform(dtm)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(self.topic_model.components_):
            # Get top words for topic
            top_words_idx = topic.argsort()[:-10:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            # Find relevant documents
            doc_scores = topic_matrix[:, topic_idx]
            top_doc_indices = doc_scores.argsort()[:-3:-1]
            
            relevant_docs = [
                {
                    'source': documents[idx]['metadata']['source'],
                    'page': documents[idx]['metadata']['page'],
                    'score': float(doc_scores[idx])
                }
                for idx in top_doc_indices
                if doc_scores[idx] > 0.1
            ]
            
            topics.append({
                'id': topic_idx + 1,
                'words': top_words,
                'relevant_documents': relevant_docs,
                'coherence': self._calculate_topic_coherence(top_words)
            })
        
        return sorted(topics, key=lambda x: len(x['relevant_documents']), reverse=True)

    def get_key_insights(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract key insights from documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of insights with types and messages
        """
        if not documents:
            return []
            
        insights = []
        
        # Document complexity analysis
        avg_words = np.mean([len(doc['text'].split()) for doc in documents])
        insights.append({
            'type': 'complexity',
            'message': f"Average section length: {int(avg_words)} words"
        })
        
        # Key terms analysis
        all_text = ' '.join(doc['text'] for doc in documents)
        blob = TextBlob(all_text)
        noun_phrases = blob.noun_phrases
        
        if noun_phrases:
            top_phrases = Counter(noun_phrases).most_common(3)
            insights.append({
                'type': 'key_terms',
                'message': f"Top key terms: {', '.join(phrase for phrase, _ in top_phrases)}"
            })
        
        # Topic analysis
        topics = self.extract_topics(documents)
        avg_coherence = np.mean([topic['coherence'] for topic in topics])
        
        insights.append({
            'type': 'topics',
            'message': (f"Document collection covers {len(topics)} main topics "
                       f"with {avg_coherence:.2f} average coherence")
        })
        
        return insights

    def _calculate_topic_coherence(self, words: List[str]) -> float:
        """Calculate semantic coherence score for a topic."""
        try:
            pairs = list(combinations(words, 2))
            scores = [self._word_similarity(w1, w2) for w1, w2 in pairs]
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def _word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words using their vectors."""
        try:
            vec1 = self.vectorizer.transform([word1]).toarray()[0]
            vec2 = self.vectorizer.transform([word2]).toarray()[0]
            return float(np.dot(vec1, vec2))
        except Exception:
            return 0.0 