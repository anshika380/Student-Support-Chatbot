"""
chatbot_engine.py
------------------
Core NLP engine for the Student Support Chatbot.

Approach:
1. Load a knowledge base of (question, answer, category) triples.
2. Encode every knowledge-base question into a dense vector using a
   Sentence-BERT model (all-MiniLM-L6-v2) -> semantic embeddings.
3. When a user asks something, encode their query the same way and run
   cosine similarity search against the knowledge base to find the most
   semantically similar FAQ (works even if the wording is different from
   the stored question - that's the whole point of embeddings over plain
   keyword matching).
4. If confidence is high -> return the matched answer.
   If confidence is low -> return a graceful fallback + top suggestions.

This file is intentionally self-contained so it's easy to explain in a
viva: no black-box RAG framework, just embeddings + cosine similarity,
which is the same core idea used in the Resume Screening project.
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional

KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.json")
EMBEDDINGS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "kb_embeddings.npy")

# Confidence thresholds (cosine similarity, range -1 to 1, realistically 0 to 1 for SBERT)
HIGH_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.35


class StudentSupportChatbot:
    def __init__(self):
        self.kb: List[Dict] = self._load_knowledge_base()
        self.model = None
        self.kb_embeddings = None
        self.backend = None  # "sbert" or "tfidf"
        self._init_engine()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _load_knowledge_base(self) -> List[Dict]:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _init_engine(self):
        """
        Try to load Sentence-BERT (preferred, true semantic embeddings).
        If the model can't be downloaded (e.g. no internet on first run),
        fall back to a TF-IDF vector space model so the app still works
        end-to-end. This fallback also makes local development/testing
        possible without a network call every time.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "sbert"
            self.kb_embeddings = self._encode([item["question"] for item in self.kb])
        except Exception as e:
            print(f"[chatbot_engine] Falling back to TF-IDF (SBERT unavailable: {e})")
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(stop_words="english")
            questions = [item["question"] for item in self.kb]
            self.kb_embeddings = self.vectorizer.fit_transform(questions).toarray()
            self.backend = "tfidf"

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sbert" or self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return self.vectorizer.transform(texts).toarray()

    # ------------------------------------------------------------------ #
    # Core similarity search
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
        return a_norm @ b_norm.T

    def _top_k_matches(self, query: str, k: int = 3):
        if self.backend == "sbert":
            query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        else:
            query_vec = self.vectorizer.transform([query]).toarray()

        sims = self._cosine_sim(query_vec, self.kb_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self.kb[i], float(sims[i])) for i in top_idx]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_response(self, query: str) -> Dict:
        query = query.strip()
        if not query:
            return {
                "response": "Could you please type your question? I'm here to help with admissions, exams, fees, hostel, placements, and more.",
                "confidence": 0.0,
                "category": None,
                "matched_question": None,
                "suggestions": self._sample_questions(),
            }

        matches = self._top_k_matches(query, k=3)
        best_item, best_score = matches[0]

        if best_score >= HIGH_CONFIDENCE_THRESHOLD:
            return {
                "response": best_item["answer"],
                "confidence": round(best_score, 3),
                "category": best_item["category"],
                "matched_question": best_item["question"],
                "suggestions": [],
            }
        elif best_score >= LOW_CONFIDENCE_THRESHOLD:
            # Medium confidence: answer, but flag uncertainty and offer alternatives
            return {
                "response": (
                    f"I think this might answer your question:\n\n{best_item['answer']}\n\n"
                    "If this isn't quite what you meant, try one of the related questions below."
                ),
                "confidence": round(best_score, 3),
                "category": best_item["category"],
                "matched_question": best_item["question"],
                "suggestions": [m[0]["question"] for m in matches[1:]],
            }
        else:
            return {
                "response": (
                    "I'm not fully sure about that one. I can help with Admissions, Examinations, "
                    "Fees & Scholarships, Hostel, Library, Placements, Technical/IT support, Attendance, "
                    "and Grievances. You could also try rephrasing, or reach out to the relevant department office directly."
                ),
                "confidence": round(best_score, 3),
                "category": None,
                "matched_question": None,
                "suggestions": [m[0]["question"] for m in matches],
            }

    def get_categories(self) -> List[str]:
        return sorted(set(item["category"] for item in self.kb))

    def get_faqs_by_category(self, category: str) -> List[Dict]:
        return [item for item in self.kb if item["category"].lower() == category.lower()]

    def _sample_questions(self, n: int = 4) -> List[str]:
        import random

        return [item["question"] for item in random.sample(self.kb, min(n, len(self.kb)))]


# Singleton instance used by the FastAPI app
chatbot = StudentSupportChatbot()
