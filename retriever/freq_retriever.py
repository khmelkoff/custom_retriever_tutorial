from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from gensim import corpora
from gensim.parsing import strip_tags, strip_numeric, \
    strip_multiple_whitespaces, stem_text, strip_punctuation, \
    remove_stopwords, preprocess_string, strip_non_alphanum
from gensim import models
from gensim import similarities
import pymorphy3

from retriever.stop_words import STOP_WORDS

morph_analyzer = pymorphy3.MorphAnalyzer()

GENSIM_PATH = './gensim'

try:
    with open(os.path.join(GENSIM_PATH, 'dictionary.pkl'), 'rb') as h:
        dictionary = pickle.load(h)
    with open(os.path.join(GENSIM_PATH, 'similarity_index.pkl'), 'rb') as h:
        similarity_index = pickle.load(h)
    model_ = models.TfidfModel()
    retriever_model = model_.load(os.path.join(GENSIM_PATH, 'tfidf_model.mo'))
except FileNotFoundError:
    raise FileNotFoundError("Could not load gensim model files, please check gensim folder")


# Filters to be executed in pipeline
transform_to_lower = lambda s: s.lower()
CLEAN_FILTERS = [
                strip_tags,
                # strip_numeric,
                strip_punctuation,
                strip_non_alphanum,
                strip_multiple_whitespaces,
                transform_to_lower,
                ]

# Method does the filtering of all the unrelevant text elements
def cleaning_pipe(text:str) -> list[str]:
    # Invoking gensim.parsing.preprocess_string method with set of filters
    processed_words = preprocess_string(text, CLEAN_FILTERS)
    processed_words = [s for s in processed_words if len(s) > 1]
    processed_words = [s for s in processed_words if s not in STOP_WORDS]
    processed_words = [morph_analyzer.parse(s)[0].normal_form for s in processed_words]
    return processed_words


class FreqRetriever(BaseRetriever):
    """TF-IDF retriever based on gensim model"""

    docs: List[Document]
    """Documents."""
    k: int = 4
    """Number of documents to return."""
    with_similarity: bool = False
    """True for return found chunk relevance"""

    @classmethod
    def from_documents(
        cls,
        docs: Iterable[Document],
        **kwargs: Any,
    ) -> FreqRetriever:
        """
        Create a FreqRetriever instance from a list of langchain Documents.
        Args:
            docs: A list of of langchain Documents.
            **kwargs: Any other arguments to pass to the retriever.
        Returns:
            A FreqRetriever instance.
        """

        return cls(
            docs=docs,
            **kwargs
        )

    def get_top_n(self, query: str, n: int, with_similarity: bool) -> List[Tuple[Document, float]] | List[Document] | None:
        """Retriever query engine
        Args:
            query: text query.
            n: number of returned documents.
        Returns:
            A list of tuples with relevance score.
        """

        query_bow = dictionary.doc2bow(query)
        sims = similarity_index[retriever_model[query_bow]]
        qty = sum(sims > 0)

        if qty > 0:
            top_idx = sims.argsort()[-1 * n:][::-1]
            result = []
            for idx in top_idx:
                relevance = round(float(sims[idx]), 3)
                doc = self.docs[idx]
                if with_similarity:
                    result.append((doc, relevance))
                else:
                    result.append(doc)
            return result
        else:
            return None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Tuple[Document, float]] | None:
        processed_query = cleaning_pipe(query)
        return_docs = self.get_top_n(processed_query, n=self.k, with_similarity=self.with_similarity)
        return return_docs
