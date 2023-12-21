from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
class SentenceTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for SentenceTransformer to be use it in sklearn pipeline
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X, show_progress_bar=False)
    