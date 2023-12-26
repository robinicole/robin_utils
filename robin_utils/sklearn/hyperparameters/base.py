from sklearn.base import BaseEstimator
from typing import List
class HyperparaMeter:
    name : str 
    models : List[BaseEstimator]

