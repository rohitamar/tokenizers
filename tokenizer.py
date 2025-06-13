from abc import ABC, abstractmethod 
from typing import List

class Tokenizer(ABC):
    def __init__(self):
        pass 

    @abstractmethod    
    def train(self, ) -> None: 
        pass 

    @abstractmethod 
    def encode(self, text) -> List[int]:
        pass 

    @abstractmethod
    def decode(self, tokens: List[int]) -> str: 
        pass 
    
    def load(self, path: str) -> None:
        pass 

    def save(self, path: str) -> None:
        pass 