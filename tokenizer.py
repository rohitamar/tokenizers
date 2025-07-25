from abc import ABC, abstractmethod 
from typing import List

class Tokenizer(ABC):
    def __init__(self):
        pass 

    @abstractmethod    
    def train(self, filename: str) -> None: 
        pass 

    @abstractmethod 
    def encode(self, text: str) -> List[int]:
        pass 

    @abstractmethod
    def decode(self, tokens: List[int]) -> str: 
        pass 
    
    def load(self, path: str) -> None:
        pass 

    def save(self, path: str) -> None:
        pass 