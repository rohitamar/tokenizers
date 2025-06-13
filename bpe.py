from tokenizer import Tokenizer 
from typing import List

class BPETokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def encode(self, text: str) -> List[int]:
        pass 
        
    def decode(self, tokens: List[int]) -> str: 
        pass 
