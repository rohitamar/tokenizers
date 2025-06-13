from tokenizer import Tokenizer
from typing import List 

class WhitespaceTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = {'<unk>': 0}
        self.rev_vocab = {0: '<unk>'}

    def train(self, filename: str):
        # punctuation stripping
        # unicode issues (removing accents)
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                for word in line.split():
                    if word not in self.vocab: 
                        word_id = len(self.vocab)
                        self.vocab[word] = word_id
                        self.rev_vocab[word_id] = word

    def encode(self, text) -> List[int]:
        text = text.strip().split()
        unk_id = self.vocab['<unk>']
        return [
            self.vocab.get(word, unk_id)
            for word in text 
        ]

    def decode(self, tokens: List[int]) -> str:
        words = [self.rev_vocab[tok] for tok in tokens]
        return " ".join(words)
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            for word, word_id in self.vocab.items():
                f.write(f"{word}\t{word_id}\n")
    
    def load(self, path: str) -> None:
        with open(path, 'r') as f:
            for line in f: 
                word, word_id = line.split("\t")
                word_id = int(word_id)
                self.vocab[word] = word_id 
                self.rev_vocab[word_id] = word 
        
class CharacterTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    
