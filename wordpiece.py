from collections import Counter
from functools import cache
from tokenizer import Tokenizer 
from typing import List
import re

class WordPiece(Tokenizer):
    def __init__(self, vocab_size):
        super().__init__()
        self.desired_size = vocab_size
        self.char_to_token = {
            '[UNK]': 0,
            '[PAD]': 1,
            '[SEP]': 2,
            '[CLS]': 3,
            '[MASK]': 4
        }

        self.special_tokens = r'\[UNK\]|\[PAD\]|\[SEP\]|\[CLS\]|\[MASK\]'

    def __format_word(self, s: str) -> List[str]:
        return [c if i == 0 else f"##{c}" for i, c in enumerate(s)]

    def __numer_word(self, s: List[str]) -> List[int]:
        return [self.char_to_token[c] for c in s]
    
    def __get_heur(self, words: List[List[int]]):
        freq = Counter() 
        for word in words:
            freq.update(word)
        freq_pair = Counter()
        for word in words:
            for a, b in zip(word, word[1:]):
                freq_pair[(a, b)] += 1
        for a, b in freq_pair.keys():
            freq_pair[(a, b)] /= (freq[a] * freq[b])
        return freq_pair

    def train(self, filename: str) -> None:
        with open(filename, 'r') as f:
            data = f.read()
        data = " ".join(data.split())
        data = re.sub(r'([^\w\s])', r' \1', data)
        words = list(map(self.__format_word, data.split()))
        for word in words:
            for c in word:
                if c not in self.char_to_token:
                    ind = len(self.char_to_token)
                    self.char_to_token[c] = ind 
                    self.token_table[ind] = c 

        words = list(map(self.__numer_word, words))
        vocab_size = len(self.char_to_token) 
        while vocab_size < self.desired_size: 
            heur = self.__get_heur(words)
            (a, b), _ = heur.most_common(1)
            self.merges.append((a, b, vocab_size))
            self.token_table[vocab_size] = self.token_table[a] + self.token_table[b]
            new_words = []
            for word in words:
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i] == b:
                        new_word.append(vocab_size)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words 
            vocab_size += 1

    def encode(self, text: str) -> List[int]:
        pass  

    def decode(self, tokens: List[int]) -> str: 
        pass 