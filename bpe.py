from collections import Counter, defaultdict
from functools import cache
from tokenizer import Tokenizer 
from typing import List
import regex  

class BPETokenizer(Tokenizer):
    def __init__(self, vocab_size):
        super().__init__()
        # following https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L53C20-L53C113
        self.pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.desired_size = vocab_size    
        self.merges = []
        self.cache_word = {}
        self.token_table = {}
        for i in range(256):
            self.token_table[i] = chr(i)

    def __word_to_token(self, s):
        return list(map(int, s.encode("utf-8")))

    def __get_freq(self, tokens_per_word: List[List[int]]):
        freq = Counter()
        for tokens in tokens_per_word:
            for a, b in zip(tokens, tokens[1:]):
                freq[(a, b)] += 1    
        return freq

    def train(self, filename: str) -> None:
        f = open(filename, 'r')
        data = f.read()
        f.close() 
        words = regex.findall(self.pattern, data)
        tokens_per_word = list(map(self.__word_to_token, words))
        vocab_size = 256
        while vocab_size < self.desired_size:
            freq = self.__get_freq(tokens_per_word)
            (a, b), _ = freq.most_common(1)
            self.merges.append((a, b, vocab_size))
            self.token_table[vocab_size] = self.token_table[a] + self.token_table[b]
            new_tokens_per_word = []
            for tokens in tokens_per_word:
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(vocab_size)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_tokens_per_word.append(new_tokens)
            tokens_per_word = new_tokens_per_word
            vocab_size += 1

    def __encode_word(self, word: str) -> List[int]:
        if word in self.cache_word:
            return self.cache_word[word]
        tokens = self.__word_to_token(word)
        for a, b, c in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(c)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        self.cache_word[word] = tokens 
        return tokens 

    def encode(self, text: str) -> List[int]:
        words = regex.findall(self.pattern, text)
        result = []
        for word in words:
            result.extend(self.__encode_word(word))
        return result 

    def decode(self, tokens: List[int]) -> str: 
        chars = []
        for token in tokens:
            chars.append(self.token_table[token])
        return "".join(chars)
