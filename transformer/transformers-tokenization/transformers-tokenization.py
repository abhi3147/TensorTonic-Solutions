import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        sentences = [s.lower() for s in texts]
        words = []

        for s in sentences:
            words.extend(s.split())
        unique = sorted(list(set(words)))
        
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        for i,w in enumerate(unique,start = len(self.word_to_id)):
            self.word_to_id[w] = i
            self.id_to_word[i] = w
            
        self.vocab_size = len(self.word_to_id)
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        tokens = []
        for w in words:
            if w in self.word_to_id:
                tokens.append(self.word_to_id[w])
            else:
                tokens.append(self.word_to_id[self.unk_token])

        return tokens
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        sentence = []
        for i in ids:
            if i in self.id_to_word:
                sentence.append(self.id_to_word[i])
            else:
                sentence.append(self.unk_token)

        final_sentence = " ".join(sentence)

        return final_sentence
