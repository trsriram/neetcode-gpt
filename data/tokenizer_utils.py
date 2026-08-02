from typing import List, Dict

class Solution:

    def tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        N = len(text)
        start = 0
        tokens = []
        max_len = max((len(k) for k in vocab), default=1)
        while start < N:
            end = min(N, start + max_len)
            while end > start:
                cur_string = text[start:end]
                if (end - start == 1) or (cur_string in vocab):
                    tokens.append(cur_string)
                    break
                end -= 1
            start = end
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self.tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        word_count = len(text.split(' '))
        token_count = self.count_tokens(text, vocab)
        return round(token_count/word_count, 4)
