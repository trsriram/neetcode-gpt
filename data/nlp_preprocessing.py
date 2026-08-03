import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        tokenized_sentences = [s.split(' ') for s in positive + negative]
        unique_words = sorted({word for tokens in tokenized_sentences for word in tokens})
        vocab = {word: i + 1 for i, word in enumerate(unique_words)}

        tensors = [
            torch.tensor([vocab[w] for w in tokens], dtype=torch.float)
            for tokens in tokenized_sentences
        ]

        print(tensors)

        return nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)

