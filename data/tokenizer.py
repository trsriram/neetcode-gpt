from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break
            pair_counts = Counter(zip(tokens, tokens[1:]))
            max_freq = max(pair_counts.values())
            best_pair = min(
                pair for pair, count in pair_counts.items() if count == max_freq
            )
            
            new_tokens = []
            i = 0
            while (i + 1 < len(tokens)):
                if tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i = i + 2
                else:
                    new_tokens.append(tokens[i])
                    i = i + 1
            tokens = new_tokens
            merges.append(list(best_pair))
        return merges
                
                



        





            