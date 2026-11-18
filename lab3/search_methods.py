"""ЛР3: лексический поиск BM25 и объединение двух ранжированных списков."""

import math
import re
from collections import Counter, defaultdict


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.lengths = {}
        for doc_id, text in documents.items():
            words = tokens(text)
            self.lengths[doc_id] = len(words)
            for word, count in Counter(words).items():
                self.postings[word].append((doc_id, count))
        n = len(self.lengths)
        self.average_length = sum(self.lengths.values()) / n
        self.idf = {
            word: math.log1p((n - len(posting) + 0.5) / (len(posting) + 0.5))
            for word, posting in self.postings.items()
        }

    def search(self, claim, depth=20):
        scores = defaultdict(float)
        # Один вклад на уникальное слово запроса. Порядок суммы воспроизводим.
        for word in dict.fromkeys(tokens(claim)):
            for doc_id, count in self.postings.get(word, ()):
                length_ratio = self.lengths[doc_id] / self.average_length
                norm = self.k1 * (1 - self.b + self.b * length_ratio)
                scores[doc_id] += self.idf[word] * count * (self.k1 + 1) / (count + norm)
        return ranked(scores, depth)


def ranked(scores, depth):
    order = sorted(scores, key=lambda doc_id: (-scores[doc_id], int(doc_id)))
    return [
        {"doc_id": doc_id, "score": float(scores[doc_id]), "rank": rank}
        for rank, doc_id in enumerate(order[:depth], 1)
    ]


def rrf(dense, bm25, depth=20, constant=60):
    contributions = defaultdict(dict)
    for method, items in (("dense", dense), ("bm25", bm25)):
        for rank, item in enumerate(items[:depth], 1):
            contributions[item["doc_id"]][method] = {"rank": rank, "score": 1 / (constant + rank)}
    scores = {doc: sum(v["score"] for v in methods.values()) for doc, methods in contributions.items()}
    result = ranked(scores, depth)
    for item in result:
        methods = contributions[item["doc_id"]]
        item["contributions"] = {
            method: methods.get(method, {"rank": None, "score": 0.0}) for method in ("dense", "bm25")
        }
    return result
