"""Обязательные расчётные проверки методички без SciFact и моделей."""

import math
import unittest

from search_methods import BM25, rrf, tokens


class SearchChecks(unittest.TestCase):
    def test_bm25_manual(self):
        # Длины 3, 2 и 1, средняя 2. Термин встречается в 2 из 3 документов.
        index = BM25({"1": "aldh1 aldh1 marker", "2": "aldh1 gene", "3": "protein"})
        idf = math.log(1.6)
        expected = [idf * 4.4 / 3.65, idf]
        result = index.search("ALDH1 aldh1")
        self.assertEqual([row["doc_id"] for row in result], ["1", "2"])
        for row, score in zip(result, expected):
            self.assertAlmostEqual(row["score"], score)
        self.assertEqual(index.search("xyz"), [])
        self.assertEqual(tokens("ALDH1, IL-6"), ["aldh1", "il", "6"])

    def test_rrf_manual(self):
        dense = [{"doc_id": "10"}, {"doc_id": "2"}]
        bm25 = [{"doc_id": "3"}, {"doc_id": "2"}]
        result = rrf(dense, bm25)
        self.assertEqual([row["doc_id"] for row in result], ["2", "3", "10"])
        self.assertAlmostEqual(result[0]["score"], 2 / 62)
        self.assertEqual(result[1]["contributions"]["dense"], {"rank": None, "score": 0.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
