# Оценка ЛР2: здесь разрешено читать эталонную разметку SciFact.
# Проверяем поиск, наличие полного обоснования, метку и ссылку ответа.
# load_gold читает dev либо явно выбранные проверочные train; holdout не оцениваем.
# score_record повторно разбирает сырой ответ, а не доверяет сохранённой метке.
# Прямой запуск — тесты на искусственных данных, без модели и без dev-эксперимента.

import json
import unittest

from prepare_data import ROOT, MANIFEST, sha256, claim_label
from answers import parse_answer, LABELS


def load_gold(source_name="claims_dev.jsonl", claim_ids=None):
    dataset = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["datasets"]["scifact"]
    if source_name == "claims_dev.jsonl":
        split_bytes = (ROOT / dataset["split_file"]).read_bytes()
        if sha256(split_bytes) != dataset["split_sha256"]:
            raise ValueError("Хеш разбиения отличается от манифеста")
        split = json.loads(split_bytes)
        if split != dataset["splits"]:
            raise ValueError("Списки разбиения отличаются от манифеста")
        if len(split["dev"]) != 90 or len(set(split["dev"])) != 90:
            raise ValueError("Ожидаются 90 уникальных ID dev")
        ids = split["dev"] if claim_ids is None else list(claim_ids)
        if not set(ids) <= set(split["dev"]):
            raise ValueError("Holdout в ЛР2 не оцениваем")
    elif source_name == "claims_train.jsonl" and claim_ids is not None:
        ids = list(claim_ids)
    else:
        raise ValueError("Допустимы dev либо явно выбранные проверочные claims_train")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Нужен непустой список уникальных ID")
    source = dataset["files"][source_name]
    raw = (ROOT / source["path"]).read_bytes()
    if sha256(raw) != source["sha256"]:
        raise ValueError("Хеш утверждений отличается от манифеста")
    allowed = set(ids)
    gold = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            claim = json.loads(line)
            if claim["id"] in allowed:
                if claim["id"] in gold or claim_label(claim) is None:
                    raise ValueError("Повтор ID или конфликтующая разметка")
                gold[claim["id"]] = claim
    if set(gold) != allowed:
        raise ValueError("Не найдены все утверждения dev")
    return gold


def oracle_plan(gold):
    # Только для oracle: выбираем одно полное обоснование.
    # Порядок фиксирован: строковый doc_id, затем номера предложений.
    plan = {}
    for claim_id, claim in gold.items():
        options = [
            (doc_id, tuple(sorted(rationale["sentences"])))
            for doc_id, rationales in claim["evidence"].items()
            for rationale in rationales
        ]
        if options:
            doc_id, sentences = min(options)
            plan[claim_id] = {"doc_id": doc_id, "sentences": list(sentences)}
    return plan


def retrieval_metrics(gold, found, k):
    relevant = set(gold["evidence"])
    if not relevant:
        return None  # NOINFO исключается из метрик поиска.
    if type(k) is not int or k <= 0:
        raise ValueError("k должен быть положительным целым числом")
    ids = [item["doc_id"] for item in found[:k]]
    first_rank = next((rank for rank, doc_id in enumerate(ids, 1) if doc_id in relevant), None)
    return {
        "hit": int(first_rank is not None),
        "recall": len(set(ids) & relevant) / len(relevant),
        "rr": 1 / first_rank if first_rank is not None else 0.0,
    }


def contains_rationale(gold, passed):
    if not gold["evidence"]:
        return None
    available = {item["doc_id"]: set(item["sentences"]) for item in passed}
    # Достаточно одного полного обоснования одного документа.
    # Предложения разных обоснований не объединяем.
    return any(
        set(rationale["sentences"]) <= available.get(doc_id, set())
        for doc_id, rationales in gold["evidence"].items()
        for rationale in rationales
    )


def score_record(gold, record):
    if gold["id"] != record["claim_id"]:
        raise ValueError("ID ответа не совпадает с ID эталона")
    label = claim_label(gold)
    if label is None:
        raise ValueError("Утверждение имеет конфликтующие метки")
    mode = record["mode"]
    passed = record.get("passed", [])
    parsed = parse_answer(record.get("raw_response"), mode, passed)
    valid = record.get("status") == "completed" and parsed["parse_status"] == "ok"
    answer = parsed["parsed"] if valid else None
    correct = valid and answer["label"] == label

    reference = None
    if mode != "llm_only" and label != "NOINFO":
        reference = bool(correct and contains_rationale(gold, [{
            "doc_id": answer["doc_id"], "sentences": answer["sentences"]
        }]))
    search = retrieval_metrics(gold, record.get("found", []), record["k"]) if mode == "rag" else None
    return {
        "claim_id": gold["id"], "gold_label": label,
        "predicted_label": answer["label"] if valid else None,
        "label_correct": bool(correct), "reference_correct": reference,
        "evidence_present": contains_rationale(gold, passed) if mode != "llm_only" else None,
        "parse_status": parsed["parse_status"],
        **(search if search is not None else {"hit": None, "recall": None, "rr": None}),
    }


def summarize(scores, expected_ids):
    expected = list(expected_ids)
    actual = [row["claim_id"] for row in scores]
    if (not expected or len(set(expected)) != len(expected)
            or len(actual) != len(expected) or set(actual) != set(expected)):
        raise ValueError("Нужна ровно одна оценка каждого запланированного утверждения")
    result = {"n": len(expected), "accuracy": sum(s["label_correct"] for s in scores) / len(expected)}
    groups = {label: [s for s in scores if s["gold_label"] == label] for label in LABELS}
    result["class_counts"] = {label: len(group) for label, group in groups.items()}
    result["recall_by_class"] = {
        label: sum(s["label_correct"] for s in group) / len(group) if group else None
        for label, group in groups.items()
    }
    for name, key in (
        ("reference_accuracy", "reference_correct"), ("evidence_coverage", "evidence_present"),
        ("hit_at_k", "hit"), ("recall_at_k", "recall"), ("mrr_at_k", "rr"),
    ):
        values = [s[key] for s in scores if s[key] is not None]
        result[name] = sum(values) / len(values) if values else None
        result[name + "_n"] = len(values)
    return result


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.gold = {"id": 1, "evidence": {
            "a": [{"label": "SUPPORT", "sentences": [0, 2]}],
            "b": [{"label": "SUPPORT", "sentences": [1]}],
        }}
        self.found = [{"doc_id": i} for i in ("x", "y", "a")]
        self.passed = [{"doc_id": "a", "sentences": [0, 1, 2]}]
        self.record = {
            "claim_id": 1, "mode": "rag", "k": 3, "status": "completed",
            "found": self.found, "passed": self.passed,
            "raw_response": json.dumps({"label": "SUPPORT", "doc_id": "a", "sentences": [0, 2]}),
        }

    def test_search_metrics(self):
        metrics = retrieval_metrics(self.gold, self.found, 3)
        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["rr"], 1 / 3)
        self.assertEqual(retrieval_metrics(self.gold, self.found, 1), {"hit": 0, "recall": 0.0, "rr": 0.0})

    def test_complete_rationale(self):
        self.assertTrue(contains_rationale(self.gold, self.passed))
        self.assertFalse(contains_rationale(self.gold, [{"doc_id": "a", "sentences": [0, 1]}]))

    def test_alternative_rationales(self):
        gold = {"evidence": {"a": [{"sentences": [0, 2]}, {"sentences": [1, 3]}]}}
        self.assertFalse(contains_rationale(gold, [{"doc_id": "a", "sentences": [0, 1]}]))
        self.assertTrue(contains_rationale(gold, [{"doc_id": "a", "sentences": [1, 3]}]))

    def test_correct_reference(self):
        score = score_record(self.gold, self.record)
        self.assertTrue(score["label_correct"])
        self.assertTrue(score["reference_correct"])

    def test_label_correct_reference_wrong(self):
        record = {**self.record, "raw_response": json.dumps({"label": "SUPPORT", "doc_id": "a", "sentences": [1]})}
        score = score_record(self.gold, record)
        self.assertTrue(score["label_correct"])
        self.assertFalse(score["reference_correct"])

    def test_failures_count_as_wrong(self):
        for change in (
            {"raw_response": "{"}, {"status": "failed"},
            {"raw_response": json.dumps({"label": "SUPPORT", "doc_id": "missing", "sentences": [0]})},
        ):
            with self.subTest(change=change):
                score = score_record(self.gold, {**self.record, **change})
                self.assertFalse(score["label_correct"])
                self.assertFalse(score["reference_correct"])

    def test_noinfo(self):
        gold = {"id": 1, "evidence": {}}
        record = {**self.record, "raw_response": json.dumps({"label": "NOINFO", "doc_id": None, "sentences": []})}
        score = score_record(gold, record)
        self.assertTrue(score["label_correct"])
        for key in ("hit", "recall", "rr", "reference_correct", "evidence_present"):
            self.assertIsNone(score[key])
        self.assertFalse(score_record(gold, {**record, "raw_response": "{"})["label_correct"])

    def test_llm_only(self):
        record = {**self.record, "mode": "llm_only", "k": None, "passed": [], "raw_response": json.dumps({"label": "SUPPORT", "doc_id": None, "sentences": []})}
        score = score_record(self.gold, record)
        self.assertTrue(score["label_correct"])
        self.assertIsNone(score["reference_correct"])
        self.assertIsNone(score["hit"])

    def test_summary_keeps_failed_answers(self):
        good = score_record(self.gold, self.record)
        bad = score_record({**self.gold, "id": 2}, {**self.record, "claim_id": 2, "status": "failed"})
        summary = summarize([good, bad], [1, 2])
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["recall_by_class"]["SUPPORT"], 0.5)
        self.assertEqual(summary["reference_accuracy"], 0.5)

    def test_summary_rejects_missing_and_duplicate(self):
        good = score_record(self.gold, self.record)
        for rows in ([good], [good, good]):
            with self.assertRaises(ValueError):
                summarize(rows, [1, 2])

    def test_oracle_plan(self):
        plan = oracle_plan({1: self.gold})
        self.assertEqual(plan[1], {"doc_id": "a", "sentences": [0, 2]})
        self.assertTrue(contains_rationale(self.gold, [plan[1]]))

    def test_oracle_skips_noinfo(self):
        self.assertEqual(oracle_plan({1: {"evidence": {}}}), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
