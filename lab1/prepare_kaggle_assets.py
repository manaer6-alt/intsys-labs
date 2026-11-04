# Выполняется на ноутбуке ДО коммита и ДО расходования GPU.
# Сохраняет книгу, точную ревизию модели, конфигурацию и манифест.
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/lab1_kaggle.json"
DATA = ROOT / "lab1/data"


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "lab1-course-client"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    DATA.mkdir(parents=True, exist_ok=True)
    # Git не должен менять байты книги и config.json при переносе Windows -> Linux.
    (DATA / ".gitattributes").write_text(
        "pride_and_prejudice.txt -text\nmodel_config.json -text\nmanifest.json text eol=lf\n",
        encoding="utf-8")
    model = config["model"]
    if not config["revision"]:
        info = json.loads(download(f"https://huggingface.co/api/models/{model}"))
        config["revision"] = info["sha"]
        CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    revision = config["revision"]
    model_path = DATA / "model_config.json"
    if not model_path.exists():
        raw = download(f"https://huggingface.co/{model}/resolve/{revision}/config.json")
        json.loads(raw)  # Проверяем, что пришёл JSON, а не страница ошибки.
        model_path.write_bytes(raw)
    book = DATA / "pride_and_prejudice.txt"
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    if not book.exists():
        raw = download(url)
        text = raw.decode("utf-8-sig")
        if "*** START OF" not in text or "*** END OF" not in text:
            raise ValueError("Не обнаружены границы текста Gutenberg")
        book.write_bytes(raw)
    manifest = {
        "source_url": url, "title": "Pride and Prejudice", "author": "Jane Austen",
        "book_file": book.name, "book_bytes": book.stat().st_size,
        "book_sha256": hashlib.sha256(book.read_bytes()).hexdigest(),
        "selection_rule": "Полный файл сохранён; для запросов берём текст между START/END Gutenberg",
        "model": model, "model_revision": revision,
        "model_config_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    path = DATA / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Существующий манифест отличается; автоматически не заменяем")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "endpoint.yaml").write_text(
        f'base_url: "{config["base_url"]}"\nmodel: "{model}"\n'
        f'model_revision: "{revision}"\ntemperature: 0\ntop_p: 1\nseed: 0\n', encoding="utf-8")
    print("Ревизия модели:", revision)
    print("Книга SHA-256:", manifest["book_sha256"])
    print("Данные готовы. Теперь зафиксируй код, конфигурацию, данные и прогноз в Git.")


if __name__ == "__main__":
    main()
