"""ЛР3, опыт 3: запускаем Qwen на двух сериях holdout."""

from inference_dev import REPO, load_inputs as load_series, main, read_json, sha

SERIES = ("dense", "rrf_ce")
METADATA_SHA = "4832505efc0d92e5e4128bc1794ad040c9a20475b9fadacd53e3e09c2bebcc13"


def load_inputs(path):
    config, tasks, warmups = load_series(path, metadata_sha=METADATA_SHA, series=SERIES, claim_count=210)
    meta = read_json(path / "metadata.json")
    if (meta["split"], meta["baseline"], meta["selected_method"], meta["claims"], meta["tasks"]) != (
        "holdout", *SERIES, 210, 420
    ):
        raise ValueError("Нужен основной архив holdout из запуска 8abf0dc9")
    frozen = path / "selection.json"
    if sha(frozen.read_bytes()) != meta["selection_sha256"]:
        raise ValueError("Повреждён снимок выбора метода")
    if read_json(frozen) != read_json(REPO / "lab3/selection.json"):
        raise ValueError("Выбор метода в архиве и в коммите различается")
    return config, tasks, warmups


if __name__ == "__main__":
    main(series=SERIES, loader=load_inputs, inputs_metadata_sha=METADATA_SHA)
