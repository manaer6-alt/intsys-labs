# Общий измеритель ЛР1.
# Отправляет один запрос серверу в потоковом режиме.
# Возвращает временные отметки, текст, число токенов и метрики.
# Сам по себе этот файл сервер не запускает и запросов не отправляет.

import time


def measure(client, messages, generation):
    # Параметры генерации передаются извне, а не зашиваются в измеритель.
    params = {
        **generation,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    result = {
        "text_chunk_times": [],
        "text_chunks": [],
        "status": "failed",
    }
    usage = None
    finish_reason = None

    result["t0"] = time.perf_counter()

    try:
        with client.chat.completions.create(
            messages=messages, **params
        ) as stream:
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage

                if chunk.choices:
                    choice = chunk.choices[0]

                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason

                    # Служебные фрагменты без текста не учитываем.
                    if choice.delta.content:
                        result["text_chunk_times"].append(
                            time.perf_counter()
                        )
                        result["text_chunks"].append(
                            choice.delta.content
                        )

        times = result["text_chunk_times"]

        if usage is None or not times:
            raise ValueError("Сервер не вернул текст или usage")

        n = usage.completion_tokens
        if n < 1:
            raise ValueError("Сервер сообщил нулевую длину ответа")

        result.update({
            "prompt_tokens": usage.prompt_tokens,
            "output_tokens": n,
            "finish_reason": finish_reason,
            "ttft_s": times[0] - result["t0"],
            "tpot_s": (
                (times[-1] - times[0]) / (n - 1)
                if n > 1 else None
            ),
            "e2e_s": times[-1] - result["t0"],
            "status": "completed",
        })

    except Exception as error:
        # Вызывающий код должен сохранить ошибку вместе с результатами.
        result["error"] = repr(error)

    result["text"] = "".join(result["text_chunks"])
    return result