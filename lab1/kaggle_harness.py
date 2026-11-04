# Общая функция потокового измерения всех трёх опытов ЛР1.
# Асинхронность позволяет тем же измерителем обслуживать несколько клиентов.
# Функция возвращает и ошибки; вызывающая программа сохраняет их в JSONL.
import time


def metrics(t0, times, output_tokens):
    if not times or output_tokens < 1:
        raise ValueError("Нет текста или выходных токенов")
    return {
        "ttft_s": times[0] - t0,
        "tpot_s": (times[-1] - times[0]) / (output_tokens - 1) if output_tokens > 1 else None,
        "e2e_s": times[-1] - t0,
    }


async def measure(client, messages, generation):
    result = {"status": "failed", "text_chunk_times": [], "text_chunks": []}
    usage = None
    finish_reason = None
    result["t0"] = time.perf_counter()
    try:
        stream = await client.chat.completions.create(
            messages=messages, **generation,
            stream=True, stream_options={"include_usage": True},
        )
        async with stream:
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                    if choice.delta.content:
                        result["text_chunk_times"].append(time.perf_counter())
                        result["text_chunks"].append(choice.delta.content)
        if usage is None:
            raise ValueError("Сервер не вернул usage")
        result.update({"prompt_tokens": usage.prompt_tokens,
                       "output_tokens": usage.completion_tokens,
                       "finish_reason": finish_reason})
        result.update(metrics(result["t0"], result["text_chunk_times"], usage.completion_tokens))
        result["status"] = "completed"
    except Exception as error:
        result["error"] = repr(error)
    result["text"] = "".join(result["text_chunks"])
    return result
