import json
import tiktoken as tikt


PRICING: dict[dict[str, any]] = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}


def get_token_count(text: str) -> int:
    encoding: tikt.Encoding = tikt.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text, disallowed_special=()))


def clean_json(resp: str) -> dict:
    try:
        resp_text = str(resp).strip("\n")
        resp_text = resp_text.replace("```json", "").replace("```", "").strip()
        return json.loads(resp_text)
    except json.JSONDecodeError:
        return {}


def print_cnt_price(
    input_text: str, output_text: str, model: str = "gpt-4o mini"
) -> None:
    input_tokens: int = get_token_count(input_text)
    output_tokens: int = get_token_count(output_text)
    input_price: float = (
        PRICING[model]["input"] * (input_tokens / 1e6) if model in PRICING else 0
    )
    output_price: float = (
        PRICING[model]["output"] * (output_tokens / 1e6) if model in PRICING else 0
    )

    print(f"\tInput tokens: {input_tokens} | ${input_price:.6f}")
    print(f"\tOutput tokens: {output_tokens} | ${output_price:.6f}")
    print(f"\tTotal Price: ${input_price + output_price:.6f}")


def truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    encoding: tikt.Encoding = tikt.encoding_for_model(model)
    tokens: list[int] = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens: list[int] = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
