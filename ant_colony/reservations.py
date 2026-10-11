"""Conservative plain-text request envelopes, never average-cost estimates.

The opt-in contract is limited to OpenAI text chat messages. It budgets one
token per serialized UTF-8 byte plus 8192 framing tokens. This is a declared
upper envelope, not an exact tokenizer count; receipts are checked against it.
Other models or message shapes retain the full configured ceiling.
"""

from decimal import Decimal

from .context import byte_size


FRAMING_TOKENS = 8192


def input_envelope(settings, *, enabled=False, messages=None, context_bytes=None):
    ceiling = settings["input_token_ceiling"]
    if not enabled or not settings["id"].startswith("openai/"):
        return ceiling
    if messages is not None:
        if not (isinstance(messages, list) and 1 <= len(messages) <= 2 and all(
            isinstance(m, dict) and set(m) == {"role", "content"}
            and m["role"] in {"system", "user"} and isinstance(m["content"], str)
            for m in messages
        )):
            return ceiling
        context_bytes = byte_size(messages)
    if context_bytes is None:
        return ceiling
    return min(ceiling, context_bytes + FRAMING_TOKENS)


def request_reservation(settings, max_tokens, **kwargs):
    tokens = input_envelope(settings, **kwargs)
    dollars = (Decimal(str(settings["input_usd_per_million"])) * tokens
               + Decimal(str(settings["output_usd_per_million"])) * max_tokens) / 1_000_000
    return tokens, dollars
