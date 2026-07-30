from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    requested: int
    resolved: int
    capped: bool


def _finish(text: str) -> str:
    text = re.sub(r"^\s*[-*]\s+", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] not in ".!?:":
        text += "."
    return text


def _split_unit(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [_finish(text)]
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[,;:])\s+|\s+", text)
        if piece.strip()
    ]
    result: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            result.append(_finish(current))
            current = piece
        elif len(piece) > max_chars and not current:
            result.extend(
                _finish(piece[index : index + max_chars])
                for index in range(0, len(piece), max_chars)
            )
        else:
            current = candidate
    if current:
        result.append(_finish(current))
    return result


def split_tts_text(text: str, max_chars: int = 90) -> list[str]:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    max_chars = max(80, int(max_chars))
    units: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        units.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", line)
            if sentence.strip()
        )

    segments: list[str] = []
    current = ""
    for unit in units:
        if re.match(r"^[-*]\s+", unit):
            if current:
                segments.append(_finish(current))
                current = ""
            segments.extend(_split_unit(unit, max_chars))
            continue
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > max_chars:
            segments.extend(_split_unit(current, max_chars))
            current = unit
        else:
            current = candidate
    if current:
        segments.extend(_split_unit(current, max_chars))
    return segments


def estimate_token_budget(
    text: str,
    *,
    chunk_size: int,
    configured_cap: int,
    safety_margin: float,
    minimum_tokens: int = 360,
) -> TokenBudget:
    text = str(text or "").strip()
    chunk_size = max(1, int(chunk_size))
    configured_cap = max(1, int(configured_cap))
    safety_margin = max(1.0, float(safety_margin))
    words = len(re.findall(r"\w+", text, flags=re.UNICODE))
    chars = len(re.sub(r"\s+", "", text))
    punctuation = sum(unicodedata.category(char).startswith("P") for char in text)
    seconds = max(words / 2.6 if words else 0, chars / 14.0 if chars else 0)
    seconds += punctuation * 0.5 + 1.0
    estimate = math.ceil(seconds * 12.5 * safety_margin)
    aligned = max(chunk_size, math.ceil(estimate / chunk_size) * chunk_size)
    requested = max(minimum_tokens, aligned)
    resolved = min(configured_cap, requested)
    return TokenBudget(requested=requested, resolved=resolved, capped=resolved < requested)
