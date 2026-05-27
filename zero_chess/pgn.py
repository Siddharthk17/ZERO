"""PGN export helpers."""

from __future__ import annotations


def export_pgn(moves_san: list[str], result: str, headers: dict[str, str] | None = None) -> str:
    headers = {
        "Event": "?",
        "Site": "?",
        "Date": "????.??.??",
        "Round": "?",
        "White": "ZERO",
        "Black": "ZERO",
        "Result": result,
        **(headers or {}),
    }
    headers["Result"] = result
    lines = [f'[{key} "{value}"]' for key, value in headers.items()]
    move_text = []
    for idx in range(0, len(moves_san), 2):
        number = idx // 2 + 1
        white = moves_san[idx]
        if idx + 1 < len(moves_san):
            move_text.append(f"{number}. {white} {moves_san[idx + 1]}")
        else:
            move_text.append(f"{number}. {white}")
    move_text.append(result)
    return "\n".join(lines) + "\n\n" + " ".join(move_text)
