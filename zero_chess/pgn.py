"""Portable Game Notation (PGN) format exporter utilities."""

from __future__ import annotations

import datetime


def export_pgn(moves_san: list[str], result: str, headers: dict[str, str] | None = None) -> str:
    """Export a list of SAN moves and metadata into a standard, valid PGN string."""
    now = datetime.datetime.now()
    default_date = now.strftime("%Y.%m.%d")

    base_headers = {
        "Event": "ZERO Self-Play Game",
        "Site": "Local Laptop",
        "Date": default_date,
        "Round": "1",
        "White": "ZERO",
        "Black": "ZERO",
        "Result": result,
    }

    if headers:
        base_headers.update(headers)
    base_headers["Result"] = result

    # Format header metadata blocks
    lines = [f'[{key} "{value}"]' for key, value in base_headers.items()]
    
    # Vectorized step loops for move formatting
    move_text = []
    limit = len(moves_san)
    for idx in range(0, limit, 2):
        number = (idx >> 1) + 1
        white = moves_san[idx]
        if idx + 1 < limit:
            move_text.append(f"{number}. {white} {moves_san[idx + 1]}")
        else:
            move_text.append(f"{number}. {white}")
            
    move_text.append(result)
    
    return "\n".join(lines) + "\n\n" + " ".join(move_text)