# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path


def load_text_file(file_name: str) -> str:
    text = Path(file_name).read_text(encoding="UTF-8")
    return text
