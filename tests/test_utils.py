# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import datetime
import re


def load_text_file(file_name: str) -> str:
    text = Path(file_name).read_text(encoding="UTF-8")
    return text


def fix_granite_date(prompt_with_wrong_date: str) -> str:
    """
    Replace the date string in a Granite prompt with today's date, in Granite prompts'
    date format
    """
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    return re.sub(
        r"Today's Date: \w+ \d\d, \d\d\d\d\.",
        f"Today's Date: {date_str}.",
        prompt_with_wrong_date,
    )
