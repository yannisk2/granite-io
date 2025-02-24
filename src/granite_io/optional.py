# SPDX-License-Identifier: Apache-2.0

"""
Utilities for optional dependencies
"""

# Standard
from contextlib import contextmanager
import logging


@contextmanager
def import_optional(extra_name: str):
    """Context manager to handle optional imports"""
    try:
        yield
    except ImportError as err:
        logging.warning(
            "%s.\nHINT: You may need to pip install %s[%s]",
            err,
            __package__,
            extra_name,
        )
        raise
