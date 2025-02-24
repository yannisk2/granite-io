# SPDX-License-Identifier: Apache-2.0

"""
Registry of all IO Processors
"""

# Local
from granite_io.factory import ImportableFactory

_IO_FACTORY = ImportableFactory("io")
io_processor = _IO_FACTORY.decorator
make_io_processor = _IO_FACTORY.construct
