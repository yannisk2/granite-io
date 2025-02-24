# SPDX-License-Identifier: Apache-2.0

"""
Registry of all backends
"""

# Local
from granite_io.factory import ImportableFactory

_BACKEND_FACTORY = ImportableFactory("backends")
backend = _BACKEND_FACTORY.decorator
make_backend = _BACKEND_FACTORY.construct
