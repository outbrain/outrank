"""
.. include:: ../DOCS.md
"""
from __future__ import annotations

import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logging.getLogger(__name__).setLevel(logging.INFO)
