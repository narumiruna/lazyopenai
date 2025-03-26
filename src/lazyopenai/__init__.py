import os
import sys
from typing import Final

from loguru import logger

from .function_schema import generate_function_schema
from .lazy import create_chat
from .lazy import generate

LOGURU_LEVEL: Final[str] = os.getenv("LOGURU_LEVEL", "INFO")
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGURU_LEVEL}])
