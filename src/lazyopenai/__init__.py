import os
import sys
from typing import Final

from loguru import logger

from .lazy import create_agent
from .lazy import generate
from .lazy import parse
from .lazy import send
from .schema import generate_function_schema

LOGURU_LEVEL: Final[str] = os.getenv("LOGURU_LEVEL", "INFO")
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGURU_LEVEL}])
