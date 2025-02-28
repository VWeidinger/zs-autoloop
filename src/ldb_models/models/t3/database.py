from dataclasses import dataclass
from pathlib import Path

COLUMN_SAMPLE_SIZE = 100


@dataclass
class FixedQueries:
    # location of query files
    path: Path


@dataclass
class Database:
    pass
