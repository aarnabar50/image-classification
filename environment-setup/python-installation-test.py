# Debug: print interpreter/env info before importing torch
import os
from pathlib import Path
import sys
print("PYTHON EXECUTABLE:", sys.executable)
print("PYTHON VERSION:", sys.version.splitlines()[0])
print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
print("sys.path (first entries):", sys.path[:5])