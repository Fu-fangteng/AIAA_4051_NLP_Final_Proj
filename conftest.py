import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_src / "task1"))
sys.path.insert(0, str(_src / "task2"))
sys.path.insert(0, str(_src / "task3"))
