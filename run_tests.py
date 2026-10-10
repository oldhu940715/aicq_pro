"""備援測試執行器：在沒有安裝 pytest 的環境執行 tests/ 內 pytest 風格的測試函式。
有 pytest 時請直接執行：pytest -q
"""
import importlib
import inspect
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent))
passed = failed = 0
for path in sorted(pathlib.Path("tests").glob("test_*.py")):
    mod = importlib.import_module(f"tests.{path.stem}")
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != mod.__name__:
            continue
        try:
            fn()
            passed += 1
            print(f"PASS {path.stem}::{name}")
        except Exception:
            failed += 1
            print(f"FAIL {path.stem}::{name}")
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
