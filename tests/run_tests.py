import inspect
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from typing import Any

# Pastikan root proyek masuk ke sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def _iter_test_cases(suite_or_case: unittest.TestSuite | unittest.TestCase | Any):
    if isinstance(suite_or_case, unittest.TestCase):
        yield suite_or_case
    elif isinstance(suite_or_case, unittest.TestSuite):
        for item in suite_or_case:
            yield from _iter_test_cases(item)


def run_all_tests():
    test_dir = Path(__file__).resolve().parent
    test_files = sorted([f for f in test_dir.glob("test_*.py") if f.name != "__init__.py"])

    total = 0
    passed = 0
    failed = 0
    errors = []

    print(f"=== Menjalankan Test Suite ({len(test_files)} file) ===")

    for tf in test_files:
        mod_name = tf.stem
        try:
            # Import module
            mod = __import__(f"tests.{mod_name}", fromlist=[mod_name])
        except Exception:  # noqa: BLE001
            failed += 1
            errors.append((f"{mod_name} (IMPORT ERROR)", traceback.format_exc()))
            print(f"E  {mod_name}: Gagal di-import")
            continue

        # Jalankan TestCase jika ada
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(mod)
        if suite.countTestCases() > 0:
            for t in _iter_test_cases(suite):
                total += 1
                res = unittest.TestResult()
                t.run(res)
                test_name = getattr(t, "_testMethodName", str(t))
                if res.wasSuccessful():
                    passed += 1
                    print(f"✓  {mod_name}::{test_name}")
                else:
                    failed += 1
                    err_msg = "".join([msg for _, msg in res.errors + res.failures])
                    errors.append((f"{mod_name}::{test_name}", err_msg))
                    print(f"✗  {mod_name}::{test_name}")

        # Jalankan fungsi test_* biasa jika bukan subclass TestCase
        for attr_name, attr_val in inspect.getmembers(mod, inspect.isfunction):
            if attr_name.startswith("test_"):
                total += 1
                try:
                    # Cek signature untuk parameter tmp_path
                    sig = inspect.signature(attr_val)
                    if "tmp_path" in sig.parameters:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            attr_val(Path(tmp_dir))
                    else:
                        attr_val()
                    passed += 1
                    print(f"✓  {mod_name}::{attr_name}")
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    errors.append((f"{mod_name}::{attr_name}", traceback.format_exc()))
                    print(f"✗  {mod_name}::{attr_name}: {e}")

    print("\n" + "=" * 50)
    print(f"Hasil Pengujian: {passed}/{total} LULUS, {failed} GAGAL")
    print("=" * 50)

    if errors:
        print("\nDetail Kegagalan:")
        for name, trace in errors:
            print(f"\n--- {name} ---")
            print(trace)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(run_all_tests())
