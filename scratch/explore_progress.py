"""Explore how to hook into Demucs/tqdm progress for MCP progress notifications."""

import inspect
import tqdm

# 1. Can we subclass or monkey-patch tqdm to intercept updates?
print("=== tqdm.__init__ signature ===")
print(inspect.signature(tqdm.tqdm.__init__))
print()

# 2. Check if tqdm has a callback parameter
print("=== tqdm.tqdm public methods ===")
for name in sorted(dir(tqdm.tqdm)):
    if not name.startswith("_"):
        print(f"  {name}")
print()

# 3. Look at how demucs uses tqdm in apply_model
from demucs.apply import apply_model
src = inspect.getsource(apply_model)
# Find the tqdm usage
for i, line in enumerate(src.split("\n")):
    if "tqdm" in line or "progress" in line:
        print(f"  line {i}: {line.strip()}")
print()

# 4. Check if we can pass a custom class via tqdm's `file` or override
# The key question: can we intercept each tqdm.update() call?
print("=== tqdm subclass test ===")
class ProgressCapture(tqdm.tqdm):
    def update(self, n=1):
        super().update(n)
        print(f"    progress: {self.n}/{self.total}")

items = list(range(5))
bar = ProgressCapture(items, disable=False, file=open("/dev/null", "w"))
for _ in bar:
    pass
bar.close()
print()

# 5. Check how demucs imports tqdm — this determines the monkey-patch strategy
import demucs.apply
print(f"=== demucs.apply.tqdm ===")
print(f"  type: {type(demucs.apply.tqdm)}")
print(f"  demucs.apply.tqdm.tqdm: {demucs.apply.tqdm.tqdm}")
print()

# demucs does `tqdm.tqdm(futures, ...)` — so we can monkey-patch demucs.apply.tqdm.tqdm
# with our own subclass that reports progress via a callback
print("=== Monkey-patch strategy ===")
print("  demucs.apply line 116: futures = tqdm.tqdm(futures, ...)")
print("  We can set demucs.apply.tqdm.tqdm = OurCustomClass")
print("  Our class wraps __iter__ to call a progress callback on each step")