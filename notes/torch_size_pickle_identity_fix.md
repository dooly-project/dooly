# Fixing torch.Size Pickle Identity Error When Patching for Taint Tracking

**Date**: 2026-02-06
**Context**: Taint tracking system for profiling vLLM with MODEL_CONFIG dimension semantics

---

## Problem Statement

When running the taint tracking profiler with torch compile enabled, we encountered this error:

```
PicklingError: Can't pickle <class 'torch.Size'>: it's not the same object as torch.Size
```

This occurred specifically when:
1. `torch.Size` was patched with our `PatchedSize` class for taint propagation
2. torch compile was enabled (it pickles FX graphs for caching)
3. Any tensor operation created shapes internally via C++ code

---

## Technical Background

### How Python Pickle Works

When pickle serializes a class instance, it follows this process:

```python
obj = some_instance
cls = type(obj)                          # Step 1: Get the class
module = sys.modules[cls.__module__]     # Step 2: Find the module
lookup = getattr(module, cls.__name__)   # Step 3: Look up class by name
assert cls is lookup                     # Step 4: IDENTITY CHECK
```

The identity check (Step 4) ensures that when we unpickle, we can reconstruct the object using the same class. If this check fails, pickle raises the error we saw.

### Our Patching Architecture

The taint tracking system patches `torch.Size` to intercept shape creation:

```python
# In hooks.py
_original_torch_size = torch.Size  # Capture original

class PatchedSize(tuple):
    def __new__(cls, sizes):
        # Extract taints from TaintedInt dimensions
        # Return TaintedShape if tainted, else plain PatchedSize
        ...

torch.Size = PatchedSize  # Replace globally
```

This allows us to propagate `MODEL_CONFIG` labels through tensor operations when dimensions come from model config attributes (hidden_size, num_heads, etc.).

### Where the Problem Originates

PyTorch is primarily implemented in C++. When C++ code needs to create a `torch.Size`, it doesn't go through our Python patch:

```cpp
// Inside PyTorch C++ (simplified)
PyObject* make_size(std::vector<int64_t> dims) {
    // Directly instantiates the original torch.Size C++ type
    // Completely bypasses Python-level torch.Size = PatchedSize
    return THPSize_New(dims);
}
```

This creates a mismatch:

| What | Class Type | Python Lookup `torch.Size` |
|------|-----------|---------------------------|
| C++ created instance | `original torch.Size` | `PatchedSize` |
| Python created instance | `PatchedSize` | `PatchedSize` |

When pickle tries to serialize a C++-created `torch.Size`:

1. `type(obj)` → original `torch.Size` class
2. Lookup `torch.Size` → finds `PatchedSize` (our patch)
3. Identity check: `original_torch.Size is PatchedSize` → **FALSE**
4. **PicklingError!**

---

## What We Tried (And Why It Didn't Work)

### Attempt 1: Adding `__reduce__` to PatchedSize

```python
class PatchedSize(tuple):
    def __reduce__(self):
        return (tuple, (tuple(self),))
```

**Why it failed**: `__reduce__` only affects instances of `PatchedSize`. The problematic instances are the **original** `torch.Size` created by C++ - they don't have our `__reduce__` method.

### Attempt 2: Setting Module/Name Attributes

```python
PatchedSize.__module__ = "torch"
PatchedSize.__name__ = "Size"
```

**Why it failed**: This changes how `PatchedSize` instances are looked up, but doesn't affect the identity check. `original_torch.Size is PatchedSize` is still False regardless of naming.

### Attempt 3: Return Original torch.Size for Clean Shapes

```python
def __new__(cls, sizes):
    if not has_taint:
        return torch._original_size(plain_sizes)  # Return original type
```

**Why it failed**: This actually made things worse. Now we're creating more original `torch.Size` instances (from Python), all of which fail the pickle identity check when `torch.Size` points to `PatchedSize`.

### Attempt 4: Inherit from torch.Size

```python
class PatchedSize(torch.Size):  # Try to inherit
    ...
```

**Why it failed**:
```
TypeError: type 'torch.Size' is not an acceptable base type
```

`torch.Size` is a C++ type exposed via pybind11. PyTorch didn't compile it with trampoline class support, so Python inheritance is impossible.

---

## The Solution: copyreg

Python's `copyreg` module provides a registry for custom pickle/unpickle functions. It allows you to register a reducer function for a specific class that bypasses the normal pickle mechanism.

### Implementation

```python
# In hooks.py, BEFORE any patching

import copyreg

_original_torch_size = torch.Size  # Capture the real class

def _pickle_original_torch_size(size_instance):
    """Custom pickle handler for original torch.Size instances."""
    return (tuple, (tuple(size_instance),))

# Register BEFORE patching torch.Size
copyreg.pickle(_original_torch_size, _pickle_original_torch_size)
```

### How copyreg Changes the Pickle Flow

**Without copyreg:**
```
C++ creates torch.Size([4096, 7168])
        ↓
pickle: type(obj) = original_torch.Size
        ↓
pickle: lookup torch.Size → PatchedSize
        ↓
pickle: original_torch.Size is PatchedSize? → NO!
        ↓
PicklingError ✗
```

**With copyreg:**
```
C++ creates torch.Size([4096, 7168])
        ↓
pickle: Is there a copyreg handler for type(obj)?
        ↓
copyreg: Yes! Use _pickle_original_torch_size()
        ↓
Returns: (tuple, ((4096, 7168),))
        ↓
Serializes as plain tuple ✓
        (No identity check needed!)
```

### Why This Works

1. **Early Registration**: We register the handler for `_original_torch_size` before any patching occurs, so we're registering for the exact C++ class that will be instantiated.

2. **Bypasses Identity Check**: When copyreg has a registered handler, pickle uses it directly without performing the class lookup and identity check.

3. **Clean Serialization**: We serialize as plain tuples, which always work. On unpickle, we get a tuple - which is fine since `torch.Size` is tuple-like anyway.

4. **Handles All Cases**: Every C++-created `torch.Size` instance will use our handler, regardless of whether it's TP=1, TP=2, or any other configuration.

---

## Complete Fix in Context

```python
# hooks.py

import copyreg
import torch

# CRITICAL: Capture originals BEFORE any patching
_original_torch_size = torch.Size

# Register pickle handler for C++-created torch.Size instances
def _pickle_original_torch_size(size_instance):
    return (tuple, (tuple(size_instance),))

copyreg.pickle(_original_torch_size, _pickle_original_torch_size)

# Now safe to define and install PatchedSize
class PatchedSize(tuple):
    def __new__(cls, sizes):
        # ... taint extraction logic ...
        if has_taint:
            return TaintedShape(tuple(plain_sizes), tuple(taints))
        return tuple.__new__(cls, plain_sizes)

    def __reduce__(self):
        # Also make PatchedSize instances pickle cleanly
        return (tuple, (tuple(self),))

def install_hooks():
    torch.Size = PatchedSize
    # ... other hooks ...
```

---

## Verification

Test both with and without copyreg to confirm the fix:

```python
import pickle
import torch

# Simulate C++ creating a torch.Size (this is what happens internally)
original_size_class = torch.Size  # Before patching

# Patch torch.Size
class PatchedSize(tuple):
    pass
torch.Size = PatchedSize

# Create an instance using the ORIGINAL class (simulates C++ behavior)
cpp_created_size = original_size_class([4096, 7168])

# Without copyreg: FAILS
try:
    pickle.dumps(cpp_created_size)
except Exception as e:
    print(f"Without copyreg: {e}")

# With copyreg: SUCCEEDS
import copyreg
copyreg.pickle(original_size_class, lambda s: (tuple, (tuple(s),)))
result = pickle.dumps(cpp_created_size)
print(f"With copyreg: SUCCESS → {pickle.loads(result)}")
```

Output:
```
Without copyreg: Can't pickle <class 'torch.Size'>: it's not the same object as torch.Size
With copyreg: SUCCESS → (4096, 7168)
```

---

## Key Takeaways

1. **C++ and Python type systems are separate**: When you patch a Python binding, C++ code doesn't see the patch.

2. **Pickle's identity check is strict**: Even if classes have the same name, `is` comparison must pass.

3. **copyreg is the escape hatch**: It lets you intercept pickle before the identity check happens.

4. **Order matters**: Capture the original class and register the handler BEFORE patching.

5. **This pattern applies broadly**: Any time you patch a C++ type exposed to Python and need pickling to work, copyreg is likely the solution.
