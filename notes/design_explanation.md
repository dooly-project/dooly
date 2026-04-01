# Design Philosophy: Hooks vs Propagation

## The Fundamental Distinction

The taint tracking system uses two complementary mechanisms for different categories of operations:

### 1. **Hooked Operations** (hooks.py)
Operations that **manipulate tensor structure** with **explicit semantic intent**

### 2. **Propagated Operations** (propagation.py)
Operations that **transform tensor values** through **mathematical computation**

---

## Hooked Operations: Structure Manipulation

These operations are explicitly hooked because they:
1. **Change tensor structure** in semantically meaningful ways
2. **Have explicit dimension semantics** that can be tracked
3. **May compute new dimensions** that need registry entries

### Categories of Hooked Operations:

#### A. Shape Transformation
- `view`, `reshape` - Reinterpret tensor layout
- `flatten`, `unflatten` - Structured flattening
- **Why hooked**: Can compute dimensions (e.g., `-1`), need to register computed values

#### B. Dimension Permutation
- `permute`, `transpose` - Reorder dimensions
- **Why hooked**: Explicit dimension mapping can be tracked precisely

#### C. Dimension Manipulation
- `squeeze`, `unsqueeze` - Add/remove dimensions
- `expand`, `broadcast` - Repeat dimensions
- **Why hooked**: Clear semantic changes to tensor structure

#### D. Tensor Combination
- `cat`, `stack` - Combine multiple tensors
- `split`, `chunk`, `unbind` - Split tensors
- **Why hooked**: Need to track how taints flow between multiple tensors

#### E. Special Operations with Known Semantics
- `matmul`, `linear` - Matrix operations with specific taint rules
- `einops.rearrange`, `einops.reduce` - Pattern-based transformations
- **Why hooked**: Have domain-specific taint propagation rules

---

## Propagated Operations: Value Transformation

These operations rely on propagation.py because they:
1. **Transform values** without changing structure
2. **Don't have explicit dimension semantics** to track
3. **Follow general mathematical patterns**

### Categories of Propagated Operations:

#### A. Element-wise Operations
- `sin`, `cos`, `exp`, `log` - Mathematical functions
- `add`, `mul`, `div` - Arithmetic operations
- **Why propagated**: Shape unchanged, taints pass through directly

#### B. Convolutions and Filters
- `conv1d`, `conv2d`, `conv3d` - Convolution operations
- `avg_pool`, `max_pool` - Pooling operations
- **Why propagated**: Output shape follows mathematical formula, no explicit semantics

#### C. Reduction Operations
- `sum`, `mean`, `max`, `min` - Aggregations
- **Why propagated**: Standard dimension reduction patterns

#### D. Custom/Fused Operations
- `unified_attention`, `flash_attention` - Complex fused kernels
- Custom CUDA operations
- **Why propagated**: Black-box operations, can only infer from shapes

---

## The Design Narrative

### Why This Distinction?

1. **Semantic Precision vs Pattern Inference**
   - Hooked: We know EXACTLY what the operation intends (e.g., "combine batch and sequence dims")
   - Propagated: We can only INFER relationships from shape changes

2. **Dimension Computation vs Dimension Preservation**
   - Hooked: May create new dimension values via computation (reshape with -1)
   - Propagated: Output dimensions follow mathematical formulas

3. **Explicit Intent vs Implicit Transformation**
   - Hooked: Developer explicitly manipulates structure (`x.view(batch, -1)`)
   - Propagated: Mathematical operations with side-effect shapes (`conv(x)`)

### Example Comparison:

```python
# HOOKED: Explicit structural intent
x = torch.randn(60, 768)  # (NUM_TOKS, MODEL_CONFIG)
y = x.view(60, 12, 64)     # Developer explicitly factors 768 = 12 * 64
# Hook can track: 768 → 12 (NUM_HEADS) × 64 (HEAD_DIM)

# PROPAGATED: Mathematical computation
x = torch.randn(60, 768)  # (NUM_TOKS, MODEL_CONFIG)
y = torch.sin(x)           # Still (60, 768), no structural change
# Propagation: Direct shape match, preserve taints

# PROPAGATED: Complex operation with formula-based output
x = torch.randn(1, 3, 224, 224)  # (BATCH, CHANNELS, HEIGHT, WIDTH)
conv = nn.Conv2d(3, 64, 3, padding=1)
y = conv(x)  # (1, 64, 224, 224) - output follows conv formula
# Propagation: Infer from shape transformation patterns
```

### The Registry's Role:

- **Only hooked operations** update the registry (specifically reshape/view)
- **Why?** They CREATE new dimension values through computation
- Registry stores these computed dimensions for later reference
- Propagated operations don't need registry - they follow patterns

---

## Design Principles

1. **Hook operations with semantic intent** - When we can understand what the developer intends
2. **Propagate operations with mathematical patterns** - When shapes follow formulas
3. **Register computed dimensions** - Only when dimensions are created, not transformed
4. **Prefer precision over inference** - Hook when we can be precise, propagate when we must infer

---

## Summary

The distinction is fundamentally about **intent vs effect**:

- **Hooks** capture the developer's **intent** ("reshape this to separate heads")
- **Propagation** infers from the **effect** ("output is this shape, figure out taints")

This two-mechanism design provides:
- Precision where possible (hooks)
- Flexibility where needed (propagation)
- A clear boundary based on operation semantics