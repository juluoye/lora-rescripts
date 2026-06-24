# V100 Compatibility Guide

## Problem

Tesla V100 (Compute Capability 7.0) training crashes with cuDNN 9.11+ because cuDNN dropped support for SM < 7.5:

```
RuntimeError: GET was unable to find an engine to execute this computation
```

Specifically affects Qwen Image VAE training with `QwenImageCausalConv3d` layers.

## Root Cause

- cuDNN 9.11+ removed support for Compute Capability < 7.5
- Tesla V100 is CC 7.0
- Conv3d operations fail when trying to use cuDNN

## Solutions

We provide two independent solutions in `mikazuki/utils/v100_compat/`:

---

### Solution C1: Runtime Patch (Recommended for testing)

**Quick and simple - just 3 lines of code!**

#### Usage

```python
# At the beginning of your training script
from mikazuki.utils.v100_compat import solution_c1_runtime_patch
# Patch is auto-applied on import!
```

#### Features
- ✅ Extremely simple
- ✅ Auto-detects single-frame inputs
- ✅ 3-5% performance overhead
- ✅ Can be enabled/disabled anytime

#### Best for
- Quick testing
- Development
- Emergency fixes

---

### Solution C2: Weight Conversion (Recommended for production)

**Best performance - convert once, use forever!**

#### Usage

```python
from mikazuki.utils.v100_compat.solution_c2_weight_conversion import convert_qwen_vae_for_v100

# After loading VAE
vae = AutoencoderKL.from_pretrained("your-vae-path")

# Convert for V100
vae = convert_qwen_vae_for_v100(vae)

# Now use normally
vae.to("cuda")
```

#### Features
- ✅ Zero runtime overhead
- ✅ Can save converted model
- ✅ ~5% smaller model size
- ✅ Best performance

#### Best for
- Production deployment
- Long-term training on V100
- Performance-critical scenarios

---

## Which Solution to Choose?

| Scenario | Recommended Solution |
|----------|---------------------|
| Quick testing | Solution C1 |
| Long-term production | Solution C2 |
| Development | Solution C1 |
| Maximum performance | Solution C2 |
| Not sure? | Start with C1 |

---

## Testing

Both solutions include test utilities:

```bash
# Check if you need the fix
python -c "
import torch
cc = torch.cuda.get_device_capability(0)
cudnn = torch.backends.cudnn.version()
print(f'Need fix: {cc < (7, 5) and cudnn >= 91100}')
"
```

---

## Technical Details

### How it works

For single-frame image training (depth=1), Conv3d and Conv2d are mathematically equivalent:
- `Conv3d(kernel=(1,3,3))` ≡ `Conv2d(kernel=(3,3))`
- `Conv3d(kernel=(3,3,3))[middle slice]` ≡ `Conv2d(kernel=(3,3))`

Both solutions leverage this equivalence to avoid cuDNN 3D operations.

### What's fixed in v1.0.1
- ✅ Correct stride handling (stride > 1 support)
- ✅ Chunking compatibility check
- ✅ All edge cases covered

---

## Alternative Solutions

If these don't work for you:

### Option A: Downgrade PyTorch
Edit `install.bash` line 165-168:
```bash
# Change from torch==2.10.0+cu128 to:
torch==2.3.0+cu118 torchvision==0.18.0+cu118
```
cuDNN 8.x fully supports V100.

### Option B: Global cuDNN disable (last resort)
Add to `mikazuki/script_runner.py` after line 88:
```python
if torch.cuda.is_available():
    min_cc = min(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
    if min_cc < (7, 5):
        torch.backends.cudnn.enabled = False
```
⚠️ 10-30% performance loss.

---

## Related Issue

See [Issue #38](https://github.com/WhitecrowAurora/lora-rescripts/issues/38) for detailed discussion and user feedback.

---

## Version

**Current version:** v1.0.1  
**Release date:** 2026-06-24  
**License:** MIT
