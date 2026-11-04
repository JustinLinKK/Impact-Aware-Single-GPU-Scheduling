"""Starting job limits for a single GPU. Dispatch still checks each FIFO head's VRAM."""
import math
import re

# Per-family starting points supplied from workload experiments.
# The published CSV mirrors this table; source-only scheduler bundles use it directly.
FAMILY_STARTS = {'A10': 3, 'A100': 5, 'H100': 8}
MAX_START = max(FAMILY_STARTS.values())
LEGACY_CEILING = 6


def hardware_prior(*, gpu_name=None, device_tflops=None, a10_tflops=None,
                   bandwidth_gbs=None, arithmetic_path_verified=False):
    """Prefer a known full-device family; otherwise use comparable dense compute and bandwidth."""
    name = (gpu_name or '').strip().upper()
    if re.search(r'\bMIG\b|\bVGPU\b|\bGRID\b', name):
        return {'alpha': 3, 'basis': 'unverified_capability_fallback', 'validated_optimum': False}
    # Accept product suffixes (e.g. A100-SXM4-80GB) but not distinct products such as A10G.
    match = re.match(r'^(?:(?:NVIDIA|TESLA)\s+)*([A-Z]+[0-9]+)(?:$|[\s-])', name)
    if match and match[1] in FAMILY_STARTS:
        return {'alpha': FAMILY_STARTS[match[1]], 'basis': 'gpu_family_table',
                'family': match[1], 'validated_optimum': False}
    if not arithmetic_path_verified:
        return {'alpha': 3, 'basis': 'unverified_capability_fallback', 'validated_optimum': False}
    for value in (device_tflops, a10_tflops, bandwidth_gbs):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('verified arithmetic path requires positive finite capabilities')
    scale = min(device_tflops / a10_tflops, bandwidth_gbs / 600.)
    return {'alpha': max(1, min(LEGACY_CEILING, math.ceil(3 * math.sqrt(scale)))),
            'basis': 'dense_precision_and_bandwidth_heuristic', 'scale': scale,
            'validated_optimum': False}
