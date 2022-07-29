import functools
import numba
import numpy as np
import numpy.typing as npt
import cmath


@numba.jit
def min_strictly_above_threshold(xx, threshold=0.0):
    res = cmath.inf
    for x in xx:
        if x > threshold:
            res = min(res, x)
    return res


def counter_dedup(hpr_datasets):
    scan_to_counter = [
        collections.Counter() for _ in range(dia_run.opentims.max_scan + 1)
    ]
    for dataset in hpr_datasets:
        for scan, tof, intensity in zip(
            dataset["scan"], dataset["tof"], dataset["intensity"]
        ):
            scan_to_counter[scan][tof] += intensity
    for scan_No, counts in enumerate(scan_to_counter, start=1):
        if len(counts):
            tofs = np.fromiter(counts.keys(), dtype=np.uint32, count=len(counts))
            intensities = np.fromiter(counts.keys(), dtype=np.uint32, count=len(counts))
            i = np.argsort(tofs)
            yield scan_No, tofs[i], intensities[i]
