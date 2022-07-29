import functools
import numba
import numpy as np
import numpy.typing as npt
import cmath

# TODO : introduce the slow/fast/safe modes?
# groups are assumed to be a sequence of ints increasing by 0 or 1.


@numba.jit
def sum_in_groups(
    xx: npt.NDArray[float],
    groups: npt.NDArray[int],
) -> npt.NDArray[float]:
    """Sum values xx in groups.

    Arguments:
        xx (np.array): Values to sum up in groups.
        groups (np.array): A mapping between xx index and group.

    Returns:
        np.array: The sum of elements in groups.
    """
    N = groups[-1] - groups[0] + 1
    res = np.zeros(shape=(N,), dtype=xx.dtype)
    g_prev = groups[0]
    i = 0
    for x, g in zip(xx, groups):
        i += g - g_prev  # if 0 does not move, else adds one
        res[i] += x
        g_prev = g
    return res


@numba.jit
def dot_product(xx, yy):
    res = 0.0
    for x, y in zip(xx, yy):
        res += x * y
    return res


@numba.jit
def count_in_groups(groups):
    res = np.zeros(shape=(groups[-1],), dtype=np.uint32)
    g_prev = groups[0]
    i = 0
    for g in groups:
        i += g - g_prev
        res[i] += 1
        g_prev = g
    return res


# TODO: this should likely go somewhere else...
@numba.jit
def weighted_variance_in_groups(xx, weights, centers, groups):
    res = np.zeros(shape=(groups[-1],), dtype=xx.dtype)
    g_prev = groups[0]
    i = 0
    for x, w, g in zip(xx, weights, groups):
        i += g - g_prev
        res[i] += w * (x - centers[i]) ** 2
        g_prev = g
    return res


@numba.jit
def reduce_groups(groups):
    res = np.empty(shape=(groups[-1] - groups[0] + 1,), dtype=groups.dtype)
    g_prev = groups[0]
    i = 0
    for g in groups:
        i += g - g_prev
        res[i] = g
        g_prev = g
    return res


@numba.jit
def group_unique_count(xx, groups):
    res = np.zeros(shape=(groups[-1],), dtype=np.uint32)
    g_prev = groups[0]
    i = 0
    curr_elements = set([xx[0]])
    for x, g in zip(xx, groups):
        i += g - g_prev
        if g == g_prev:
            curr_elements.add(x)
        else:
            res[i - 1] = len(curr_elements)
            curr_elements = set([x])
        g_prev = g
    res[i] = len(curr_elements)
    return res


@numba.jit
def min_strictly_above_threshold(xx, threshold=0.0):
    res = cmath.inf
    for x in xx:
        if x > threshold:
            res = min(res, x)
    return res


# def cast_to_numpy_array(xx):
#     if isintance(xx, np.array):
#         return xx
#     if isintance(xx, pd.Series):
#         return xx.to_numpy()
#     return np.array(xx)


# def cast_inputs_to_numpy(foo):
#     @wraps(foo)
#     def wrapper(*args, **kwargs):
#         return foo(
#             *(cast_to_numpy_array(x) for x in args),
#             **{name: cast_to_numpy_array(x) for name, x in kwargs.items()}
#         )
#     return wrapper


@numba.jit
def normalize_to_sum(xx):
    return xx / np.sum(xx)


@numba.jit
def get_weighted_mean_and_variance(xx, weights):
    weighted_mean = np.sum(xx * weights)
    weighted_var = np.sum(weights * (xx - weighted_mean) ** 2)
    return weighted_mean, weighted_var


@numba.jit
def dot_product(a, b):
    return np.sum(a * b)


@numba.jit(boundscheck=True)
def unique_in_consecutive_groups(xx):
    x_prev = xx[0]
    x_unique = [x_prev]
    for x in xx:
        if x != x_prev:
            x_unique.append(x)
        x_prev = x
    return np.array(x_unique)


@numba.jit(boundscheck=True)
def sum_in_consecutive_groups(
    xx: npt.NDArray,
    groups: npt.NDArray[int],
    group_cnt: int,
) -> npt.NDArray[float]:
    res = np.zeros(shape=(group_cnt,), dtype=xx.dtype)
    g_prev = groups[0]
    i = 0
    for x, g in zip(xx, groups):
        i += int(g_prev != g)
        res[i] += x
        g_prev = g
    return res


@numba.jit(boundscheck=True)
def dot_product_in_consecutive_groups(
    xx: npt.NDArray,
    yy: npt.NDArray,
    groups: npt.NDArray[int],
    group_cnt: int,
) -> npt.NDArray[float]:
    res = np.zeros(shape=(group_cnt,), dtype=xx.dtype)
    g_prev = groups[0]
    i = 0
    for x, y, g in zip(xx, yy, groups):
        i += int(g_prev != g)
        res[i] += x * y
        g_prev = g
    return res


@numba.jit(boundscheck=True)
def weighted_dot_product_in_consecutive_groups(
    xx: npt.NDArray,
    yy: npt.NDArray,
    weights: npt.NDArray,
    groups: npt.NDArray[int],
    group_cnt: int,
) -> npt.NDArray[float]:
    res = np.zeros(shape=(group_cnt,), dtype=xx.dtype)
    g_prev = groups[0]
    i = 0
    for x, y, w, g in zip(xx, yy, weights, groups):
        i += int(g_prev != g)
        res[i] += x * y * w
        g_prev = g
    return res


@numba.jit(boundscheck=True)
def divide_in_consecutive_groups(
    xx: npt.NDArray,
    divisors_in_groups: npt.NDArray,
    groups: npt.NDArray[int],
) -> npt.NDArray[float]:
    res = np.zeros(shape=xx.shape, dtype=np.float64)
    g_prev = groups[0]
    i = 0
    j = 0
    for x, g in zip(xx, groups):
        i += int(g_prev != g)
        res[j] = x / divisors_in_groups[i]
        g_prev = g
        j += 1
    return res


def weighted_mean_in_consecutive_groups(xx, group_weights, groups, group_cnt):
    return dot_product_in_consecutive_groups(
        xx=xx,
        yy=group_weights,
        groups=groups,
        group_cnt=group_cnt,
    )


@numba.jit(boundscheck=True)
def weighted_var_in_consecutive_groups(
    xx, xx_weighted_means, group_weights, groups, group_cnt
):
    res = np.zeros(shape=(group_cnt,), dtype=xx.dtype)
    i = 0
    g_prev = groups[0]
    for x, w, g in zip(xx, group_weights, groups):
        i += int(g_prev != g)
        res[i] += w * (x - xx_weighted_means[i]) ** 2
        g_prev = g
    return res


def weighted_mean_and_var_in_consecutive_groups(
    xx: npt.NDArray,
    group_weights: npt.NDArray[float],
    groups: npt.NDArray[int],
    group_cnt: int,
) -> npt.NDArray:
    xx_weighted_means = weighted_mean_in_consecutive_groups(
        xx=xx, group_weights=group_weights, groups=groups, group_cnt=group_cnt
    )
    xx_weighted_vars = weighted_var_in_consecutive_groups(
        xx=xx,
        xx_weighted_means=xx_weighted_means,
        group_weights=group_weights,
        groups=groups,
        group_cnt=group_cnt,
    )
    return xx_weighted_means, xx_weighted_vars


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
