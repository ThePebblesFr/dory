from __future__ import annotations

import numpy as np


def clip8(x: int) -> int:
    return max(0, min(255, int(x)))


def pulp_nn_quant_u8(phi: int, m: int, d: int) -> int:
    """Match PULP's pulp_nn_quant_u8 exactly.

    C reference:
        int32_t x = (m * phi) >> d;
        uint8_t res = clip8(x);

    Important detail: m * phi must overflow like int32 arithmetic before shifting.
    """
    phi_i32 = np.int32(phi)
    m_i16 = np.int16(m)
    prod_i64 = np.int64(m_i16) * np.int64(phi_i32)
    prod_i32 = np.int32(prod_i64)
    x_i32 = np.int32(prod_i32 >> np.int8(d))
    return clip8(int(x_i32))


def fc_u8_u8_i8(
    x_u8: np.ndarray,
    w_i8: np.ndarray,
    b_i32: np.ndarray,
    out_mult: int,
    out_shift: int,
) -> np.ndarray:
    x_u8 = np.asarray(x_u8, dtype=np.uint8).reshape(-1)
    w_i8 = np.asarray(w_i8, dtype=np.int8)
    b_i32 = np.asarray(b_i32, dtype=np.int32).reshape(-1)

    y = np.zeros(w_i8.shape[0], dtype=np.uint8)
    for i in range(w_i8.shape[0]):
        phi = int(b_i32[i]) + int(np.dot(w_i8[i].astype(np.int64), x_u8.astype(np.int64)))
        y[i] = pulp_nn_quant_u8(phi, out_mult, out_shift)
    return y


def fc_u8_i32_i8(
    x_u8: np.ndarray,
    w_i8: np.ndarray,
    b_i32: np.ndarray,
) -> np.ndarray:
    x_u8 = np.asarray(x_u8, dtype=np.uint8).reshape(-1)
    w_i8 = np.asarray(w_i8, dtype=np.int8)
    b_i32 = np.asarray(b_i32, dtype=np.int32).reshape(-1)

    y = np.zeros(w_i8.shape[0], dtype=np.int32)
    for i in range(w_i8.shape[0]):
        phi = int(b_i32[i]) + int(np.dot(w_i8[i].astype(np.int64), x_u8.astype(np.int64)))
        y[i] = np.int32(phi)
    return y

def fc_u8_u8_i8_onnx_requant(x_u8, w_i8, b_i32, out_mult, out_shift, post_div_add=0.0):
    x_u8 = np.asarray(x_u8, dtype=np.uint8).reshape(-1)
    w_i8 = np.asarray(w_i8, dtype=np.int8)
    b_i32 = np.asarray(b_i32, dtype=np.int32).reshape(-1)

    y = np.zeros((w_i8.shape[0],), dtype=np.uint8)
    div_val = float(2 ** int(out_shift))

    for i in range(w_i8.shape[0]):
        phi = int(b_i32[i]) + int(np.dot(w_i8[i].astype(np.int64), x_u8.astype(np.int64)))

        # same overflow behaviour as target
        prod_i64 = np.int64(np.int16(out_mult)) * np.int64(np.int32(phi))
        prod_i32 = np.int32(prod_i64)

        z = float(prod_i32) / div_val
        z = z + float(post_div_add)
        z = np.floor(z)

        if z < 0:
            z = 0
        elif z > 255:
            z = 255

        y[i] = np.uint8(z)

    return y

SUPPORTED_KERNELS = {
    "Linear+RequantShift": fc_u8_u8_i8,
    "Linear": fc_u8_i32_i8,
}
