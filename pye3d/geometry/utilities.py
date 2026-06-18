import math

import numpy as np


def cart2sph(x):
    # Scalar trig/sqrt via the math module -- far less per-call overhead than the
    # numpy ufuncs on 0-d scalars. norm of a vector == sqrt(x . x). np.arccos is
    # kept (not math.acos) to preserve the boundary behavior: a fp-overshoot
    # |arg|>1 yields nan -> handled downstream, whereas math.acos would raise.
    phi = math.atan2(x[2], x[0])
    theta = np.arccos(x[1] / math.sqrt(np.dot(x, x)))

    return phi, theta


def sph2cart(phi, theta):
    result = np.empty(3)

    sin_theta = math.sin(theta)
    result[0] = sin_theta * math.cos(phi)
    result[1] = math.cos(theta)
    result[2] = sin_theta * math.sin(phi)

    return result


def normalize(v, axis=-1):
    # Fast path for the overwhelmingly common 1-D (single vector) case,
    # sidestepping np.linalg.norm's heavy per-call Python dispatch. Use
    # sqrt(sum(v*v)) -- the *same* elementwise-square + add.reduce that
    # np.linalg.norm(v, axis=-1) performs -- so the result is bit-identical
    # (the BLAS dot path would differ by ~1 ULP).
    if isinstance(v, np.ndarray) and v.ndim == 1:
        return v / math.sqrt((v * v).sum())
    return v / np.linalg.norm(v, axis=axis)


def enclosed_angle(v1, v2, unit="deg", axis=-1):
    v1 = normalize(v1, axis=axis)
    v2 = normalize(v2, axis=axis)

    alpha = np.arccos(np.clip(np.dot(v1.T, v2), -1, 1))

    if unit == "deg":
        return 180.0 / np.pi * alpha
    else:
        return alpha


def make_homogeneous_vector(v):
    return np.hstack((v, [0.0]))


def make_homogeneous_point(p):
    return np.hstack((p, [1.0]))


def transform_as_homogeneous_point(p, trafo):
    p = make_homogeneous_point(p)
    return (trafo @ p)[:3]


def transform_as_homogeneous_vector(v, trafo):
    v = make_homogeneous_vector(v)
    return (trafo @ v)[:3]


def rotate_v1_on_v2(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    cos_angle = np.dot(v1, v2)

    if not np.allclose(np.abs(cos_angle), 1):
        u = np.cross(v1, v2)
        s = np.linalg.norm(u)
        c = np.dot(v1, v2)

        ux = np.asarray([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

        R = np.eye(3) + ux + np.dot(ux, ux) * (1 - c) / s**2

    elif np.allclose(cos_angle, 1):
        R = np.eye(3)

    elif np.allclose(cos_angle, -1):
        R = -np.eye(3)

    return R
