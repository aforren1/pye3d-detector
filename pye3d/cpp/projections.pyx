import logging
import traceback
import warnings
from functools import wraps

import cython
import numpy as np

from libc.math cimport sqrt
from libcpp.pair cimport pair

from .common_types cimport Vector3d

from ..geometry.primitives import Circle, Conic, Conicoid

logger = logging.getLogger(__name__)

cdef extern from "unproject_conicoid.h":

    cdef struct Circle3D:
        Vector3d center
        Vector3d normal
        double radius

    cdef pair[Circle3D, Circle3D] unproject_conicoid(
        const double a,
        const double b,
        const double c,
        const double f,
        const double g,
        const double h,
        const double u,
        const double v,
        const double w,
        const double focal_length,
        const double circle_radius
    )

def raise_np_errors(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        old_settings = np.seterr(all="raise")
        result = f(*args, **kwds)
        np.seterr(**old_settings)
        return result
    return wrapper

@raise_np_errors
def unproject_ellipse(ellipse, focal_length, radius=1.0):
    cdef Circle3D c
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            conic = Conic(ellipse)
            pupil_cone = Conicoid(conic, [0, 0, -focal_length])

            circles = unproject_conicoid(
                pupil_cone.A,
                pupil_cone.B,
                pupil_cone.C,
                pupil_cone.F,
                pupil_cone.G,
                pupil_cone.H,
                pupil_cone.U,
                pupil_cone.V,
                pupil_cone.W,
                focal_length,
                radius
            )

            # cannot iterate over C++ std::pair, that's why this looks so ugly
            circle_A = Circle(
                    center=(circles.first.center[0], circles.first.center[1], circles.first.center[2]),
                    normal=(circles.first.normal[0], circles.first.normal[1], circles.first.normal[2]),
                    radius=circles.first.radius
                )
            circle_B = Circle(
                    center=(circles.second.center[0], circles.second.center[1], circles.second.center[2]),
                    normal=(circles.second.normal[0], circles.second.normal[1], circles.second.normal[2]),
                    radius=circles.second.radius
                )
            # cannot iterate over C++ std::pair, that's why this looks so ugly
            if np.isnan([circle_A.radius, *circle_A.center, *circle_A.normal, circle_B.radius, *circle_B.center, *circle_B.normal]).any():
                return False
            else:
                return [circle_A, circle_B]
        except FloatingPointError:
            return False
        except Warning:
            logger.debug(f"Unexpected warning caught in:\n{traceback.format_exc()}")
            return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _fill_aux3d(double[:, :, :] a3, int i,
                             double cx, double cy, double cz,
                             double nx, double ny, double nz,
                             double eye_radius) nogil:
    # Dierkes line: origin = center - R*normal, direction = normalize(center).
    # aux_3d[i] = [ I3 - dd^T | (I3 - dd^T) @ origin ]  (projector is symmetric).
    cdef double ox = cx - eye_radius * nx
    cdef double oy = cy - eye_radius * ny
    cdef double oz = cz - eye_radius * nz
    cdef double inv = 1.0 / sqrt(cx * cx + cy * cy + cz * cz)
    cdef double dx = cx * inv, dy = cy * inv, dz = cz * inv
    cdef double p00 = 1.0 - dx * dx, p01 = -dx * dy, p02 = -dx * dz
    cdef double p11 = 1.0 - dy * dy, p12 = -dy * dz, p22 = 1.0 - dz * dz
    a3[i, 0, 0] = p00; a3[i, 0, 1] = p01; a3[i, 0, 2] = p02
    a3[i, 1, 0] = p01; a3[i, 1, 1] = p11; a3[i, 1, 2] = p12
    a3[i, 2, 0] = p02; a3[i, 2, 1] = p12; a3[i, 2, 2] = p22
    a3[i, 0, 3] = p00 * ox + p01 * oy + p02 * oz
    a3[i, 1, 3] = p01 * ox + p11 * oy + p12 * oz
    a3[i, 2, 3] = p02 * ox + p12 * oy + p22 * oz


@cython.boundscheck(False)
@cython.wraparound(False)
def build_observation_aux(double[:] c0_center, double[:] c0_normal,
                          double[:] c1_center, double[:] c1_normal,
                          double focal_length, double eye_radius):
    """Build (gaze_2d_line[4], aux_2d[2,3], aux_3d[2,3,4]) for an Observation
    from the unprojected circle pair. Pure fixed-size scalar math -- replaces a
    cluster of tiny per-frame numpy ops (eye/reshape/matmul/normalize), whose
    cost was Python/numpy dispatch overhead, not arithmetic. Mirrors the
    reference exactly (Line normalizes its direction, hence the unit vectors)."""
    gaze_2d_line = np.empty(4)
    aux_2d = np.empty((2, 3))
    aux_3d = np.empty((2, 3, 4))
    cdef double[:] g2 = gaze_2d_line
    cdef double[:, :] a2 = aux_2d
    cdef double[:, :, :] a3 = aux_3d

    # gaze ray of circle 0: origin = center, direction = normalize(normal)
    cdef double ox = c0_center[0], oy = c0_center[1], oz = c0_center[2]
    cdef double ninv = 1.0 / sqrt(c0_normal[0] * c0_normal[0]
                                  + c0_normal[1] * c0_normal[1]
                                  + c0_normal[2] * c0_normal[2])
    cdef double dx = c0_normal[0] * ninv, dy = c0_normal[1] * ninv, dz = c0_normal[2] * ninv
    # project origin (p1) and origin+dir (p2) into the image plane: (f/z)*p, take xy
    cdef double s1 = focal_length / oz
    cdef double p1x = s1 * ox, p1y = s1 * oy
    cdef double s2 = focal_length / (oz + dz)
    cdef double p2x = s2 * (ox + dx), p2y = s2 * (oy + dy)
    # gaze_2d: origin = p1, direction = normalize(p2 - p1)
    cdef double gx = p2x - p1x, gy = p2y - p1y
    cdef double ginv = 1.0 / sqrt(gx * gx + gy * gy)
    gx *= ginv; gy *= ginv
    g2[0] = p1x; g2[1] = p1y; g2[2] = gx; g2[3] = gy
    # aux_2d = [ I2 - vv^T | (I2 - vv^T) @ origin ], v = (gx, gy)
    cdef double q00 = 1.0 - gx * gx, q01 = -gx * gy, q11 = 1.0 - gy * gy
    a2[0, 0] = q00; a2[0, 1] = q01; a2[0, 2] = q00 * p1x + q01 * p1y
    a2[1, 0] = q01; a2[1, 1] = q11; a2[1, 2] = q01 * p1x + q11 * p1y

    _fill_aux3d(a3, 0, c0_center[0], c0_center[1], c0_center[2],
                c0_normal[0], c0_normal[1], c0_normal[2], eye_radius)
    _fill_aux3d(a3, 1, c1_center[0], c1_center[1], c1_center[2],
                c1_normal[0], c1_normal[1], c1_normal[2], eye_radius)

    return gaze_2d_line, aux_2d, aux_3d
