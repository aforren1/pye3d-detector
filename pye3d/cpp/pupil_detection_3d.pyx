import cv2
import numpy as np

cimport numpy as np

# Structuring element for the ROI MORPH_OPEN. Cached at import (was rebuilt on
# every get_edges call).
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

from .common_types cimport MatrixXd, Vector3d

from ..geometry.projections import (
    project_circle_into_image_plane,
    project_point_into_image_plane,
    unproject_edges_to_sphere,
)
from ..geometry.utilities import normalize

_EYE_RADIUS_DEFAULT: float = 10.392304845413264
# Cap the 3D-search ROI half-window to this fraction of the smaller sensor
# dimension. Bounds the morphology/median latency tail on pathological (blink)
# frames where the eyeball estimate collapses and the window would otherwise
# balloon toward the full frame. Chosen above the normal ROI range (which peaks
# well under it) so valid frames are untouched.
_ROI_MAX_AXIS_FRAC: float = 0.3

cdef extern from "search_on_sphere.h":

    cdef struct numpy_matrix_view:
        double * data
        unsigned int rows
        unsigned int cols

    cdef struct numpy_vector3d:
        double x
        double y
        double z

    cdef struct search_3d_result:
        MatrixXd inliers
        Vector3d gaze_vector
        double pupil_radius

    # NOTE: This is basically `import search_on_sphere as c_search_on_sphere` in cython
    search_3d_result c_search_on_sphere "search_on_sphere"(
        numpy_matrix_view &,
        numpy_vector3d &,
        double &,
        numpy_vector3d &,
        double &,
        double &
    )

    #Vector3d correct_gaze_vector(numpy_matrix_view &, numpy_matrix_view &, numpy_matrix_view &, numpy_matrix_view &)

cdef eigen2np(MatrixXd data):

    d1 = data.rows()
    d2 = data.cols()
    data_np = np.zeros((d1,d2))

    for row in range(d1):
        for column in range(d2):
            data_np[row, column] = data.coeff(row,column)

    return data_np


def get_edges(frame,
              predicted_gaze_vector,
              predicted_pupil_radius,
              sphere_center,
              sphere_radius,
              focal_length,
              resolution,
              major_axis_factor=1.5):


    predicted_pupil_center = sphere_center + _EYE_RADIUS_DEFAULT * predicted_gaze_vector
    projected_pupil_center = project_point_into_image_plane(predicted_pupil_center, focal_length)
    major_axis_estimate = predicted_pupil_radius/np.linalg.norm(predicted_pupil_center)*focal_length

    x, y = projected_pupil_center
    x = x + resolution[0]/2
    y = y + resolution[1]/2
    major_axis =  major_axis_factor * major_axis_estimate
    # Cap the search half-window at a fraction of the sensor: a pupil cannot
    # plausibly fill the frame. On transient/blink frames the eyeball estimate
    # can collapse (tiny |predicted_pupil_center|), inflating major_axis toward
    # the full frame and making the morphology/median pass (which scale with ROI
    # area) the latency tail. Normal ROIs sit well below this cap, so valid
    # frames are unaffected; it only trims the pathological ones.
    major_axis = min(major_axis, _ROI_MAX_AXIS_FRAC * min(resolution[0], resolution[1]))
    N,M = frame.shape
    ymin, ymax = max(0,int(y-major_axis)), min(N,int(y+major_axis))
    xmin, xmax = max(0,int(x-major_axis)), min(M,int(x+major_axis))

    if ymin>=ymax or xmin>=xmax:
           return None, None, None, [], [ymin,ymax,xmin,xmax]

    frame_roi = frame[ymin:ymax, xmin:xmax]
    frame_roi = cv2.morphologyEx(frame_roi, cv2.MORPH_OPEN, _MORPH_KERNEL)
    frame_roi = cv2.medianBlur(frame_roi, 5)
    frame_roi = cv2.normalize(frame_roi, 0, 255, norm_type=cv2.NORM_MINMAX)
    edge_frame = cv2.Canny(frame_roi, 100, 100, 5)

    edges = []
    if cv2.countNonZero(edge_frame)>0:
        edges = np.asarray(cv2.findNonZero(edge_frame))
        edges = edges[:,0,:] + np.asarray([xmin, ymin])

    return frame[ymin:ymax, xmin:xmax].copy(), frame_roi, edge_frame, edges, [ymin,ymax,xmin,xmax]

def search_on_sphere(edges,
                  predicted_gaze_vector,
                  predicted_pupil_radius,
                  sphere_center,
                  sphere_radius,
                  focal_length,
                  resolution):

    edges_on_sphere, idxs = unproject_edges_to_sphere(
        edges, focal_length, sphere_center, sphere_radius, resolution[0], resolution[1]
    )
    if len(edges_on_sphere)<=0:
         return np.asarray([0.,0.,-1.]), 0.0, [], []

    # convert edges numpy array to custom struct for passing to c++
    if edges_on_sphere.shape[0] == 0:
        return np.asarray([0, 0, -1]), 0, [], edges_on_sphere
    # first: make sure 2d data is c-contiguous, otherwise make c-contiguous copies
    if not edges_on_sphere.flags["F_CONTIGUOUS"]:  #EIGEN USES COLUMN-MAJOR MEMORY ALIGNMENT (AS FORTRAN DOES)
        edges_on_sphere = np.asfortranarray(edges_on_sphere)
    # then create cython memory view, for raw pointer access
    cdef double [:, :] edges_on_sphere_memview = edges_on_sphere
    # then copy data information
    # NOTE: This will provide direct access to the underlying data of the numpy array
    #  (given that it is c-contiguous and we did not copy). Therefore watch out to only
    # use this memory view as long as the numpy array exists. Also you shouldn't
    # probably change the underlying data.
    cdef numpy_matrix_view edges_on_sphere_memstruct
    edges_on_sphere_memstruct.data = &edges_on_sphere_memview[0, 0]
    edges_on_sphere_memstruct.rows = edges_on_sphere_memview.shape[0]
    edges_on_sphere_memstruct.cols = edges_on_sphere_memview.shape[1]

    # convert numpy vectors to custom vector representation for passing to c++
    cdef numpy_vector3d predicted_gaze_vector_struct
    predicted_gaze_vector_struct.x = predicted_gaze_vector[0]
    predicted_gaze_vector_struct.y = predicted_gaze_vector[1]
    predicted_gaze_vector_struct.z = predicted_gaze_vector[2]
    cdef numpy_vector3d sphere_center_struct
    sphere_center_struct.x = sphere_center[0]
    sphere_center_struct.y = sphere_center[1]
    sphere_center_struct.z = sphere_center[2]

    result = c_search_on_sphere(edges_on_sphere_memstruct,
                      predicted_gaze_vector_struct,
                      predicted_pupil_radius,
                      sphere_center_struct,
                      sphere_radius,
                      focal_length)

    gaze_vector = np.asarray([result.gaze_vector[0],
                              result.gaze_vector[1],
                              result.gaze_vector[2]])
    pupil_radius = result.pupil_radius
    inliers = eigen2np(result.inliers).T

    return gaze_vector, pupil_radius, inliers, edges_on_sphere
