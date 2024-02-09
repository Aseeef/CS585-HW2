import math
import random
import statistics
from typing import Tuple, Union, Sequence

import cv2 as cv
import numpy as np
from cv2 import UMat
from cv2 import VideoCapture

HAND_TEMPLATES: dict = {}


def load_binary_templates():
    for i in range(0, 6):
        HAND_TEMPLATES[i] = cv.imread(f"{i}-fingers.png", cv.IMREAD_GRAYSCALE)
        if HAND_TEMPLATES[i] is None:
            print(f"ERROR loading {i}-fingers.png.")


def get_munirian_fingers(img: Union[UMat, np.ndarray]) -> int:
    # Find contours
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    # Get the largest contour
    contour = max(contours, key=cv.contourArea)

    # Find the convex hull and the convexity defects
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)

    if defects is None:
        return 0

    # Count the defects (number of fingers)
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Use triangle similarity to estimate whether the defect is between fingers
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

        # If the angle is less than 100 degrees, it's likely a finger
        if angle < np.deg2rad(100) / 2:
            count += 1

    return count


def get_aseefian_fingers(img: Union[UMat, np.ndarray]) -> Union[None, int]:
    arg_min, smallest = None, None
    # TODO: figure out the threshold
    # This threshold decides what we will even consider that it might be a potential match
    # if no template matches the threshold, then we return None since nothing matched
    threshold = 100
    for i in range(0, 6):
        res = cv.matchTemplate(img, HAND_TEMPLATES[i], cv.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        print(f'Template {i} match score:', min_val)
        if min_val > threshold:
            continue

        # note: since we are using squared diff, we want to minimize the score value
        # however in many other methods, you want to maximize.
        if smallest is None or min_val < smallest:
            smallest = min_val
            arg_min = i

    return arg_min


def get_palm(img: Union[UMat, np.ndarray], region_of_interest):
    # test different sizes to find best match
    best_sf = None
    best_val = None
    best_loc = None
    best_size = None
    for i in range(300, 100, -10):
        scale_factor = i / 100
        scaled_template = cv.resize(HAND_TEMPLATES[0].copy(), None, fx=scale_factor, fy=scale_factor)
        res = cv.matchTemplate(img, scaled_template, cv.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if best_val is None or min_val < best_val:
            best_val = min_val
            best_loc = min_loc
            best_size = scaled_template.shape
            best_sf = scale_factor

    # draw the best match
    x, y = best_loc
    h, w = best_size
    print(best_val, best_sf)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("Palm", img)


def count_fingers_around_circle(img: Union[UMat, np.ndarray], center: Tuple[int, int], radius: int):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # pi = 3.1415926535
    for theta in range(0, int(math.pi * 100 * 2), 1):
        theta = theta / 100
        x = radius * math.cos(theta) + center[0]
        y = radius * math.sin(theta) + center[1]

        img = cv.circle(img, (int(x), int(y)), 3, (0, 255, 0), 1)

    cv.imshow("TEST", img)
    ...


def calc_angle_between_points(a, b, c) -> float:
    len_ab = calc_dist_between_points(a, b)
    len_bc = calc_dist_between_points(b, c)
    len_ac = calc_dist_between_points(a, c)

    assert len_ab > 0
    assert len_bc > 0
    assert len_ac > 0

    cos_theta = (len_ab ** 2 + len_bc ** 2 - len_ac ** 2) / (2 * len_ab * len_bc)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    return theta

LEN_THRESHOLD_PIXELS: int = 16

def calc_dist_between_points(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def angle_contour_reducer(img: Union[UMat, np.ndarray], contour: UMat) -> \
        Tuple[Union[UMat, np.ndarray], Union[UMat, np.ndarray]]:
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    if len(contour) < 3:
        return img

    new_contours = [contour[0], contour[1]]

    for i in range(len(contour)):
        if i < 2:
            continue

        # using law of cosine calculate theta

        a: np.ndarray = contour[i - 2][0]
        b: np.ndarray = contour[i - 1][0]
        c: np.ndarray = contour[i][0]

        # if a b or c equal, continue
        if np.array_equal(a, b) or np.array_equal(b, c) or np.array_equal(a, c):
            continue

        theta = calc_angle_between_points(a, b, c)

        # if the change in angle is almost 180, merge the contour into 1
        if math.fabs(theta - 180.0) < 30:
            new_contours.pop()
            contour[i - 1][0] = a
            new_contours += [[c]]
        else:
            new_contours += [[c]]

        # if the change in angle is greater than 30 degrees, put a marker
        # if math.fabs(theta - 180.0) > 80:
        #     img = cv.drawMarker(img, (b[0], b[1]), color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)

    new_contours = np.array(new_contours)
    img = extract_obj_in_contour(img, new_contours)

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return img, new_contours


def defects_remover_via_angle_checking(img: Union[UMat, np.ndarray], contour: UMat):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    if len(contour) < 3:
        return img

    # if angle changes too rapidly over a short distance,
    # try to see if we can find a close alternative to smooth out the edge

    new_contours = [contour[0], contour[1]]
    total_recent_theta_change = []
    recent_distance_traveled = []

    for i in range(len(contour)):
        if i < 2:
            continue

        # using law of cosine calculate theta
        a = contour[i - 2][0]
        b = contour[i - 1][0]
        c = contour[i][0]

        # if a b or c equal, continue
        if np.array_equal(a, b) or np.array_equal(b, c) or np.array_equal(a, c):
            continue

        theta = calc_angle_between_points(a, b, c)

        total_recent_theta_change += [math.fabs(theta - 180.0)]
        recent_distance_traveled += [calc_dist_between_points(b, c)]
        while sum(recent_distance_traveled) > 13:
            total_recent_theta_change.pop(0)
            recent_distance_traveled.pop(0)

        # if the change in angle is greater than 60 degrees suddenly, draw marker, and remove
        if math.fabs(theta - 180.0) > 60:
            if len(new_contours) == 0:
                continue
            new_contours.pop()
            contour[i - 1][0] = a
            new_contours += [[c]]
        elif sum(total_recent_theta_change) > 500:
            if len(new_contours) == 0:
                continue
            new_contours.pop()
            contour[i - 1][0] = a
        else:
            new_contours += [[c]]

    new_contours = np.array(new_contours)
    img = extract_obj_in_contour(img, new_contours)

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return img, new_contours


def plt_show_img(name: str, img: UMat):
    # plt.imshow(img)
    # plt.plot()
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, img)
    # cv.waitKey(0)


def rescaleFrame(frame: Union[UMat, np.ndarray], scale) -> UMat:
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def calc_area(img: np.ndarray) -> int:
    return np.count_nonzero(img)


def find_centroid(img: np.ndarray) -> Tuple[float, float]:
    area = calc_area(img)

    # calculate the first moment
    # m10 = 0
    # m01 = 0
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j] > 0:
    #             m10 += i
    #             m01 += j

    # above code works, but numpy is faster!
    m10 = np.sum(np.where(img > 0)[0])
    m01 = np.sum(np.where(img > 0)[1])

    # calculate the centroid
    x = m10 / area
    y = m01 / area

    return x, y


def find_axis_of_least_inertia(img: Union[UMat, np.ndarray], display_visual: bool) -> Tuple[
    int, Tuple[int, int], float]:
    area = calc_area(img)
    x, y = find_centroid(img)

    # calculate the centroid
    a = np.sum((np.where(img > 0)[0] - x) ** 2)
    b = 2 * np.sum((np.where(img > 0)[0] - x) * (np.where(img > 0)[1] - y))
    c = np.sum((np.where(img > 0)[1] - y) ** 2)

    # calculate the angle of least inertia
    theta = 0.5 * np.arctan2(b, a - c)

    # draw the angle line with midpoint at centroid (using polar to cartesian conversion_
    axis_len = 100
    x1 = int(x + (axis_len * np.cos(theta)))
    y1 = int(y + (axis_len * np.sin(theta)))
    x2 = int(x - (axis_len * np.cos(theta)))
    y2 = int(y - (axis_len * np.sin(theta)))

    if display_visual:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        # draw line of least inertia
        img = cv.line(img, (y1, x1), (y2, x2), (0, 255, 0), 2)
        # draw marker at the centroid
        img = cv.drawMarker(img, (int(y), int(x)), color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10,
                            thickness=2)
        cv.putText(img, f'{theta}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        plt_show_img("Angle of Least Inertia", img)

    return area, (int(x), int(y)), theta


def mask_image(img: Union[UMat, np.ndarray]) -> Union[UMat, np.ndarray]:
    # convert the video frame into a binary image
    # so that all pixels that look like skin color
    # are included in the binary object
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # Define the thresholds for HSV color space
    lower_hsv = np.array([0, 15, 0], dtype=np.uint8)
    upper_hsv = np.array([17, 170, 255], dtype=np.uint8)

    # Define the thresholds for YCrCb color space
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)

    # Create masks for each color space
    mask_hsv = cv.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine the masks
    mask_combined = cv.bitwise_and(mask_hsv, mask_ycrcb)

    # Apply the combined mask to the original frame
    img = cv.bitwise_and(img, img, mask=mask_combined)

    return img


def apply_smoothing(img: Union[UMat, np.ndarray]) -> Union[UMat, np.ndarray]:
    img = cv.GaussianBlur(img, (7, 7), 0)
    img = cv.medianBlur(img, 11)
    return img


def convert_to_binary(img: Union[UMat, np.ndarray]) -> Union[UMat, np.ndarray]:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 20, 255, cv.THRESH_BINARY)
    return img


def extract_obj_in_contour(img: Union[UMat, np.ndarray], contour: Union[UMat, np.ndarray]) -> Union[UMat, np.ndarray]:
    # get rid of everything except the largest object
    mask = np.zeros(img.shape, dtype=np.uint8)
    # fill holes in the contour
    img = cv.fillPoly(mask, pts=[contour], color=(255, 255, 255))

    return img


def binary_img_extract_largest_obj(img: Union[UMat, np.ndarray]) -> Tuple[
    Union[None, Union[UMat, np.ndarray]], Union[None, UMat], Union[None, Sequence[int]]
]:
    # find largest object in the binary image
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    # if no contours, return the black, empty image
    if len(contours) == 0:
        return None, None, None

    # draw the largest object
    x1, y1, x2, y2 = cv.boundingRect(contours[0])

    # get rid of everything except the largest object
    img = extract_obj_in_contour(img, contours[0])

    return img, contours[0], (x1, y1, x2, y2)


def move_obj_to_center(img: Union[UMat, np.ndarray], centroid: Tuple[int, int], region_of_interest: Sequence[int]) -> \
        Tuple[Union[UMat, np.ndarray], Sequence[int]]:
    (rows, cols) = img.shape

    img_center_x = rows / 2
    img_center_y = cols / 2

    # translate to center
    M = np.float32([[1, 0, img_center_y - centroid[1]], [0, 1, img_center_x - centroid[0]]])
    img = cv.warpAffine(img, M, (cols, rows))

    x1, x2, y1, y2 = region_of_interest
    x1 += img_center_y - centroid[1]
    x2 += img_center_y - centroid[1]
    y1 += img_center_x - centroid[0]
    y2 += img_center_x - centroid[0]

    return img, (x1, x2, y1, y2)


def scale_obj(img: Union[UMat, np.ndarray], region_of_interest: Sequence[int]) -> \
        Tuple[Union[UMat, np.ndarray], Sequence[int]]:
    x1, y1, x2, y2 = region_of_interest

    object_roi = img[y1: y1 + y2, x1: x1 + x2]

    # Define the scaling factor (e.g., scale by 0.7)
    scale_factor = 0.7

    # Resize the object ROI
    scaled_object_roi = cv.resize(object_roi, None, fx=scale_factor, fy=scale_factor)

    # Update the original image with the scaled object
    img = np.zeros(img.shape, dtype=np.uint8)

    y2 = y1 + round((y2 * scale_factor))
    x2 = x1 + round((x2 * scale_factor))

    img[y1:y2, x1:x2] = scaled_object_roi

    new_region_of_interest = y1, y2, x1, x2

    return img, new_region_of_interest


def rotate_at_center(img: Union[UMat, np.ndarray], theta: float) -> Union[UMat, np.ndarray]:
    (rows, cols) = img.shape

    # rotate at center
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), -math.degrees(theta), 1)
    img = cv.warpAffine(img, M, (cols, rows))

    return img


def show_finger_count(img: Union[UMat, np.ndarray], fingers: Union[None, int]):
    rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.putText(rgb_img, f'Fingers: {fingers}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    plt_show_img("Fingers", rgb_img)


def main():
    # Connect to web cam
    print("Connecting to webcam...")
    webcam = VideoCapture(0)

    # load templates for template matching
    print("Loading image templates...")
    load_binary_templates()

    print("Starting display..!")
    while True:
        status, frame = webcam.read()
        if not status:
            print('Failed to capture frame')
            return

        original = frame.copy()

        plt_show_img("Original", frame)

        # apply blurring to remove noise
        frame = apply_smoothing(frame)

        # mask to find skin colored obj
        frame = mask_image(frame)
        plt_show_img("Masked", frame)

        # convert to a binary image
        frame = convert_to_binary(frame)
        # make contours around objects and extract the contour
        # with the largest area (the biggest obj) while also
        # filling in the extracted shape (further removing noise)
        frame, contour, region_of_interest = binary_img_extract_largest_obj(frame)

        # if no obj is detected, continue
        if frame is None:
            continue

        plt_show_img("Pre-edge smoothing", frame)
        frame, contour = angle_contour_reducer(frame, contour)
        frame, contour = defects_remover_via_angle_checking(frame, contour)
        plt_show_img("Post-edge smoothing", frame)

        # now scale the object in preperation for when we rotate it
        # (because we don't want it to get cropped)
        frame, region_of_interest = scale_obj(frame, region_of_interest)
        # find the axis of least inertia
        area, centroid, theta = find_axis_of_least_inertia(frame, True)
        # translate the centroid of the obj to the center of the image
        frame, region_of_interest = move_obj_to_center(frame, centroid, region_of_interest)
        # rotate the image based of the axis of least inertia
        frame = rotate_at_center(frame, theta)

        if area < 22000:
            print(area)
            # add cv text to move closer
            cv.putText(original, f'Please move closer', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)


        plt_show_img("Final", original)

        #print(get_munirian_fingers(frame))

        # template match
        # fingers = get_aseefian_fingers(frame)
        # display num of fingers
        # show_finger_count(frame, fingers)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break


if __name__ == "__main__":
    main()
