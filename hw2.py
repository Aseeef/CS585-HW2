import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Any, Tuple, Union
from numpy import ndarray, dtype, generic
from cv2 import Mat, UMat
from cv2 import VideoCapture
from cv2.gapi.wip.draw import Image


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

def calc_area(img: np.ndarray):
    return np.count_nonzero(img)

def find_centroid(img: np.ndarray):
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

def find_axis_of_least_inertia(img: Union[UMat, np.ndarray], display_visual: bool):
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

    return area, (x, y), theta


def mask_image(img: Union[UMat, np.ndarray]) -> UMat:
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


def apply_smoothing(img: Union[UMat, np.ndarray]):
    return cv.medianBlur(img, 13)


def convert_to_binary(img: Union[UMat, np.ndarray]):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 20, 255, cv.THRESH_BINARY)
    return img


def binary_img_extract_largest_obj(img: Union[UMat, np.ndarray]):
    # find largest object in the binary image
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    # if no contours, return the black, empty image
    if len(contours) == 0:
        return None

    # draw the largest object
    x1, y1, x2, y2 = cv.boundingRect(contours[0])

    # get rid of everything except the largest object
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv.drawContours(mask, contours, 0, (255, 255, 255), -1)
    img = cv.bitwise_and(img, mask)
    # fill holes in the contour
    cv.fillPoly(img, pts=[contours[0]], color=(255, 255, 255))

    return img, (x1, y1, x2, y2)

def move_obj_to_center(img: Union[UMat, np.ndarray], centroid: Tuple[int, int]):
    (rows, cols) = img.shape

    img_center_x = rows / 2
    img_center_y = cols / 2

    # translate to center
    M = np.float32([[1, 0, img_center_y - centroid[1]], [0, 1, img_center_x - centroid[0]]])
    img = cv.warpAffine(img, M, (cols, rows))

    return img

def scale_obj(img: Union[UMat, np.ndarray], region_of_interest):
    x1, y1, x2, y2 = region_of_interest

    object_roi = img[y1 : y1+y2, x1 : x1+x2]

    # Define the scaling factor (e.g., scale by 0.7)
    scale_factor = 0.7

    # Resize the object ROI
    scaled_object_roi = cv.resize(object_roi, None, fx=scale_factor, fy=scale_factor)

    # Update the original image with the scaled object
    img = np.zeros(img.shape, dtype=np.uint8)
    img[y1 : y1+ round((y2*scale_factor)), x1 : x1+round((x2*scale_factor))] = scaled_object_roi

    return img


def rotate_at_center(img: Union[UMat, np.ndarray], theta) -> UMat:
    (rows, cols) = img.shape

    # rotate at center
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), -math.degrees(theta), 1)
    img = cv.warpAffine(img, M, (cols, rows))

    return img


def main():
    webcam = VideoCapture(0)

    while True:
        status, frame = webcam.read()
        if not status:
            print('Failed to capture frame')
            return

        plt_show_img("Original", frame)

        # mask to find skin colored obj
        frame = mask_image(frame)
        plt_show_img("Masked", frame)

        # apply blurring to remove noise
        frame = apply_smoothing(frame)
        # convert to a binary image
        frame = convert_to_binary(frame)
        # make contours around objects and extract the contour
        # with the largest area (the biggest obj) while also
        # filling in the extracted shape (further removing noise)
        frame, region_of_interest = binary_img_extract_largest_obj(frame)

        # if no obj is detected, continue
        if frame is None:
            continue

        # now scale the object in preperation for when we rotate it
        # (because we don't want it to get cropped)
        frame = scale_obj(frame, region_of_interest)
        # find the axis of least inertia
        area, centroid, theta = find_axis_of_least_inertia(frame, True)
        # translate the centroid of the obj to the center of the image
        frame = move_obj_to_center(frame, centroid)
        # rotate the image based of the axis of least inertia
        frame = rotate_at_center(frame, theta)

        plt_show_img("Rotated & Centered", frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break


if __name__ == "__main__":
    main()
