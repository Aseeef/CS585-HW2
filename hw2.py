"""
    The Memory Game

    The Memory Game is a game where the player is shown a sequence of numbers for a few seconds and then asked to
    remember the sequence. The player then has to guess the sequence by showing the number of fingers corresponding to
    each number in the sequence. The game then checks if the player's guess is correct or not.
    
    The game is implemented using OpenCV and uses several techniques to detect the number of fingers shown by the player.
    
    Implemented by:
    - Muhammad Aseef Imran
    - Munir Siddiqui

    Background Image Source: https://stock.adobe.com/images/white-particle-coming-from-the-background-above-squared-floor-4k/454107697?prev_url=detail
"""

import math
import statistics
import random
from typing import Tuple, Union, Sequence

import cv2 as cv
import numpy as np
from cv2 import UMat
from cv2 import VideoCapture


def hull_finger_counter(img: Union[UMat, np.ndarray], display_visual: bool, save_frame=False) -> int:

    bin_img = img

    # Find contours
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # If no contours found return
    if not contours:
        return 0

    # Get the largest contour
    contour = max(contours, key=cv.contourArea)

    # Find the centroid
    center_x, center_y = find_centroid(img)

    # For the visuals, convert the image to rgb (rn its binary)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # draw centroid
    img = cv.drawMarker(img, (int(center_y), int(center_x)), color=(0, 255, 0), markerType=cv.MARKER_CROSS,
                        markerSize=10)

    # find the bounding box
    x1, y1, w, h = cv.boundingRect(contour)

    # use the bounding box and knowledge of hand proportions to guess how big the circle radius should be
    # in our case, it's the distance to the top of the bounding box minus 1/6 of box height. We found experimentally
    # this gives a pretty good circle radius
    r: int = round((h / 2) - (h / 6))

    # checks for intersection with the fingers by drawing a circle with radius r around the hand
    fingers_counter = 0
    current_status = None
    last_status = None
    for theta in range(72, 288, 1):
        x = int((r * math.cos(math.radians(theta))) + center_x)
        y = int((r * math.sin(math.radians(theta))) + center_y)

        # print(f"Pixel at y={y}, x={x} is {bin_img[x, y]}")
        # print(f"Middle: {center_y}, {center_x} is {bin_img[int(center_x), int(center_y)]}")
        # print(f"Full img: {bin_img.shape}")
        current_status = 1 if bin_img[x, y] == 255 else 0

        if current_status == 1:
            img = cv.drawMarker(img, (y, x), color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=3)
        else:
            img = cv.drawMarker(img, (y, x), color=(0, 100, 0), markerType=cv.MARKER_CROSS, markerSize=3)

        if last_status is not None and last_status == 0 and current_status == 1:
            fingers_counter += 1
            img = cv.drawMarker(img, (y, x), color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=10)
            # print("LS1", last_status, "CS1", current_status)
        elif last_status is not None and last_status == 1 and current_status == 0:
            fingers_counter += 1
            img = cv.drawMarker(img, (y, x), color=(255, 0, 255), markerType=cv.MARKER_CROSS, markerSize=10)
            # print("LS2", last_status, "CS2", current_status)

        last_status = current_status

    # show
    if display_visual:
        plt_show_img("Primary Analysis Image", img, save_frame)

    return math.ceil(fingers_counter / 2)


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


def calc_dist_between_points(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def angle_contour_reducer(img: Union[UMat, np.ndarray], contour: UMat) -> \
        Tuple[Union[UMat, np.ndarray], Union[UMat, np.ndarray]]:
    if len(contour) < 3:
        return img, contour

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

    return img, new_contours


def defects_remover_via_angle_checking(img: Union[UMat, np.ndarray], contour: UMat):
    if len(contour) < 3:
        return img, contour

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

    return img, new_contours


def plt_show_img(name: str, img: UMat | np.ndarray, save_frame=False):
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, img)
    cv.waitKey(1)
    if save_frame:
        cv.imwrite(f"{name}.png", img)
        print(f"Written image {name}.png!")


def rescaleFrame(frame: Union[UMat, np.ndarray], scale) -> UMat:
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def setResLiveVideo(webcam: VideoCapture, width: int, framerate: int):
    # given the width we automatically figure out the height
    scale = width / webcam.get(3)
    height = int(webcam.get(4) * scale)
    webcam.set(3, width)
    webcam.set(4, height)
    # reduce frame rate
    webcam.set(cv.CAP_PROP_FPS, framerate)


def calc_area(img: np.ndarray) -> int:
    return np.count_nonzero(img)


def find_centroid(img: np.ndarray) -> Tuple[float, float]:
    area = calc_area(img)
    m10 = np.sum(np.where(img > 0)[0])
    m01 = np.sum(np.where(img > 0)[1])

    # calculate the centroid
    x = m10 / area
    y = m01 / area

    return x, y


def find_axis_of_least_inertia(img: Union[UMat, np.ndarray], display_visual: bool, save_frame=False) -> Tuple[
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
        plt_show_img("Angle of Least Inertia Image", img, save_frame)

    return area, (int(x), int(y)), theta


def calc_roundness(img: Union[UMat, np.ndarray]):
    # use Emin and Emax in moment of inertia to calculate roundedness
    # E = (1/2) (a + b) - (1/2) (a - c) cos2θ - (1/2) b sin2θ
    x, y = find_centroid(img)

    # calculate the centroid
    a = np.sum((np.where(img > 0)[0] - x) ** 2)
    b = 2 * np.sum((np.where(img > 0)[0] - x) * (np.where(img > 0)[1] - y))
    c = np.sum((np.where(img > 0)[1] - y) ** 2)

    Emin = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + (b / 2) ** 2)
    Emax = (a + c) / 2 + np.sqrt(((a - c) / 2) ** 2 + (b / 2) ** 2)

    # calculate the roundedness
    roundedness = Emin / Emax
    return roundedness


def mask_image(img: Union[UMat, np.ndarray]) -> Union[UMat, np.ndarray]:
    # Convert the image to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define the lower and upper thresholds for skin color in HSV space
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask1 = cv.inRange(hsv, lower_skin, upper_skin)

    # Convert image to CRCB
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    lower_crcb = np.array((0, 120, 70))
    upper_crcb = np.array((255, 180, 127))

    mask2 = cv.inRange(ycrcb, lower_crcb, upper_crcb)

    # Combine the masks
    mask = cv.bitwise_and(mask1, mask2)

    # Apply the mask to the original image
    img = cv.bitwise_and(img, img, mask=mask)

    return img


def apply_smoothing(img: Union[UMat, np.ndarray]) -> Union[UMat, np.ndarray]:
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


count = fingers_detected = 0
debug = False


def count_fingers():
    global count, fingers_detected

    # Connect to webcam
    # print("Connecting to webcam...")
    webcam = cv.VideoCapture(0, cv.CAP_DSHOW)
    # lower resolution and frame rate for performance
    setResLiveVideo(webcam, 500, 20)

    # print("Starting display..!")
    finger_detections = []
    while True:
        status, frame = webcam.read()
        if not status:
            print('Failed to capture frame')
            return None

        original = frame.copy()

        save_frame = False
        if debug:
            cv.imshow("Original", original)
            save_frame = (cv.waitKey(1) & 0xFF) == ord('c')

        # Apply blurring to remove noise
        frame = apply_smoothing(frame)

        if debug:
            plt_show_img("Smoothed Image", frame, save_frame)

        # Mask to find skin-colored objects
        frame = mask_image(frame)

        if debug:
            plt_show_img("Masked Image", frame, save_frame)

        # Convert to a binary image
        frame = convert_to_binary(frame)

        if debug:
            plt_show_img("Binary Image", frame, save_frame)

        # Dilate before extracting the largest object
        # This further removes noise in the skin mask
        frame = cv.dilate(frame, np.array([9, 9]), iterations=4)

        if debug:
            plt_show_img("Dilated Image", frame, save_frame)

        # Make contours around objects and extract the contour with the largest area
        # This object is hopefully our hand. This technique makes our recognition resistant
        # to a bit of extra "stuff" in the background.
        frame, contour, region_of_interest = binary_img_extract_largest_obj(frame)

        if debug:
            plt_show_img("LO Extracted Image", frame, save_frame)

        # If no object is detected, continue to the next frame because most likely
        # there is nothing here.
        if frame is None:
            continue

        # Now we need to do a 2nd round of object refining and preprocessing...

        # First we simplify the contours a bit by merging contour lines that are almost
        # parallel anyway. This helps with performance for further processing and is a
        # precursor to our next method call...
        frame, contour = angle_contour_reducer(frame, contour)

        if debug:
            plt_show_img("Contour Reduced Image", frame, save_frame)

        # This part checks for any steep changes in angle that seem unnatural.
        # Such steep changes in angle are most likely defects from masking so this method
        # removes any steep changes in angle (which are often just intrusions or extrusions
        # on the hand object
        frame, contour = defects_remover_via_angle_checking(frame, contour)

        if debug:
            plt_show_img("Defect Reduced Image", frame, save_frame)

        # The additional dilation process further reduces defects from masking
        frame = cv.dilate(frame, np.array([11, 11]), iterations=7)

        if debug:
            plt_show_img("Dilated Image 2", frame, save_frame)

        # now we extract the final hand object
        frame, contour, region_of_interest = binary_img_extract_largest_obj(frame)

        if debug:
            plt_show_img("LO2 Extracted Image", frame, save_frame)

        # Now scale the object in preparation for when we rotate it (so it doesn't get cropped off after rotation)
        frame, region_of_interest = scale_obj(frame, region_of_interest)

        if debug:
            plt_show_img("Scaled Image", frame)

        # Find the axis of least inertia using techniques learned in class
        area, centroid, theta = find_axis_of_least_inertia(frame, debug, save_frame)

        # Translate the centroid of the object to the center of the image (to reduce the chances of any
        # cropping happening due to the rotation and for easier analysis later
        frame, region_of_interest = move_obj_to_center(frame, centroid, region_of_interest)

        if debug:
            plt_show_img("Centered Image", frame, save_frame)

        # Rotate the image based on the axis of least inertia
        frame = rotate_at_center(frame, theta)

        if debug:
            plt_show_img("Rotated Image", frame, save_frame)

        # if the image area is less than 1500, then we may not be able to analyse the hand properly (or maybe
        # the hand isn't even there yet) so we just tell the user to move closer
        if area < 1500:
            # Add cv text to move closer
            cv.putText(original, f'Please move closer', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                       cv.LINE_AA)
            continue

        # The magic sauce! Basically draws a circle around the palm and based on how many times the circle intersects
        # a finger, figures out how many fingers there are
        fingers = hull_finger_counter(frame, debug, save_frame)
        # the thumb isn't detecting super well with the above technique so we also calculate the roundness of the
        # whole hand. The thumb is only active when all 5 fingers are the way we defined our gestures and at that
        # point that hand is actually pretty round. So we use the roundness of the hand also to know if all 5 fingers
        # are up
        roundness = calc_roundness(frame)
        fingers = 5 if (roundness > 0.5 and (fingers >= 4)) else fingers
        # you cant have more than 5 fingers (probably)
        fingers = min(fingers, 5)

        # we use the mean and std of the finger counter to fight of random noise as well
        # we simply don't guess anything unless we are sure by using the std
        if len(finger_detections) > 10 and statistics.stdev(finger_detections) < 0.65:
            cv.putText(original, f'Fingers: {int(statistics.mean(finger_detections))}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        elif len(finger_detections) > 10:
            cv.putText(original, f'Calculating...', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Display the final frame with finger count information
        plt_show_img("Enter your guess", original, save_frame)

        if len(finger_detections) > 10 and statistics.stdev(finger_detections) < 0.65:
            finger_count = int(statistics.mean(finger_detections))

            if count == 0 and 1 <= finger_count <= 5:
                fingers_detected = finger_count
                count += 1
            else:
                if fingers_detected == finger_count:
                    count += 1
                else:
                    count = 0

            # print(count)

            if count == 30 and not debug:
                webcam.release()
                cv.destroyWindow("Enter your guess")
                count = fingers_detected = 0
                return finger_count

        # Remove oldest detections
        finger_detections += [fingers]
        if len(finger_detections) > 20:
            finger_detections.pop(0)


def main():
    while True:

        # if debugging, skip straight to finger detection
        if debug:
            count_fingers()
            continue

        image = cv.imread("background.jpg")
        image = cv.resize(image, (1000, 2 * image.shape[0]))
        blank = image.copy()
        welcome_message = "Welcome to the Memory Game!"
        font = cv.FONT_HERSHEY_DUPLEX
        color = (255, 255, 255)
        textsize = cv.getTextSize(welcome_message, font, 1.5, 1)[0]
        textX = (image.shape[1] - textsize[0]) // 2
        textY = 70

        cv.putText(image, welcome_message, (textX, textY), font, 1.5, color, 1, cv.LINE_AA)
        cv.imshow("The Memory Game", image)
        background = image.copy()

        for i in range(7, 0, -1):
            begin_image = np.copy(image)
            begin_message = "The game begins in " + str(i)
            textsize = cv.getTextSize(begin_message, font, 1, 1)[0]
            textX = (image.shape[1] - textsize[0]) // 2
            textY = 350
            cv.putText(begin_image, begin_message, (textX, textY), font, 1, (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow("The Memory Game", begin_image)
            cv.waitKey(1000)  # Wait for 1 second

        cv.imshow("The Memory Game", background)

        NUMBER_LENGTH = 5

        number_to_guess = ""
        for _ in range(NUMBER_LENGTH):
            number_to_guess += random.choice("1234")

        guess_message = "Here is the number to remember: " + number_to_guess
        textsize = cv.getTextSize(guess_message, font, 1, 1)[0]
        textX = (image.shape[1] - textsize[0]) // 2
        textY = 350

        cv.putText(image, guess_message, (textX, textY), font, 1, color, 1, cv.LINE_AA)
        cv.imshow("The Memory Game", image)

        for i in range(5, 0, -1):
            countdown_image = np.copy(image)
            countdown_message = "Disappearing in " + str(i)
            textsize = cv.getTextSize(countdown_message, font, 1, 1)[0]
            textX = (image.shape[1] - textsize[0]) // 2
            textY = 390
            cv.putText(countdown_image, countdown_message, (textX, textY), font, 1, (0, 0, 255), 1, cv.LINE_AA)
            cv.imshow("The Memory Game", countdown_image)
            cv.waitKey(1000)  # Wait for 1 second

        cv.imshow("The Memory Game", background)

        nums_chosen = []
        number_guessed = ""

        for i in range(NUMBER_LENGTH):
            num = count_fingers()
            nums_chosen.append(num)
            number_guessed += str(num)

            guess_image = np.copy(background)
            guess_message = "You guessed: " + number_guessed
            textsize = cv.getTextSize(guess_message, font, 1, 1)[0]
            textX = (image.shape[1] - textsize[0]) // 2
            textY = 350
            cv.putText(guess_image, guess_message, (textX, textY), font, 1, color, 1, cv.LINE_AA)
            cv.imshow("The Memory Game", guess_image)
            cv.waitKey(1000)  # Wait for 1 second

        result_message = "Correct!" if number_guessed == number_to_guess else "Incorrect!"
        your_answer_message = "Your answer: " + number_guessed
        correct_answer_message = "Correct answer: " + number_to_guess

        textsize = cv.getTextSize(result_message, font, 2, 1)[0]
        result_color = (0, 255, 0) if result_message == "Correct!" else (0, 0, 255)
        textX = (image.shape[1] - textsize[0]) // 2
        textY = 350
        cv.putText(blank, result_message, (textX, textY), font, 2, result_color, 1, cv.LINE_AA)

        if number_guessed != number_to_guess:
            textsize = cv.getTextSize(your_answer_message, font, 1, 1)[0]
            textX = (image.shape[1] - textsize[0]) // 2
            textY += textsize[1] + 20
            cv.putText(blank, your_answer_message, (textX, textY), font, 1, color, 1, cv.LINE_AA)

            textsize = cv.getTextSize(correct_answer_message, font, 1, 1)[0]
            textX = (image.shape[1] - textsize[0]) // 2
            textY += textsize[1] + 20
            cv.putText(blank, correct_answer_message, (textX, textY), font, 1, color, 1, cv.LINE_AA)

        replay_message = "Press 'r' to replay or 'q' to quit"
        textsize = cv.getTextSize(replay_message, font, 1, 1)[0]
        textX = (image.shape[1] - textsize[0]) // 2
        textY += textsize[1] + 20
        cv.putText(blank, replay_message, (textX, textY), font, 1, color, 1, cv.LINE_AA)

        cv.imshow("The Memory Game", blank)

        key = cv.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            continue

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
