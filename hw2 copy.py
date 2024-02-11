import math
import statistics
import random
import time
from typing import Tuple, Union, Sequence

import cv2 as cv
import numpy as np
from cv2 import UMat
from cv2 import VideoCapture


def calculate_angle(start, far, end):
    a = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    b = np.sqrt((start[0] - far[0]) ** 2 + (start[1] - far[1]) ** 2)
    c = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    return np.degrees(angle)


def bounding_box_from_contour(contour):
    x_axis = contour[:, 0, 0]
    y_axis = contour[:, 0, 1]

    x1, x2 = min(x_axis), max(x_axis)
    y1, y2 = min(y_axis), max(y_axis)

    return x1, x2, y1, y2


def hull_finger_counter(img: Union[UMat, np.ndarray], display_visual: bool) -> int:
    img = img.copy()
    # Find contours
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    # Get the largest contour
    contour = max(contours, key=cv.contourArea)

    center_x, center_y = find_centroid(img)

    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # draw centroid todo: x,y flipped fix later
    img = cv.drawMarker(img, (int(center_y), int(center_x)), color=(0, 255, 0), markerType=cv.MARKER_CROSS,
                        markerSize=10)

    # delete the bottom 20% of the image obj
    x1, y1, w, h = cv.boundingRect(contour)

    # Find the convex hull
    hull = cv.convexHull(contour, returnPoints=True)
    # Draw the hull
    cv.drawContours(img, [hull], -1, (0, 255, 0), 3)

    dist_to_edges = []
    for p in hull:
        img = cv.drawMarker(img, (int(p[0][0]), int(p[0][1])), color=(255, 255, 0), markerType=cv.MARKER_CROSS,
                            markerSize=10)
        edge_point = p[0]
        dist = calc_dist_between_points(edge_point, (center_x, center_y))
        dist_to_edges += [(dist, edge_point)]

    r: int = round((h / 2) - (h / 6))

    bin_img = img.copy()
    bin_img = cv.cvtColor(bin_img, cv.COLOR_RGB2GRAY)

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
        plt_show_img("Hull", img)

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


def plt_show_img(name: str, img: UMat):
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, img)
    cv.waitKey(1)


def rescaleFrame(frame: Union[UMat, np.ndarray], scale) -> UMat:
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def setResLiveVideo(webcam: VideoCapture, width: int):
    # given the width we automatically figure out the height
    scale = width / webcam.get(3)
    height = int(webcam.get(4) * scale)
    webcam.set(3, width)
    webcam.set(4, height)
    # reduce frame rate
    webcam.set(cv.CAP_PROP_FPS, 20)


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
    # Convert the image to HSV and YCrCb color spaces
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # Calculate the mean and standard deviation of each color channel
    mean_hsv, std_hsv = cv.meanStdDev(hsv)
    mean_ycrcb, std_ycrcb = cv.meanStdDev(ycrcb)

    # Define the thresholds for HSV color space
    lower_hsv = np.maximum(mean_hsv - 2 * std_hsv, 0)
    upper_hsv = np.minimum(mean_hsv + 2 * std_hsv, 255)

    # Define the thresholds for YCrCb color space
    lower_ycrcb = np.maximum(mean_ycrcb - 2 * std_ycrcb, 0)
    upper_ycrcb = np.minimum(mean_ycrcb + 2 * std_ycrcb, 255)

    # Create masks for each color space
    mask_hsv = cv.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine the masks
    mask_combined = cv.bitwise_and(mask_hsv, mask_ycrcb)

    # Invert the mask to work with white background
    mask_combined = cv.bitwise_not(mask_combined)

    # Apply the combined mask to the original frame
    img = cv.bitwise_and(img, img, mask=mask_combined)

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

def count_fingers():
    global count, fingers_detected
    
    # Connect to webcam
    # print("Connecting to webcam...")
    webcam = cv.VideoCapture(0, cv.CAP_DSHOW)
    setResLiveVideo(webcam, 500)

    # print("Starting display..!")
    finger_detections = []
    while True:
        status, frame = webcam.read()
        if not status:
            print('Failed to capture frame')
            return None

        rescaleFrame(frame, 0.6)

        original = frame.copy()

        # Apply blurring to remove noise
        frame = apply_smoothing(frame)

        # Mask to find skin-colored objects
        frame = mask_image(frame)

        # Convert to a binary image
        frame = convert_to_binary(frame)

        # Dilate before extracting the largest object
        frame = cv.dilate(frame, np.array([9, 9]), iterations=4)

        # Make contours around objects and extract the contour with the largest area
        frame, contour, region_of_interest = binary_img_extract_largest_obj(frame)

        # If no object is detected, continue
        if frame is None:
            continue

        frame, contour = angle_contour_reducer(frame, contour)
        frame, contour = defects_remover_via_angle_checking(frame, contour)
        frame = cv.dilate(frame, np.array([11, 11]), iterations=7)
        frame, contour, region_of_interest = binary_img_extract_largest_obj(frame)

        # Now scale the object in preparation for when we rotate it
        frame, region_of_interest = scale_obj(frame, region_of_interest)

        # Find the axis of least inertia
        area, centroid, theta = find_axis_of_least_inertia(frame, False)

        # Translate the centroid of the object to the center of the image
        frame, region_of_interest = move_obj_to_center(frame, centroid, region_of_interest)

        # Rotate the image based on the axis of least inertia
        frame = rotate_at_center(frame, theta)

        if area < 1500:
            # Add cv text to move closer
            cv.putText(original, f'Please move closer', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            continue

        fingers = hull_finger_counter(frame, False)
        roundness = calc_roundness(frame)
        fingers = 5 if (roundness > 0.5 and (fingers >= 4)) else fingers
        fingers = min(fingers, 5)

        if len(finger_detections) > 10 and statistics.stdev(finger_detections) < 0.65:
            cv.putText(original, f'Fingers: {int(statistics.mean(finger_detections))}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        elif len(finger_detections) > 10:
            cv.putText(original, f'Calculating...', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Display the final frame with finger count information
        plt_show_img("Enter your guess", original)

        if len(finger_detections) > 10 and statistics.stdev(finger_detections) < 0.8:
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

            if count == 30:
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

        number_to_guess = ""
        for _ in range(5):
            number_to_guess += random.choice("1234")

        guess_message = "Here is the number to remember: " + number_to_guess
        textsize = cv.getTextSize(guess_message, font, 1, 1)[0]
        textX = (image.shape[1] - textsize[0]) // 2
        textY = 350

        cv.putText(image, guess_message, (textX, textY), font, 1, color, 1, cv.LINE_AA)
        cv.imshow("The Memory Game", image)
        
        for i in range(7, 0, -1):
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

        for i in range(5):
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
