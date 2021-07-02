import cv2 as cv
import numpy as np
import time
import logging
import math

cap = cv.VideoCapture("race.avi")


def region_of_interest(edges, frame):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus on part of screen
    polygon = np.array([[
        (0, 55),                    # LB
        (width, 55),                # RB
        (width, height * 1 / 2),    #RO
        (0, height * 1 / 2)         #LO


    ]], np.int32)

    cv.fillPoly(mask, polygon, 255)
    cropped_edges = cv.bitwise_and(edges, mask)
    res = cv.bitwise_and(frame, frame, mask=mask)
    return cropped_edges, res


def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 20

    line_segments = cv.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                   np.array([]), minLineLength=10, maxLineGap=40)

    return line_segments


def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("no line segments detected")
        return lane_lines

    height, width, extra = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 2
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
        AngleLinks = GetAngle(make_points(frame, left_fit_average))
        #print(f"links: {AngleLinks}")

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
        AngleRechts = GetAngle(make_points(frame, right_fit_average))
        #print(f"rechts: {AngleRechts}")

    return lane_lines


def make_points(frame, line):
    height, width, extra = frame.shape
    slope, intercept = line

    y1 = height
    y2 = int(y1 / 4)

    if slope == 0:
        slope = 0.1

    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))

    return [[x1, y1, x2, y2]]


def display_lines(frame, lines, line_color=(0, 255, 255), line_width=5):
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def GetAngle(line):
    x1, y1, x2, y2 = line[0]
    deltaX = x2 - x1
    deltaY = y2 - y1
    angle = math.atan2(deltaY, deltaX) * 180 / math.pi
    return angle


def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape

    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg

    print(steering_angle)
    return steering_angle


def display_heading_line(frame, steering_angle):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = (steering_angle + 90) / 180.0 * math.pi

    #van onderaan midden scherm
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv.line(heading_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    heading_image = cv.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


MovingAverageWaardes = np.zeros(5)  # array [0, 0, 0, 0, 0}

while True:

    succes, frame = cap.read()
    #cv.imshow("Video", frame)
    try:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    except:
        print("End of video reached")
        break

    #hsv hue saturation value
    lower_red = np.array([0, 80, 50])
    upper_red = np.array([6, 255, 220])
    mask0 = cv.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([172, 80, 50])
    upper_red = np.array([180, 255, 220])
    mask1 = cv.inRange(hsv, lower_red, upper_red)
    mask = mask0 + mask1


    res = cv.bitwise_and(frame, frame, mask = mask)

    blur  = cv.GaussianBlur(res, (15, 15), 0)
    median = cv.medianBlur(res, 5)

    iets = cv.erode(mask, None, iterations=2)
    iets = cv.dilate(iets, None, iterations=2)


    edges = cv.Canny(iets, 200, 400)

    roi, frame_roi = region_of_interest(edges, frame)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    heading_image = display_heading_line(lane_lines_image, steering_angle)
    image = display_lines(frame, line_segments)

    MovingAverageWaardes = np.insert(MovingAverageWaardes, 0, steering_angle, axis=0)
    MovingAverageWaardes = np.delete(MovingAverageWaardes, 5, axis=0)
    print(MovingAverageWaardes)

    cv.imshow('roi', frame_roi)
    cv.imshow('lines', image)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    cv.imshow('blur', blur)
    cv.imshow('median', median)
    cv.imshow('iets', iets)
    cv.imshow('edges', edges)
    cv.imshow("average slope lines", lane_lines_image)
    cv.imshow("heading line", heading_image)

    time.sleep(0.03)

    if cv.waitKey(1) & 0xFF ==ord('q'):
        break

cv.destroyAllWindows()