"""
Detect.py program

Implementing a machine vision of a robot that
must recognize printable characters height, S, U
using cv2 library

Baum-Tech Team

Designed by: Simankov Artyom Vladislavovich
Date: 10/15/2019

The program selects a letter, localizes, and then classifies
"""

# !/usr/bin/env python
# license removed for brevity
# pylint: disable=no-member
# import rospy
# from std_msgs.msg import Int32
# import smbus
import cv2
import numpy as np


DEBUG = 1

DIR_PATH = 'templates/' 
MASK = 100

CAP = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
CAP.set(cv2.CAP_PROP_FPS, 15)
CAP.set(3, 256)
CAP.set(4, 256)

H_TEMPLATE = cv2.imread(DIR_PATH + 'H.png', 0)
S_TEMPLATE = cv2.imread(DIR_PATH + 'S.png', 0)
U_TEMPLATE = cv2.imread(DIR_PATH + 'U.png', 0)
TEST_IMG = cv2.imread(DIR_PATH + 'test.png')

MATCHING = 0.88

"""
# ros init

pub = rospy.Publisher('victim_detect', Int32, queue_size=10)
rospy.init_node('talker', anonymous=False)
rate = rospy.Rate(10) # 10hz
global msg
msg = Int32()
"""


def max_square(contours):
    """
    Helper function that simply discards excess garbage
    (finding the largest area of the figure).
    The input is an array of contours,
    which were obtained from cv2.findContours. at
    the output gives the coordinates, width and height of the
    largest figure in the image
"""
    square = 0
    index = 0
    for i, contour in enumerate(contours):
        x_coord, y_coord, width, height = cv2.boundingRect(contour)
        if width * height > square:
            square = width * height
            index = i
    x_coord, y_coord, width, height = cv2.boundingRect(contours[index])
    return x_coord, y_coord, width, height


def classification(frame):
    """
        Helper function that takes images with symbol
        and classify it.
        Output: letter
    """
    result = None

    # Counting coefficient of concurrency image and template
    h_match = cv2.matchTemplate(frame, H_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    s_match = cv2.matchTemplate(frame, S_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    u_match = cv2.matchTemplate(frame, U_TEMPLATE, cv2.TM_CCOEFF_NORMED)

    # Saving lists that have coefficient more that MATCHING
    h_match = np.where(h_match >= MATCHING)
    s_match = np.where(s_match >= MATCHING)
    u_match = np.where(u_match >= (MATCHING-0.1))

    # MATCHING H
    if len(list(zip(*h_match[::-1]))) > 0:
        result = 'H'
    # MATCHING S
    elif len(list(zip(*s_match[::-1]))) > 0:
        result = 'S'
    # MATCHING U
    elif len(list(zip(*u_match[::-1]))) > 0:
        result = 'U'
    return result


def make_template(path):
    """
    Making photo of template
    :param path: name of file (means, that photo would be saved in folder, where program exists)
    """
    while True:
        _, image = CAP.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)

        _, thresh = cv2.threshold(gray, thresh=MASK, maxval=255, type=cv2.THRESH_BINARY_INV)
        cv2.imshow("Threshold", thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            x_coord, y_coord, width, height = max_square(contours=contours)
            print x_coord, y_coord, width, height
            cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            cv2.rectangle(image,
                          (x_coord, y_coord),
                          (x_coord + width, y_coord + height),
                          (70, 0, 0),
                          4)
            cv2.imshow("image", image)

            # Taking letter image from original
            letter_crop = gray[y_coord:y_coord + height, x_coord:x_coord + width]
            biggest = max(width, height)
            letter_square = np.ones(shape=[biggest, biggest], dtype=np.uint8) * 255
            if width > height:
                y_pos = biggest // 2 - height // 2
                letter_square[y_pos:y_pos + height, 0:width] = letter_crop
            elif width < height:
                x_pos = biggest // 2 - width // 2
                letter_square[0:height, x_pos:x_pos + width] = letter_crop
            else:
                letter_square = letter_crop

            cv2.imshow('Letter', letter_square)
            letter_square = cv2.resize(letter_square, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('sample', letter_square)
            print 'contour here at', x_coord, y_coord, width, height)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                cv2.imwrite(path, letter_square)
                break
    CAP.release()
    cv2.destroyAllWindows()


def cam():
    """
        Main function at this module, that
        starts up on our robot, if he see any letter,
        returns it to ROS with message of matching template,
        and prints this letter in console line
    """
    h_detect_counter = 0
    s_detect_counter = 0
    u_detect_counter = 0
    while True:
        _, image = CAP.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, thresh=MASK, maxval=255, type=cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            x_coord, y_coord, width, height = max_square(contours=contours)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            cv2.rectangle(image,
                          (x_coord, y_coord),
                          (x_coord + width, y_coord + height),
                          (70, 0, 0),
                          4)

            # Taking letter image from original
            letter_crop = gray[y_coord:y_coord + height, x_coord:x_coord + width]
            biggest = max(width, height)
            letter_square = np.ones(shape=[biggest, biggest], dtype=np.uint8) * 255
            if width > height:
                y_pos = biggest // 2 - height // 2
                letter_square[y_pos:y_pos + height, 0:width] = letter_crop
            elif width < height:
                x_pos = biggest // 2 - width // 2
                letter_square[0:height, x_pos:x_pos + width] = letter_crop
            else:
                letter_square = letter_crop

            sample = cv2.resize(letter_square, (64, 64), interpolation=cv2.INTER_CUBIC)
            print 'contour here at', x_coord, y_coord, width, height
            letter = classification(sample)
            if letter is not None:
                if letter == 'H':
                    h_detect_counter += 1
                    if h_detect_counter == 7:
                        print 'H'
                        #  msg.data = 1
                        #  pub.publish(msg)
                        return 'H'
                elif letter == 'S':
                    s_detect_counter += 1
                    if s_detect_counter == 7:
                        print 'S'
                        #  msg.data = 1
                        #  pub.publish(msg)
                        return 'S'
                elif letter == 'U':
                    u_detect_counter += 1
                    if u_detect_counter == 7:
                        print 'U'
                        #  msg.data = 1
                        #  pub.publish(msg)
                        return 'U'

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


def debug_vi_template_matching():
    """
        Repeats behavior of main function cam(), excepting
        that fact, that he outputs a lot of debug info and
        didn't stops his work if he got a letter in screen
    """
    while True:
        _, image = CAP.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)

        _, thresh = cv2.threshold(gray, thresh=MASK, maxval=255, type=cv2.THRESH_BINARY_INV)
        cv2.imshow("Threshold", thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            x_coord, y_coord, width, height = max_square(contours=contours)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            cv2.rectangle(image,
                          (x_coord, y_coord),
                          (x_coord + width, y_coord + height),
                          (70, 0, 0),
                          4)
            cv2.imshow("image", image)

            # Taking letter image from original
            letter_crop = gray[y_coord:y_coord + height, x_coord:x_coord + width]
            biggest = max(width, height)
            letter_square = np.ones(shape=[biggest, biggest], dtype=np.uint8) * 255
            if width > height:
                y_pos = biggest // 2 - height // 2
                letter_square[y_pos:y_pos + height, 0:width] = letter_crop
            elif width < height:
                x_pos = biggest // 2 - width // 2
                letter_square[0:height, x_pos:x_pos + width] = letter_crop
            else:
                letter_square = letter_crop

            cv2.imshow('Letter', letter_square)
            sample = cv2.resize(letter_square, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('sample', sample)

            print classification(sample), 'contour here at', x_coord, y_coord, width, height

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    if DEBUG == 1:
        # try:
        # senorInit()
        cam()
        # except:  # rospy.ROSInterruptException:
        # pass
    elif DEBUG == 2:
        make_template(path=DIR_PATH + 'U.png')

    else:
        debug_vi_template_matching()
