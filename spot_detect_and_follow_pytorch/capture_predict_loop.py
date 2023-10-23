# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use the Boston Dynamics API to detect and follow an object"""
import argparse
import io
import json
import math
import os
from glob import glob
import signal
import sys
import time
from multiprocessing import Barrier, Process, Queue, Value
from queue import Empty, Full
from threading import BrokenBarrierError, Thread

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from ultralytics import YOLO

import bosdyn.client
import bosdyn.client.util
from bosdyn import geometry
from bosdyn.api import geometry_pb2 as geo
from bosdyn.api import image_pb2, trajectory_pb2
from bosdyn.api.image_pb2 import ImageSource
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.frame_helpers import (GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body)
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_command import (CommandFailedError, CommandTimedOutError,
                                         RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
import object_detector as odapi


LOGGER = bosdyn.client.util.get_logger()

SHUTDOWN_FLAG = Value('i', 0)

# Don't let the queues get too backed up
QUEUE_MAXSIZE = 10

# This is a multiprocessing.Queue for communication between the main process and the
# Tensorflow processes.
# Entries in this queue are in the format:

# {
#     'source': Name of the camera,
#     'world_tform_cam': transform from VO to camera,
#     'world_tform_gpe':  transform from VO to ground plane,
#     'raw_image_time': Time when the image was collected,
#     'cv_image': The decoded image,
#     'visual_dims': (cols, rows),
#     'depth_image': depth image proto,
#     'system_cap_time': Time when the image was received by the main process,
#     'image_queued_time': Time when the image was done preprocessing and queued
# }
RAW_IMAGES_QUEUE = Queue(QUEUE_MAXSIZE)

# This is a multiprocessing.Queue for communication between the Tensorflow processes and
# the bbox print process. This is meant for running in a containerized environment with no access
# to an X display
# Entries in this queue have the following fields in addition to those in :
# {
#   'processed_image_start_time':  Time when the image was received by the TF process,
#   'processed_image_end_time':  Time when the image was processing for bounding boxes
#   'boxes': list of detected bounding boxes for the processed image
#   'classes': classes of objects,
#   'scores': confidence scores,
# }
PROCESSED_BOXES_QUEUE = Queue(QUEUE_MAXSIZE)

# Barrier for waiting on Tensorflow processes to start, initialized in main()
TENSORFLOW_PROCESS_BARRIER = None

COCO_CLASS_DICT = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'trafficlight',
    11: 'firehydrant',
    13: 'stopsign',
    14: 'parkingmeter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sportsball',
    38: 'kite',
    39: 'baseballbat',
    40: 'baseballglove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennisracket',
    44: 'bottle',
    46: 'wineglass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hotdog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'pottedplant',
    65: 'bed',
    67: 'diningtable',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cellphone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddybear',
    89: 'hairdrier',
    90: 'toothbrush'
}

# Mapping from visual to depth data
VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE = {
    'frontleft_fisheye_image': 'frontleft_depth_in_visual_frame',
    'frontright_fisheye_image': 'frontright_depth_in_visual_frame'
}
ROTATION_ANGLES = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}



class AsyncImage(AsyncPeriodicQuery):
    """Grab image."""

    def __init__(self, image_client, image_sources):
        # Period is set to be about 15 FPS
        super(AsyncImage, self).__init__('images', image_client, LOGGER, period_sec=0.067)
        self.image_sources = image_sources

    def _start_query(self):
        return self._client.get_image_from_sources_async(self.image_sources)


class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        # period is set to be about the same rate as detections on the CORE AI
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.02)

    def _start_query(self):
        return self._client.get_robot_state_async()


def capture_images(sleep_between_capture):
    """ Captures images and places them on the queue

    Args:
        image_task (AsyncImage): Async task that provides the images response to use
        sleep_between_capture (float): Time to sleep between each image capture
    """
    img_dir = "/home/abhay/code"
    images = glob(os.path.join(img_dir, "*.jpg"))
    print(images)
    entry = {}
    source = "camera"
    while not SHUTDOWN_FLAG.value:
        if len(images) > 0:
            image = images.pop()
            cv_image = cv2.imread(image)
            print("read cv2 image..")
            entry[source] = {
                'source': "camera",
                'cv_image': cv_image,
                'visual_dims': cv_image.shape,
                'image_queued_time': time.time()
            }
            try:
                RAW_IMAGES_QUEUE.put_nowait(entry)
            except Full as exc:
                print(f'RAW_IMAGES_QUEUE is full: {exc}')
        time.sleep(sleep_between_capture)


def start_pytorch_processes(num_processes, model_path, detection_class, detection_threshold,
                               max_processing_delay):
    """Starts Tensorflow processes in parallel.

    It does not keep track of the processes once they are started because they run indefinitely
    and are never joined back to the main process.

    Args:
        num_processes (int): Number of Tensorflow processes to start in parallel.
        model_path (str): Filepath to the Tensorflow model to use.
        detection_class (int): Detection class to detect
        detection_threshold (float): Detection threshold to apply to all Tensorflow detections.
        max_processing_delay (float): Allowed delay before processing an incoming image.
    """
    processes = []
    for _ in range(num_processes):
        process = Process(
            target=process_images, args=(
                model_path,
                detection_class,
                detection_threshold,
                max_processing_delay,
            ), daemon=True)
        process.start()
        processes.append(process)
    return processes


def process_images(model_path, detection_class, detection_threshold, max_processing_delay):
    """Starts Tensorflow and detects objects in the incoming images.

    Args:
        model_path (str): Filepath to the Tensorflow model to use.
        detection_class (int): Detection class to detect
        detection_threshold (float): Detection threshold to apply to all Tensorflow detections.
        max_processing_delay (float): Allowed delay before processing an incoming image.
    """

    model = YOLO("yolov8l.pt")
    num_processed_skips = 0

    print("Downloaded and initialsed model...")
    while not SHUTDOWN_FLAG.value:
        try:
            entry = RAW_IMAGES_QUEUE.get_nowait()
        except Empty:
            time.sleep(0.1)
            continue
        for _, capture in entry.items():
            start_time = time.time()
            image = capture['cv_image']
            boxes, scores, classes = odapi.predict(model, image)

            confident_boxes = []
            confident_object_classes = []
            confident_scores = []
            if len(boxes) == 0:
                print('no detections founds')
                continue
            for box, score, box_class in sorted(zip(boxes, scores, classes), key=lambda x: x[1],
                                                reverse=True):
                if score > detection_threshold:
                    confident_boxes.append(box)
                    confident_object_classes.append(model.model.names[box_class])
                    confident_scores.append(score)
                    box = list(map(int, box))
                    image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

            capture['processed_image_start_time'] = start_time
            capture['processed_image_end_time'] = time.time()
            capture['boxes'] = confident_boxes
            capture['classes'] = confident_object_classes
            capture['scores'] = confident_scores
            capture['cv_image'] = image
        try:
            PROCESSED_BOXES_QUEUE.put_nowait(entry)
        except Full as exc:
            print(f'PROCESSED_BOXES_QUEUE is full: {exc}')
    print('tf process ending')
    return True



def _find_highest_conf_source(processed_boxes_entry):
    highest_conf_source = None
    max_score = 0
    for key, capture in processed_boxes_entry.items():
        if 'scores' in capture.keys():
            if len(capture['scores']) > 0 and capture['scores'][0] > max_score:
                highest_conf_source = key
                max_score = capture['scores'][0]
    return highest_conf_source


def signal_handler(signal, frame):
    print('Interrupt caught, shutting down')
    SHUTDOWN_FLAG.value = 1


def main(argv):
    """Command line interface.

    Args:
        argv: List of command-line arguments passed to the program.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--number-tensorflow-processes', default=1, type=int,
                        help='Number of Tensorflow processes to run in parallel')
    parser.add_argument('--detection-threshold', default=0.1, type=float,
                        help='Detection threshold to use for Tensorflow detections')
    parser.add_argument(
        '--sleep-between-capture', default=5, type=float,
        help=('Seconds to sleep between each image capture loop iteration, which captures '
              'an image from all cameras'))
    parser.add_argument(
        '--detection-class', default=1, type=int,
        help=('Detection classes to use in the Tensorflow model.'
              'Default is to use 1, which is a person in the Coco dataset'))
    parser.add_argument(
        '--max-processing-delay', default=7.0, type=float,
        help=('Maximum allowed delay for processing an image. '
              'Any image older than this value will be skipped'))
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    signal.signal(signal.SIGINT, signal_handler)

    # Start Tensorflow processes
    tf_processes = start_pytorch_processes(options.number_tensorflow_processes,
                                              None, options.detection_class,
                                              options.detection_threshold,
                                              options.max_processing_delay)


    # This thread starts the async tasks for image and robot state retrieval
    # Start image capture process
    image_capture_thread = Process(target=capture_images,
                                   args=([options.sleep_between_capture]),
                                   daemon=True)
    image_capture_thread.start()
    while not SHUTDOWN_FLAG.value:
        # This comes from the tensorflow processes and limits the rate of this loop
        try:
            entry = PROCESSED_BOXES_QUEUE.get_nowait()
        except Empty:
            continue
        # find the highest confidence bounding box
        highest_conf_source = _find_highest_conf_source(entry)
        if highest_conf_source is None:
            # no boxes or scores found
            continue
        capture_to_use = entry[highest_conf_source]
        print("classesry={}, scores={}".format(entry["camera"]["classes"],
                                                entry["camera"]["scores"]))
    return True


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
