# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to add and delete user nogo regions"""
import argparse
import math
import json
import sys
import time

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import robot_command_pb2, world_object_pb2
from bosdyn.api.geometry_pb2 import Vec2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import (WorldObjectClient, make_add_world_object_req,
                                        make_delete_world_object_req, send_add_mutation_requests,
                                        send_delete_mutation_requests)
from bosdyn.util import now_timestamp, seconds_to_duration

# Mobility command end time parameter.
_SECONDS_FULL = 10

# Dimensions of the box obstacle.  Frame for this obstacle is defined by the user when making the
# world object proto.
BOX_LEN_X = 1.0
BOX_LEN_Y_LONG = 1.0


def set_and_test_user_obstacles(config, json_obstacle_map):
    """A simple example of using the Boston Dynamics internal API to set user-defined boxes that
    represent body and/or foot obstacles.

    Please be aware that this demo causes the robot to walk at fake obstacles, then through the
    obstacles later to test that all have been successfully cleared.

    The robot requires about 2m of open space in front of it to complete this example."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('UserNoGoClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    # The user-defined no-go regions are in the WorldObject proto; get the world object client.
    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=20)
        robot.logger.info('Robot standing.')

        # Get robot pose in vision frame from robot state.  We will define some obstacles relative
        # to this snapshot of the robot body frame at startup.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        obstacles = []
        for obst in json_obstacle_map["obstacles"]:
            body_T_obs0 = math_helpers.SE3Pose(x=obst['x'], y=obst['y'], z=0, rot=math_helpers.Quat())
            vis_T_obs0 = vision_T_body * body_T_obs0
            obs = create_body_obstacle_box('obstacle0', BOX_LEN_X, BOX_LEN_Y_LONG, VISION_FRAME_NAME,
                                            vis_T_obs0, 60*10)
            print("obstacles={}".format(obs))
            obstacles.append(obs)

        send_add_mutation_requests(world_object_client, obstacles)

        # Verify that we have correctly added a world object by requesting and printing the list.
        # Request the list of world objects, filtered so it only returns ones of type
        # WORLD_OBJECT_USER_NOGO.
        request_nogos = [world_object_pb2.WORLD_OBJECT_USER_NOGO]
        nogo_objects = world_object_client.list_world_objects(
            object_type=request_nogos).world_objects
        print_object_names_and_ids(nogo_objects, "List of user nogo regions after initial add:")

        for milestone in json_obstacle_map["milestones"]:
            x = milestone['x']
            y = milestone['y']
            cmd1, traj_time = create_mobility_goto_command(x, y, vision_T_body)
            robot.logger.info('Sending  body trajectory command to ({},{}).'.format(x, y))
            command_client.robot_command(cmd1, end_time_secs=time.time() + 10)

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')


# Smaller print for objects
def print_object_names_and_ids(world_object_list, optional_str="World objects: "):
    print(optional_str + " [")
    for obj in world_object_list:
        print('\t' + obj.name + ' (' + str(obj.id) + ') ')
    print(']\n')


# Create the body obstacle box world object.  Does not fill out any transform_snapshot info, since
# the Box2WithFrame proto already has fields for the frame information locally.
def create_body_obstacle_box(obsname, x_span, y_span, frame_name, frame_T_box, lifetime_sec,
                             disable_foot=False, disable_body=False, disable_foot_inflate=False):
    time_now = now_timestamp()
    obs_lifetime = seconds_to_duration(lifetime_sec)
    obs = world_object_pb2.WorldObject(name=obsname, acquisition_time=time_now,
                                       object_lifetime=obs_lifetime)
    obs.nogo_region_properties.disable_foot_obstacle_generation = disable_foot
    obs.nogo_region_properties.disable_body_obstacle_generation = disable_body
    obs.nogo_region_properties.disable_foot_obstacle_inflation = disable_foot_inflate
    obs.nogo_region_properties.box.frame_name = frame_name
    obs.nogo_region_properties.box.box.size.CopyFrom(Vec2(x=x_span, y=y_span))
    obs.nogo_region_properties.box.frame_name_tform_box.CopyFrom(frame_T_box.to_proto())
    return obs


# Create a mobility command to send the body to a point [x,y] in the frame defined by the transform
# vision_T_frame.  Z is assumed zero when the transform is applied.
def create_mobility_goto_command(x_rt_frame, y_rt_frame, vision_T_frame):
    frame_name = VISION_FRAME_NAME
    command = robot_command_pb2.RobotCommand()
    command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = frame_name

    # Generate a point in front of the robot, on the far side of the series of obstacles we just defined.
    x_ewrt_v, y_ewrt_v, z_ewrt_v = vision_T_frame.transform_point(x_rt_frame, y_rt_frame, 0)
    point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
    point.pose.position.x = x_ewrt_v
    point.pose.position.y = y_ewrt_v
    point.pose.angle = vision_T_frame.rot.to_yaw()

    # Scale the trajectory time based on requested distance.
    traj_time = max([4, 1.5 * math.sqrt(x_rt_frame * x_rt_frame + y_rt_frame * y_rt_frame)])
    duration = seconds_to_duration(traj_time)
    point.time_since_reference.CopyFrom(duration)

    # Just return the command; don't send it yet.
    return command, traj_time


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--obstacle', default="obstacle_map1.json", type=str,
                        help='Path of obstacle map file')

    options = parser.parse_args(argv)
    try:
        obstacle_map = json.load(open(options.obstacle))
        set_and_test_user_obstacles(options, obstacle_map)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)