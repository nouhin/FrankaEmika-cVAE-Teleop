#!/usr/bin/env python

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import tf
import numpy
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from control_msgs.msg import JointTrajectoryControllerState
import sys
import termios
import tty
import numpy as np
from moveit_msgs.msg import Grasp

from select import select


def all_close(goal, actual, tolerance=0.01):
	"""
	Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
	@param: goal       A list of floats, a Pose or a PoseStamped
	@param: actual     A list of floats, a Pose or a PoseStamped
	@param: tolerance  A float
	@returns: bool
	"""
	if type(goal) is list:
			for index in range(len(goal)):
					if abs(actual[index] - goal[index]) > tolerance:
							return False

	elif type(goal) is geometry_msgs.msg.PoseStamped:
			return all_close(goal.pose, actual.pose, tolerance)

	elif type(goal) is geometry_msgs.msg.Pose:
			return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

	return True

def callback(data):
	global action_array
	global state_array
	global state_action_array

	position = np.array(data.actual.positions)
	action   = np.array(data.actual.velocities)

	both_arrays = np.append(position, action)

	state_array.append(position)
	action_array.append(action)
	state_action_array.append(both_arrays)

def getch(timeout=0.01):

	if not sys.stdin.isatty():
		return sys.stdin.read(1)
	fileno = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fileno)
	ch = None
	try:
		tty.setraw(fileno)
		rlist = [fileno]
		if timeout >= 0:
			[rlist, _, _] = select(rlist, [], [], timeout)
		if fileno in rlist:
			ch = sys.stdin.read(1)
	except Exception as ex:
		print("getch", ex)
		raise OSError
	finally:
		termios.tcsetattr(fileno, termios.TCSADRAIN, old_settings)
	return ch


			
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('panda_teleop_node', anonymous=False)

group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

hand_group = moveit_commander.MoveGroupCommander("panda_hand")
# We can get the joint values from the group and adjust some of the values:
joint_goal = move_group.get_current_joint_values()
joint_goal[0] = -0.074
joint_goal[1] =  0.077
joint_goal[2] =  0.021
joint_goal[3] = -1.6
joint_goal[4] = 0
joint_goal[5] = 1.71
joint_goal[6] = -0.84

### Getting the robot to inti pose ###
while not(all_close(joint_goal, move_group.get_current_joint_values(), tolerance=0.01)):
	# The go command can be called with joint values, poses, or without any
	# parameters if you have already set the pose or joint target for the group
	move_group.go(joint_goal, wait=True)




exp_save = input('[INFO] Please enter [y/n] to save teleop states and actions: ')
if exp_save == 'y':
	exp_name = input('[INFO] Please enter the name of the teleop exp name : ')

	rospy.Subscriber("/panda_arm_controller/state", JointTrajectoryControllerState, callback, queue_size=1)

action_array        = []
state_array         = []
state_action_array  = []

done = False
while not done and not rospy.is_shutdown():
	old_pose = move_group.get_current_pose().pose
	new_pose = geometry_msgs.msg.Pose()
	x_diff, y_diff, z_diff, yaw, pitch, roll = 0,0,0,0,0,0
	new_cmd = False
	c = getch()
	if c:
		#catch Esc or ctrl-c
		if c in ['\x1b', '\x03']:
			done = True
			#rospy.signal_shutdown("Example finished.")

		if c =='1': 
			x_diff = 0.1
			new_cmd = True

		if c =='a': 
			x_diff = -0.1
			new_cmd = True

		if c =='2': 
			y_diff = 0.1
			new_cmd = True

		if c =='z': 
			y_diff = -0.1
			new_cmd = True

		if c =='3': 
			z_diff = 0.1
			new_cmd = True

		if c =='e': 
			z_diff = -0.1
			new_cmd = True

		if c =='4': 
			yaw = 0.1
			new_cmd = True

		if c =='r': 
			yaw = -0.1
			new_cmd = True

		if c =='5': 
			pitch = 0.1
			new_cmd = True

		if c =='t': 
			pitch = -0.1
			new_cmd = True

		if c =='6': 
			roll = 0.1
			new_cmd = True

		if c =='y': 
			roll = -0.1
			new_cmd = True
		if new_cmd:
			local_move = numpy.array((x_diff, y_diff, z_diff, 1.0))
			q = numpy.array((old_pose.orientation.x,
							 old_pose.orientation.y,
							 old_pose.orientation.z,
							 old_pose.orientation.w))

			xyz_move = numpy.dot(tf.transformations.quaternion_matrix(q),local_move)
			new_pose.position.x = old_pose.position.x + x_diff #xyz_move[0]
			new_pose.position.y = old_pose.position.y + y_diff #xyz_move[1]
			new_pose.position.z = old_pose.position.z + z_diff #xyz_move[2]

			diff_q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
			new_q = tf.transformations.quaternion_multiply(q, diff_q)

			new_pose.orientation.x = new_q[0]
			new_pose.orientation.y = new_q[1]
			new_pose.orientation.z = new_q[2]
			new_pose.orientation.w = new_q[3]

			move_group.set_pose_target(new_pose)
			move_group.go(wait=False)
		#move_group.stop()

		#current_pose = self.move_group.get_current_pose().pose
		#all_close(pose_goal, current_pose, 0.01)

move_group.stop()
#exp_name = '1'
if exp_save == 'y':
	np.save('./saved_trajectoires/state_action_array_'+exp_name+'.npy', np.array(state_action_array))
	np.save('./saved_trajectoires/action_array_'+exp_name+'.npy', np.array(action_array))
	np.save('./saved_trajectoires/state_array_'+exp_name+'.npy', np.array(state_array))
