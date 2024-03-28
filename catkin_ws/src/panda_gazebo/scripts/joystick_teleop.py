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
from sensor_msgs.msg import Joy
import sys
import termios
import tty
import numpy as np

from select import select

class JoyStatus():
	def __init__(self):
		self.center = False
		self.select = False
		self.start = False
		self.L3 = False
		self.R3 = False
		self.square = False
		self.up = False
		self.down = False
		self.left = False
		self.right = False
		self.triangle = False
		self.cross = False
		self.circle = False
		self.L1 = False
		self.R1 = False
		self.L2 = False
		self.R2 = False
		self.left_analog_x = 0.0
		self.left_analog_y = 0.0
		self.right_analog_x = 0.0
		self.right_analog_y = 0.0

class XBoxStatus(JoyStatus):
	def __init__(self, msg):
		JoyStatus.__init__(self)
		if msg.buttons[8] == 1:
			self.center = True
		else:
			self.center = False
		if msg.buttons[6] == 1:
			self.select = True
		else:
			self.select = False
		if msg.buttons[7] == 1:
			self.start = True
		else:
			self.start = False
		if msg.buttons[9] == 1:
			self.L3 = True
		else:
			self.L3 = False
		if msg.buttons[10] == 1:
			self.R3 = True
		else:
			self.R3 = False
		if msg.buttons[2] == 1:
			self.square = True
		else:
			self.square = False
		if msg.buttons[1] == 1:
			self.circle = True
		else:
			self.circle = False
		if msg.axes[7] > 0.1:
			self.up = True
		else:
			self.up = False
		if msg.axes[7] < -0.1:
			self.down = True
		else:
			self.down = False
		if msg.axes[6] > 0.1:
			self.left = True
		else:
			self.left = False
		if msg.axes[6] < -0.1:
			self.right = True
		else:
			self.right = False
		if msg.buttons[3] == 1:
			self.triangle = True
		else:
			self.triangle = False
		if msg.buttons[0] == 1:
			self.cross = True
		else:
			self.cross = False
		if msg.buttons[4] == 1:
			self.L1 = True
		else:
			self.L1 = False
		if msg.buttons[5] == 1:
			self.R1 = True
		else:
			self.R1 = False
		if msg.axes[2] < -0.5:
			self.L2 = True
		else:
			self.L2 = False
		if msg.axes[5] < -0.5:
			self.R2 = True
		else:
			self.R2 = False
		self.left_analog_x = msg.axes[0]
		self.left_analog_y = msg.axes[1]
		self.right_analog_x = msg.axes[3]
		self.right_analog_y = msg.axes[4]
		self.orig_msg = msg

def signedSquare(val):
	if val > 0:	sign = 1
	else: sign = -1
	return val * val * sign

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
	global action_array, state_array, state_action_array, exp_save
	if exp_save == 'y':
		position = np.array(data.actual.positions)
		action   = np.array(data.actual.velocities)

		both_arrays = np.append(position, action)

		state_array.append(position)
		action_array.append(action)
		state_action_array.append(both_arrays)

def callback_joy(data):
	global x_diff, y_diff, z_diff, yaw, pitch, roll, publish
	x_diff, y_diff, z_diff, yaw, pitch, roll, publish = 0,0,0,0,0,0, False
	status = XBoxStatus(data)

	dist = status.left_analog_y * status.left_analog_y + status.left_analog_x * status.left_analog_x
	
	if abs(dist) > 0.1: 
		scale = 0.5
		x_diff = status.left_analog_y* scale
		y_diff = status.left_analog_x* scale
		publish = True


	if status.cross: 
		z_diff = 0.2
		publish = True
	elif status.triangle: 
		z_diff = -0.2
		publish = True


	if status.L1: 
		yaw = 0.1
		publish = True

	elif status.R1:	
		yaw = -0.1
		publish = True

	if status.up: 
		pitch = 0.1
		publish = True

	elif status.down: 
		pitch = -0.1
		publish = True

	if status.right: 
		roll = 0.1
		publish = True

	elif status.left: 
		roll = -0.1
		publish = True


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


# We can get the joint values from the group and adjust some of the values:
joint_goal = move_group.get_current_joint_values()
joint_goal[0] = 0
joint_goal[1] = -pi/4
joint_goal[2] = 0
joint_goal[3] = -pi/2
joint_goal[4] = 0
joint_goal[5] = pi/3
joint_goal[6] = 0

### Getting the robot to inti pose ###
while not(all_close(joint_goal, move_group.get_current_joint_values(), tolerance=0.01)):
	# The go command can be called with joint values, poses, or without any
	# parameters if you have already set the pose or joint target for the group
	move_group.go(joint_goal, wait=True)

exp_save = input('[INFO] Please enter [y/n] to save teleop states and actions: ')
if exp_save == 'y':
	exp_name = input('[INFO] Please enter the name of the teleop exp name : ')

rospy.Subscriber("/panda_arm_controller/state", JointTrajectoryControllerState, callback, queue_size=1)
rospy.Subscriber("/joy", Joy, callback_joy, queue_size=1)

action_array        = []
state_array         = []
state_action_array  = []
x_diff, y_diff, z_diff, yaw, pitch, roll, publish = 0,0,0,0,0,0, False
done = False
while not done and not rospy.is_shutdown():

	#c = getch()

	#catch Esc or ctrl-c
	#if c in ['\x1b', '\x03']:done = True
		#rospy.signal_shutdown("Example finished.")

	#print(publish)
	if publish:
		old_pose = move_group.get_current_pose().pose
		new_pose = geometry_msgs.msg.Pose()
		# move in local
		local_move = numpy.array((x_diff, y_diff, z_diff, 1.0))
		q = numpy.array((old_pose.orientation.x,
						 old_pose.orientation.y,
						 old_pose.orientation.z,
						 old_pose.orientation.w))

		xyz_move = numpy.dot(tf.transformations.quaternion_matrix(q),local_move)
		new_pose.position.x = old_pose.position.x + xyz_move[0]
		new_pose.position.y = old_pose.position.y + xyz_move[1]
		new_pose.position.z = old_pose.position.z + xyz_move[2]

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
