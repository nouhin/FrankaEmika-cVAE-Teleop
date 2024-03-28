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

import torch
import torch.nn as nn
import torch.nn.functional as F

from select import select

device = 'cpu'

# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 1000)
		self.e2 = nn.Linear(1000, 500)

		self.mean = nn.Linear(500, latent_dim)
		self.log_std = nn.Linear(500, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 500)
		self.d2 = nn.Linear(500, 1000)
		self.d3 = nn.Linear(1000, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim

	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], dim=1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)

		log_std = self.log_std(z)#.clamp(-4, 15)

		std = torch.exp(log_std)

		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z, clip=1)

		return u, mean, std

	def decode(self, state, z, clip=None):

		if clip is not None: z = z.clamp(-clip, clip)

		a = F.relu(self.d1(torch.cat((state, z), dim=1)))
		a = F.relu(self.d2(a))
		a = self.d3(a)
		return self.max_action*torch.tanh(a)
	
def predict_action(vae, state, z, denorm_factor):
	vae.eval()
	phase = 'predict'
	with torch.set_grad_enabled(phase == 'train'):
		norm_action = vae.decode(state, z)
	return denorm_factor*norm_action

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

latent_dim = 1
vae = VAE(state_dim=7, action_dim=7, latent_dim=latent_dim, max_action=1.)
vae.load_state_dict(torch.load('./cvae_model/1_sin_cvae.pth', map_location=device))

group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)


# We can get the joint values from the group and adjust some of the values:
joint_goal = move_group.get_current_joint_values()
joint_goal[0] = -0.074
joint_goal[1] =  0.077
joint_goal[2] =  0.021
joint_goal[3] = -1.6
joint_goal[4] = 0
joint_goal[5] = 1.71
joint_goal[6] = -0.84

"""
joint_goal[0] = 0
joint_goal[1] = -pi/4
joint_goal[2] = 0
joint_goal[3] = -pi/2
joint_goal[4] = 0
joint_goal[5] = pi/3
joint_goal[6] = 0

"""

### Getting the robot to inti pose ###
while not(all_close(joint_goal, move_group.get_current_joint_values(), tolerance=0.01)):
	# The go command can be called with joint values, poses, or without any
	# parameters if you have already set the pose or joint target for the group
	move_group.go(joint_goal, wait=True)

exp_name = input('[INFO] Please enter start the teleoperation using cVAE ...')

action_array        = []
state_array         = []
state_action_array  = []

done = False
z_first_indx = 0
z_second_indx = 0

while not done and not rospy.is_shutdown():
	new_input = False
	old_state = move_group.get_current_joint_values()
	c = getch()
	if c:
		#catch Esc or ctrl-c
		if c in ['\x1b', '\x03']:
			done = True
			#rospy.signal_shutdown("Example finished.")

		if c =='a': 
			if z_first_indx  <=  1: 
				z_first_indx = 0.3
				new_input = True
		if c =='q':
			if z_first_indx  >= -1: 
				z_first_indx = -0.3
				new_input = True

		if c =='z':
			if z_second_indx <=  1: 
				z_second_indx = 0.3
				new_input = True

		if c =='s':
			if z_second_indx >= -1: 
				z_second_indx = -0.3
				new_input = True

		if new_input :
			state = np.expand_dims([old_state[0]/10, old_state[1]/10, old_state[2]/10, old_state[3]/10,
				old_state[4]/10, old_state[5]/10, old_state[6]/10], axis=0)

			z = np.expand_dims([z_first_indx], axis=0)#, z_second_indx], axis=0)

			action = predict_action(vae, torch.Tensor(state),  torch.Tensor(z), 10.)
			action = np.array(action[0])

			old_state[0] += float(action[0])
			old_state[1] += float(action[1])
			old_state[2] += float(action[2])
			old_state[3] += float(action[3])
			old_state[4] += float(action[4])
			old_state[5] += float(action[5])
			old_state[6] += float(action[6])

			move_group.go(old_state, wait=False)

move_group.stop()
