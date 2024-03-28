#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from control_msgs.msg import JointTrajectoryControllerState
from visualization_msgs.msg import InteractiveMarkerInit
import numpy as np

def callback_jt(data):

	#print(data.actual.positions)
	#print(np.array(data.actual.velocities))
	#msg = data
	pos = data.position
	data.position = (pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],pos[7],1.)
	pub_jt.publish(data)

def callback(data):

	#print(data.actual.positions)
	#print(np.array(data.actual.velocities))
	#msg = data
	pos = data.desired.positions
	data.desired.positions = (pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],3.0)
	#pub.publish(data)

	#rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.header.stamp)

def listener():

	# In ROS, nodes are uniquely named. If two nodes with the same
	# name are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rospy.init_node('testing', anonymous=True)
	
	#rospy.Subscriber("/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update_full", InteractiveMarkerInit, callback, queue_size=1)

	rospy.Subscriber("/panda_arm_controller/state", JointTrajectoryControllerState, callback, queue_size=1)

	rospy.Subscriber("/joint_states", JointState, callback_jt, queue_size=1)
	
	#rate = rospy.Rate(25) # 10hz
	#while not rospy.is_shutdown():
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	pub = rospy.Publisher('/panda_arm_controller/state', JointTrajectoryControllerState, queue_size=10)
	pub_jt = rospy.Publisher('/joint_states', JointState, queue_size=10)
	print("starting...")
	listener()
