#!/usr/bin/env python

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
from std_msgs.msg import ColorRGBA
import numpy as np


rospy.init_node('marker_test')
marker_pub = rospy.Publisher('marker_test', Marker,queue_size=5)



def make_marker( marker_type,scale, r, g, b, a):
    # make a visualization marker array for the occupancy grid
    m = Marker()
    m.action = Marker.ADD
    m.header.frame_id = '/world'
    m.header.stamp = rospy.Time.now()
    m.ns = 'marker_test_%d' % marker_type
    m.id = 0
    m.type = marker_type
    m.pose.orientation.y = 0
    m.pose.orientation.w = 1
    m.scale = scale
    m.color.r = 1.0;
    m.color.g = 0.5;
    m.color.b = 0.2;
    m.color.a = 0.3;
    m.color.r = r;
    m.color.g = g;
    m.color.b = b;
    m.color.a = a;

    x=np.arange(-0.8,0.1,0.05)
    y=np.sin(x)
    m.points = [ Point(0.25*np.sin(-v*7)+0.5,v,0.49) for v in x]
    return m


while not rospy.is_shutdown():

    scale = Vector3(0.01,0.05,0.01)
    marker_pub.publish(make_marker(Marker.LINE_STRIP,   scale, 1, .5, .2, .3))
    rospy.sleep(1.0)