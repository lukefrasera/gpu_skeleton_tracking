/*
gpu_skeleton_tracking
Copyright (C) 2015  Luke Fraser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <ros/ros.h>
#include <stdio.h>
#include "seq_skel_track.h"
// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
// #include <glib-object.h>
// #include <sensor_msgs/image_encodings.h>
// #include <skeltrack.h>
// #include <math.h>
// #include "opencv2/opencv.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "seq_skeleton");
  printf("SkeletonTracking\n");
  sk_track::SeqSkelTrack track;
  ros::spin();
  return 0;
}
