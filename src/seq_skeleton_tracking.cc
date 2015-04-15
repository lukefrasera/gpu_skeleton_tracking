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
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <glib-object.h>
#include <sensor_msgs/image_encodings.h>
#include <skeltrack.h>
#include <math.h>
#include "opencv2/opencv.hpp"

namespace enc = sensor_msgs::image_encodings;
static char WINDOW[] = "Depth Image";


class SkeletonTrack {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber cam_depth_sub_;
  image_transport::Subscriber cam_image_sub_;
  cv_bridge::CvImageConstPtr cv_depth, cv_image;

 public:
  SkeletonTrack(): it_(nh_) {
    cam_depth_sub_ = it_.subscribe("camera/depth_registered/image_raw",
      5, &SkeletonTrack::OnDepth, this);
    cam_image_sub_ = it_.subscribe("camera/rgb/image_raw",
      5, &SkeletonTrack::OnImage, this);
    cv::namedWindow(WINDOW);
    printf("Constructed\n");
  }
  ~SkeletonTrack() {
    cv::destroyWindow(WINDOW);
  }

  void OnDepth(const sensor_msgs::ImageConstPtr &msg) {
    try {
      cv_depth = cv_bridge::toCvShare(msg);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("CV_BRIDGE  exception: %s", e.what());
      return;
    }

    // Show Depth imag to window
    cv::imshow(WINDOW, cv_depth->image);
    cv::waitKey(3);
  }

  void OnImage(const sensor_msgs::ImageConstPtr &msg) {
    try {
      cv_image = cv_bridge::toCvShare(msg);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("CV_BRIDGE  exception: %s", e.what());
      return;
    }
  }
};
int main(int argc, char **argv) {
  ros::init(argc, argv, "seq_skeleton");
  printf("SkeletonTracking\n");
  SkeletonTrack track;
  ros::spin();
  return 0;
}
