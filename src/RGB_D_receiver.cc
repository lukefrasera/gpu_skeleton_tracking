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
#include <stdio.h>
#include "RGB_D_receiver.h"

namespace sk_track {
static char WINDOW[] = "RGB Image";

RGBDReceive::RGBDReceive(): it_(nh_) {
  cam_depth_sub_ = it_.subscribe("/camera/depth/image",
    5, &RGBDReceive::OnDepth, this);
  cam_image_sub_ = it_.subscribe("camera/rgb/image_raw",
    5, &RGBDReceive::OnImage, this);
  cv::namedWindow(WINDOW);
  printf("Constructed\n");
}
RGBDReceive::~RGBDReceive() {
  cv::destroyWindow(WINDOW);
}

void RGBDReceive::OnDepth(const sensor_msgs::ImageConstPtr &msg) {
  try {
    cv_depth = cv_bridge::toCvShare(msg);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV_BRIDGE  exception: %s", e.what());
    return;
  }
  // Call Depth map tracking function
  uint8_t *ptr = reinterpret_cast<uint8_t*>(cv_depth->image.data);
  TrackSkel(ptr);
}

void RGBDReceive::OnImage(const sensor_msgs::ImageConstPtr &msg) {
  try {
    cv_image = cv_bridge::toCvShare(msg);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV_BRIDGE  exception: %s", e.what());
    return;
  }
#ifdef Debug
  cv::imshow(WINDOW, cv_depth->image);
  cv::waitKey(3);
#endif
  RGBImage(&(cv_image->image));
}

void RGBDReceive::TrackSkel(const uint8_t *data) {
  printf("Hello I am tracking\n");
}

void RGBDReceive::RGBImage(const cv::Mat *image) {}
}  // namespace sk_track
