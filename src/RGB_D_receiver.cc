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

RGBDReceive::RGBDReceive(): it_(nh_) {
  cam_depth_sub_ = it_.subscribe("camera/depth/image",
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
  cv::Mat *image = new cv::Mat(cv_depth->image.size(), cv_depth->image.type());
  cv_depth->image.copyTo(*image);
  guint16 *ptr = reinterpret_cast<guint16*>(image->data);
  TrackSkel(ptr, msg->width, msg->height);
}

void RGBDReceive::OnImage(const sensor_msgs::ImageConstPtr &msg) {
  try {
    cv_image = cv_bridge::toCvShare(msg);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV_BRIDGE  exception: %s", e.what());
    return;
  }
#ifdef DEBUG
  // cv::imshow(WINDOW, cv_image->image);
  // cv::waitKey(30);
#endif
  RGBImage(&(cv_image->image));
}

void RGBDReceive::TrackSkel(guint16 *data, guint width, guint height) {}

void RGBDReceive::RGBImage(const cv::Mat *image) {}
}  // namespace sk_track
