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
#ifndef RGBDReceive_H_
#define RGBDReceive_H_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <stdint.h>
#include <cv_bridge/cv_bridge.h>
#include <skeltrack.h>
#include <glib-object.h>
#include "opencv2/opencv.hpp"

namespace sk_track {
class RGBDReceive {
 public:
  RGBDReceive();
  virtual ~RGBDReceive();

  void OnDepth(const sensor_msgs::ImageConstPtr &msg);
  void OnImage(const sensor_msgs::ImageConstPtr &msg);

  virtual void TrackSkel(guint16 *data, guint width, guint height);
  virtual void RGBImage(const cv::Mat *image);

 protected:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber cam_depth_sub_;
  image_transport::Subscriber cam_image_sub_;
  cv_bridge::CvImageConstPtr cv_depth, cv_image;
};
static char WINDOW[] = "RGB Image";
}  // namespace sk_track
#endif
