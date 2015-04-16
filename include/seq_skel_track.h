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
#ifndef SEQ_SKEL_TRACK_H_
#define SEQ_SKEL_TRACK_H_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <stdint.h>
#include <skeltrack.h>
#include <glib-object.h>
#include "RGB_D_receiver.h"
#include "opencv2/opencv.hpp"

namespace sk_track {
class SeqSkelTrack : public RGBDReceive {
 public:
  SeqSkelTrack();
  virtual ~SeqSkelTrack();

  virtual void TrackSkel(cv_bridge::CvImageConstPtr data, guint width, guint height);
  void GetJoints();
  virtual void RGBImage(const cv::Mat *image);
 protected:
  SkeltrackSkeleton * skeleton;
};
}  // namespace sk_track
#endif
