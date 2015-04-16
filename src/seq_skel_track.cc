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
#include "seq_skel_track.h"

namespace sk_track {

#define FAR_CUTOFF 2.0
#define CLOSE_CUTOFF 0.9
struct BufferInfo {
  guint16 *image_data;
  gint width;
  gint height;
};
SeqSkelTrack::SeqSkelTrack() {
  skeleton = skeltrack_skeleton_new();
}
SeqSkelTrack::~SeqSkelTrack() {}

static void OnTrack(GObject *obj, GAsyncResult *res, gpointer user_data) {
  printf("Received SkeletonTrack\n");
  SeqSkelTrack *ptr = reinterpret_cast<SeqSkelTrack*>(obj);
  ptr->GetJoints();
  BufferInfo *buffer = reinterpret_cast<BufferInfo*>(user_data);
  delete [] buffer->image_data;
  g_slice_free(BufferInfo, user_data);
}
void SeqSkelTrack::TrackSkel(cv_bridge::CvImageConstPtr data, guint width, guint height) {
  cv::Mat thresh_image;
  // float min, max;
  // float *img_ptr = reinterpret_cast<float*>(data->image.data);
  // min = max = img_ptr[0];
  // for (int i = 0; i < width * height; ++i) {
  //   if (min > img_ptr[i]) {min = img_ptr[i];}
  //   if (max < img_ptr[i]) {max = img_ptr[i];}
  // }
  // printf("Min: %f, Max:%f\n", min, max);
  cv::threshold(data->image, thresh_image, CLOSE_CUTOFF, 0, CV_THRESH_TOZERO);
  cv::threshold(thresh_image, thresh_image, FAR_CUTOFF, 0, CV_THRESH_TOZERO_INV);

  cv::Mat *image = new cv::Mat(cv::Size(40,30), data->image.type());
  cv::imshow(WINDOW, thresh_image);
  cv::waitKey(30);
  cv::resize(thresh_image, *image, image->size());
  *image = *image * 500;
  image->convertTo(*image, CV_16U);
  // uint16_t *img_ptr = reinterpret_cast<uint16_t*>(image->data);
  // uint16_t min, max;
  // min = max = img_ptr[0];
  // for (int i = 0; i < 40*30; ++i) {
  //   if (min > img_ptr[i]) {min = img_ptr[i];}
  //   if (max < img_ptr[i]) {max = img_ptr[i];}
  // }
  // printf("Min: %u, Max:%u\n", min, max);
  // data->image.copyTo(*image);
  guint16 *ptr = reinterpret_cast<guint16*>(image->data);

  BufferInfo *userinfo = new BufferInfo;
  userinfo->image_data = ptr;
  userinfo->width = image->size().width;
  userinfo->height = image->size().height;
  GError  *error;
  // skeltrack_skeleton_track_joints(
  //   skeleton,
  //   data,
  //   width,
  //   height,
  //   NULL,
  //   OnTrack,
  //   userinfo);
  SkeltrackJointList list = skeltrack_skeleton_track_joints_sync(
    skeleton,
    ptr,
    image->size().width,
    image->size().height,
    NULL,
    &error);
  if(list != NULL) {
    printf("Found Skeleton\n");
    SkeltrackJoint *head = skeltrack_joint_list_get_joint(list,
                                           SKELTRACK_JOINT_ID_HEAD);
    SkeltrackJoint *left_hand = skeltrack_joint_list_get_joint(list,
                                                SKELTRACK_JOINT_ID_LEFT_HAND);
    SkeltrackJoint *right_hand = skeltrack_joint_list_get_joint(list,
                                                 SKELTRACK_JOINT_ID_RIGHT_HAND);
    SkeltrackJoint *left_shoulder = skeltrack_joint_list_get_joint(list,
                                         SKELTRACK_JOINT_ID_LEFT_SHOULDER);
    SkeltrackJoint *right_shoulder = skeltrack_joint_list_get_joint(list,
                                         SKELTRACK_JOINT_ID_RIGHT_SHOULDER);
    SkeltrackJoint *left_elbow = skeltrack_joint_list_get_joint(list,
                                                 SKELTRACK_JOINT_ID_LEFT_ELBOW);
    SkeltrackJoint *right_elbow = skeltrack_joint_list_get_joint(list,
                                                  SKELTRACK_JOINT_ID_RIGHT_ELBOW);
    skeltrack_joint_list_free(list);
  } else {printf("Not Found\n");}
  delete userinfo;
}

void SeqSkelTrack::GetJoints() {
  printf("Hello getting joints\n");
}

void SeqSkelTrack::RGBImage(const cv::Mat *image) {}
}  // namespace sk_track
