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
#include <glib-object.h>
#include "gpu-nsssp.h"

namespace sk_track {

static gfloat SMOOTHING_FACTOR = 0.2;


#define FAR_CUTOFF 2.0
#define CLOSE_CUTOFF 0.9
struct BufferInfo {
  guint16 *image_data;
  gint width;
  gint height;
};
SeqSkelTrack::SeqSkelTrack() {
  skeleton = skeltrack_skeleton_new();
  vis_pub = nh_.advertise<visualization_msgs::MarkerArray>("skeleton_markers",
    0);
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

visualization_msgs::Marker JointToMarker(SkeltrackJoint *joint,
    unsigned int id) {
  visualization_msgs::Marker marker;

  marker.header.frame_id = "camera_link";
  marker.header.stamp = ros::Time();
  marker.ns = "behavior_network";
  marker.id = id;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = joint->screen_x/100.0;
  marker.pose.position.y = joint->screen_y/100.0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.5;
  marker.scale.y = 0.5;
  marker.scale.z = 0.5;
  marker.color.a = 1.0;  // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;

  return marker;
}
void SeqSkelTrack::TrackSkel(cv_bridge::CvImageConstPtr data, guint width,
    guint height) {

  clock_t c_start, c_end;
  cv::Mat thresh_image;
  cv::Mat convert_image(data->image.size(), CV_16U);
  cv::threshold(data->image, thresh_image, CLOSE_CUTOFF, 0, CV_THRESH_TOZERO);
  cv::threshold(thresh_image, thresh_image, FAR_CUTOFF, 0,
    CV_THRESH_TOZERO_INV);

  cv::Mat *image = new cv::Mat(cv::Size(160, 120), CV_16U);

  float *img_ptr = reinterpret_cast<float*>(thresh_image.data);
  uint16_t *conv_ptr = reinterpret_cast<uint16_t*>(convert_image.data);

  for (int i = 0; i < width * height; ++i) {
    conv_ptr[i] = img_ptr[i]*600;
  }
  // cv::Mat *image = new cv::Mat(data->image.size(), data->image.type());
  cv::resize(convert_image, *image, image->size());
  // cv::imshow(WINDOW, convert_image*50);
  // cv::waitKey(30);
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
#ifdef GPU
  Node *centroid = skeltrack_skeleton_track_joints_sync_part1(
    skeleton,
    ptr,
    image->size().width,
    image->size().height,
    NULL,
    &error);
  Graph_t adj_list = GetAdjList(skeleton);
  int extrema_vertex;
  // for (int i = 0; i < adj_list.size_v; ++i) {
  //   printf("V[%d] = %d\n", i, adj_list.vertices[i]);
  // }
  c_start = clock();
  Extremas(adj_list.vertices, adj_list.edges, adj_list.size_v, adj_list.size_e, &extrema_vertex, centroid->index);
  c_end = clock();
  printf("%f\n", 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC);
  SkeltrackJointList list = skeltrack_skeleton_track_joints_sync_part2(skeleton, centroid);
#else
  SkeltrackJointList list = skeltrack_skeleton_track_joints_sync(
    skeleton,
    ptr,
    image->size().width,
    image->size().height,
    NULL,
    &error);
#endif

  if (list != NULL) {
    // printf("Found Skeleton\n");
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
    visualization_msgs::MarkerArray marker_list;

    if (head) {
      // printf("Head\n");
      visualization_msgs::Marker marker = JointToMarker(head, 0);
      marker_list.markers.push_back(marker);
    }
    if (left_hand) {
      // printf("Left Hand\n");
      visualization_msgs::Marker marker = JointToMarker(left_hand, 1);
      marker_list.markers.push_back(marker);
    }
    if (right_hand) {
      // printf("right_hand\n");
      visualization_msgs::Marker marker = JointToMarker(right_hand, 2);
      marker_list.markers.push_back(marker);
    }
    if (left_shoulder) {
      // printf("left_shoulder\n");
      visualization_msgs::Marker marker = JointToMarker(left_shoulder, 3);
      marker_list.markers.push_back(marker);
    }
    if (right_shoulder) {
      // printf("right_shoulder\n");
      visualization_msgs::Marker marker = JointToMarker(right_shoulder, 4);
      marker_list.markers.push_back(marker);
    }
    if (left_elbow) {
      // printf("left_elbow\n");
     visualization_msgs::Marker marker  = JointToMarker(left_elbow, 5);
      marker_list.markers.push_back(marker);
    }
    if (right_elbow) {
      // printf("right_elbow\n");
      visualization_msgs::Marker marker = JointToMarker(right_elbow, 6);
      marker_list.markers.push_back(marker);
    }

    vis_pub.publish(marker_list);

    skeltrack_joint_list_free(list);
  }
  delete userinfo;
  delete image;
}

void SeqSkelTrack::GetJoints() {
  printf("Hello getting joints\n");
}

void SeqSkelTrack::RGBImage(const cv::Mat *image) {}
}  // namespace sk_track
