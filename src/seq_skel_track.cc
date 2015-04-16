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
void SeqSkelTrack::TrackSkel(guint16 *data, guint width, guint height) {
  BufferInfo *userinfo = new BufferInfo;
  userinfo->image_data = data;
  userinfo->width = width;
  userinfo->height = height;
  GError  *error;
  // skeltrack_skeleton_track_joints(
  //   skeleton,
  //   data,
  //   width,
  //   height,
  //   NULL,
  //   OnTrack,
  //   userinfo);
  printf("Before Track\n");
  skeltrack_skeleton_track_joints_sync(
    skeleton,
    data,
    width,
    height,
    NULL,
    &error);
  printf("After Track\n");
  delete [] userinfo->image_data;
  delete userinfo;
}

void SeqSkelTrack::GetJoints() {
  printf("Hello getting joints\n");
}

void SeqSkelTrack::RGBImage(const cv::Mat *image) {}
}  // namespace sk_track
