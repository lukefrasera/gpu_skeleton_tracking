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
SeqSkelTrack::SeqSkelTrack() {
  skeleton = skeltrack_skeleton_new();
}
SeqSkelTrack::~SeqSkelTrack() {
  delete skeleton;
}

void SeqSkelTrack::TrackSkel(guint16 *data, guint width, guint height) {
  skeltrack_skeleton_track_joints(skeleton, data,
    )
  printf("SEQ received depth frame\n");
}
void SeqSkelTrack::RGBImage(const cv::Mat *image) {
  printf("SEQ received image frame\n");
}
}  // namespace sk_track
