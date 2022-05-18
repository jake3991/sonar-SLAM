#pragma once
#include <bruce_msgs/ISAM2Update.h>

#include <gtsam/nonlinear/ISAM2.h>

namespace bruce_msgs
{
ISAM2UpdatePtr toMsg(const gtsam::ISAM2 &isam2, const gtsam::NonlinearFactorGraph &graph = gtsam::NonlinearFactorGraph(),
                    const gtsam::Values &values = gtsam::Values());
void fromMsg(const ISAM2Update &slam_update_msg, gtsam::ISAM2 &isam2);
void fromMsg(const ISAM2Update &slam_update_msg, gtsam::NonlinearFactorGraph &graph, gtsam::Values &values);
void fromMsg(const ISAM2Update &slam_update_msg, gtsam::ISAM2 &isam2, gtsam::NonlinearFactorGraph &graph,
             gtsam::Values &values);

}  // namespace bruce_msgs