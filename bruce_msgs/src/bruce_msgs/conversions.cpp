// GTSAM type serialization
#include <gtsam/base/GenericValue.h>
#include <gtsam/base/serialization.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/GaussianISAM.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "bruce_msgs/conversions.h"

using namespace gtsam;

// https://bitbucket.org/gtborg/gtsam/issues/307/serialization-of-gtsam-classes-such-as
GTSAM_VALUE_EXPORT(Pose3);
GTSAM_VALUE_EXPORT(Point3);
GTSAM_VALUE_EXPORT(Pose2);
GTSAM_VALUE_EXPORT(Point2);
BOOST_CLASS_EXPORT_GUID(PriorFactor<Pose3>, "gtsam_PriorFactor_Pose3");
BOOST_CLASS_EXPORT_GUID(BetweenFactor<Pose3>, "gtsam_BetweenFactor_Pose3");
BOOST_CLASS_EXPORT_GUID(PriorFactor<Pose2>, "gtsam_PriorFactor_Pose2");
BOOST_CLASS_EXPORT_GUID(BetweenFactor<Pose2>, "gtsam_BetweenFactor_Pose2");
BOOST_CLASS_EXPORT_GUID(noiseModel::Constrained, "gtsam_noiseModel_Constrained");
BOOST_CLASS_EXPORT_GUID(noiseModel::Diagonal, "gtsam_noiseModel_Diagonal");
BOOST_CLASS_EXPORT_GUID(noiseModel::Gaussian, "gtsam_noiseModel_Gaussian");
BOOST_CLASS_EXPORT_GUID(noiseModel::Unit, "gtsam_noiseModel_Unit");
BOOST_CLASS_EXPORT_GUID(noiseModel::Isotropic, "gtsam_noiseModel_Isotropic");
BOOST_CLASS_EXPORT_GUID(noiseModel::mEstimator::Cauchy, "gtsam_noiseModel_Cauchy");
BOOST_CLASS_EXPORT_GUID(noiseModel::mEstimator::DCS, "gtsam_noiseModel_DCS");
BOOST_CLASS_EXPORT_GUID(noiseModel::mEstimator::Huber, "gtsam_noiseModel_Huber");
BOOST_CLASS_EXPORT_GUID(noiseModel::Robust, "gtsam_noiseModel_Robust");
BOOST_CLASS_EXPORT_GUID(SharedNoiseModel, "gtsam_SharedNoiseModel");
BOOST_CLASS_EXPORT_GUID(SharedDiagonal, "gtsam_SharedDiagonal");
BOOST_CLASS_EXPORT_GUID(JacobianFactor, "gtsam_JacobianFactor");
BOOST_CLASS_EXPORT_GUID(HessianFactor, "gtsam_HessianFactor");
BOOST_CLASS_EXPORT_GUID(GaussianConditional, "gtsam_GaussianConditional");

namespace bruce_msgs
{
ISAM2UpdatePtr toMsg(const ISAM2 &isam2, const NonlinearFactorGraph &graph, const Values &values)
{
  ISAM2UpdatePtr slam_update_msg(new ISAM2Update);
  std::string isam2_str = serializeBinary(isam2);
  slam_update_msg->isam2.insert(slam_update_msg->isam2.begin(), isam2_str.begin(), isam2_str.end());
  std::string graph_str = serializeBinary(graph);
  slam_update_msg->graph.insert(slam_update_msg->graph.begin(), graph_str.begin(), graph_str.end());
  std::string values_str = serializeBinary(values);
  slam_update_msg->values.insert(slam_update_msg->values.begin(), values_str.begin(), values_str.end());
}
void fromMsg(const ISAM2Update &slam_update_msg, gtsam::ISAM2 &isam2)
{
  std::string isam2_str(slam_update_msg.isam2.begin(), slam_update_msg.isam2.end());
  deserializeBinary(isam2_str, isam2);
}
void fromMsg(const ISAM2Update &slam_update_msg, gtsam::NonlinearFactorGraph &graph, gtsam::Values &values)
{
  std::string graph_str(slam_update_msg.graph.begin(), slam_update_msg.graph.end());
  deserializeBinary(graph_str, graph);
  std::string values_str(slam_update_msg.values.begin(), slam_update_msg.values.end());
  deserializeBinary(values_str, values);
}
void fromMsg(const ISAM2Update &slam_update_msg, gtsam::ISAM2 &isam2, gtsam::NonlinearFactorGraph &graph,
             gtsam::Values &values)
{
  std::string isam2_str(slam_update_msg.isam2.begin(), slam_update_msg.isam2.end());
  deserializeBinary(isam2_str, isam2);
  std::string graph_str(slam_update_msg.graph.begin(), slam_update_msg.graph.end());
  deserializeBinary(graph_str, graph);
  std::string values_str(slam_update_msg.values.begin(), slam_update_msg.values.end());
  deserializeBinary(values_str, values);
}
}  // namespace bruce_msgs