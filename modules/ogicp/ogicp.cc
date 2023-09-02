#include "ogicp.h"

ogicpRegistration::ogicpRegistration(const Param& config) {
  ogicp_ptr = std::make_shared<fast_ogicp::FastOGICP<PointType, PointType>>();

  PARAM = config;

  ogicp_ptr->setNumThreads(PARAM.threads);
  ogicp_ptr->setResolution(PARAM.resolution);
  ogicp_ptr->setFirstSearchDepth(PARAM.first_search_depth);
  ogicp_ptr->setDepthChange(PARAM.setDepthChange);
  if (PARAM.setCovMode == 0)
    ogicp_ptr->setRegMode(fast_ogicp::OGICP_REG::PLANE_TO_PLANE);
  else if (PARAM.setCovMode == 1)
    ogicp_ptr->setRegMode(fast_ogicp::OGICP_REG::POINT_TO_PLANE);
  else
    ogicp_ptr->setRegMode(fast_ogicp::OGICP_REG::POINT_TO_POINT);
  ogicp_ptr->setOctreeBuildMethod(PARAM.setOctreeBuildMethod);

  // Notice :  below used to avoid: No input target dataset was given!
  pcl::PointCloud<PointType>::Ptr cloud_ptr(new pcl::PointCloud<PointType>());

  PointType p;
  p.x = 0;
  p.y = 1;
  p.z = 2;
  cloud_ptr->push_back(p);
  ogicp_ptr->setInputTarget(cloud_ptr);
}
void ogicpRegistration::setInputTargetOctree(const std::string& path) {
  ogicp_ptr->setInputTargetOctree(path);
}

bool ogicpRegistration::SetInputTarget(const pcl::PointCloud<PointType>::Ptr& input_source) {
  ogicp_ptr->setInputTarget(input_source);
  return true;
}

bool ogicpRegistration::ScanMatch(const pcl::PointCloud<PointType>::Ptr& input_source,
                                  const Eigen::Matrix4f&                 predict_pose,
                                  pcl::PointCloud<PointType>::Ptr&       result_cloud_ptr,
                                  Eigen::Matrix4f&                       result_pose) {
  ogicp_ptr->setInputSource(input_source);
  ogicp_ptr->align(*result_cloud_ptr, predict_pose);
  result_pose = ogicp_ptr->getFinalTransformation();

  return true;
}

float ogicpRegistration::GetFitnessScore() {
  return ogicp_ptr->getFitnessScore();
}