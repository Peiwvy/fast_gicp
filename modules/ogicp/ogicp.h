#pragma once

#include "../registration_interface.h"
#include "ogicp/fast_ogicp.hpp"

class ogicpRegistration : public RegistrationInterface {
 public:
  struct Param {
    int    threads              = 2;
    double resolution           = 0.5;
    int    first_search_depth   = 15;     // 用于设置初始搜索层数
    bool   setDepthChange       = false;  // 用于设置是否开启 层切换机制 默认未开启
    int    setOctreeBuildMethod = 1;      // 1 先落在栅格 再计算协方差 2. 先计算每个点协方差 再计算栅格
    int    setCovMode           = 0;
    // 0. GICP 1. Point to plane ICP 2. Standard ICP
    //               [layer:16 15 14 13 12 11]
    //               [0.5 1.0 2.0 4.0 8.0 16.0]
    // recommended * [1.0 2.0 4.0 8.0 16.0 32.0]
  };

  ogicpRegistration(const Param& config);
  void  setInputTargetOctree(const std::string& path);
  bool  SetInputTarget(const pcl::PointCloud<PointType>::Ptr& input_source) override;
  bool  ScanMatch(const pcl::PointCloud<PointType>::Ptr& input_source,
                  const Eigen::Matrix4f&                 predict_pose,
                  pcl::PointCloud<PointType>::Ptr&       result_cloud_ptr,
                  Eigen::Matrix4f&                       result_pose) override;
  float GetFitnessScore() override;

 private:
  Param                                                        PARAM;
  std::shared_ptr<fast_ogicp::FastOGICP<PointType, PointType>> ogicp_ptr;
};
