#ifndef FAST_OGICP_FAST_OGICP_HPP
#define FAST_OGICP_FAST_OGICP_HPP

#include <unordered_map>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/registration.h>
#include <pcl/search/kdtree.h>

#include "octree/GaussionOcTree.h"
#include "ogicp/gicp_settings.hpp"
#include "ogicp/lsq_registration_depth.hpp"

namespace fast_ogicp {

enum class OGICP_REG { PLANE_TO_PLANE, POINT_TO_PLANE, POINT_TO_POINT };  // from source to target

/**
 * @brief Fast Octolized GICP algorithm boosted with OpenMP
 */
template <typename PointSource, typename PointTarget> class FastOGICP : public LsqRegistrationDepth<PointSource, PointTarget> {
 public:
  using Scalar  = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource         = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr      = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget         = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr      = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr      = pcl::shared_ptr<FastOGICP<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const FastOGICP<PointSource, PointTarget>>;
#else
  using Ptr      = boost::shared_ptr<FastOGICP<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const FastOGICP<PointSource, PointTarget>>;
#endif

 protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;

 public:
  FastOGICP();
  virtual ~FastOGICP() override;
  void         setRegMode(OGICP_REG mode);
  void         setOctreeBuildMethod(int type);  // 1 先落在栅格 再计算协方差 2. 先计算每个点协方差 再计算栅格
  void         setDepthChange(bool run);        // 用于设置是否开启 层切换机制 默认未开启
  void         setFirstSearchDepth(int depth);  // 用于设置初始搜索层数 默认 14层 if 0.5m 意味2m ，if 1m 意味4m
  void         setNumThreads(int n);
  void         setResolution(double resolution);
  void         setInputTargetOctree(std::string octree_path);
  void         saveOctree(const PointCloudTargetConstPtr& cloud, std::string octree_path);  // 独立程序 用于保存octree
  virtual void clearSource() override;
  virtual void clearTarget() override;
  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

 protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual void   update_correspondences(const Eigen::Isometry3d& trans);
  virtual double linearize(const Eigen::Isometry3d& trans, int depth, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;
  virtual double compute_error(const Eigen::Isometry3d& trans, int depth) override;

  bool build_goctomap(const PointCloudTargetConstPtr& cloud);
  template <typename PointT>
  bool build_goctomap_new(const PointCloudTargetConstPtr&                                          cloud,
                          pcl::search::KdTree<PointT>&                                             kdtree,
                          std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);
  template <typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr&                        cloud,
                             pcl::search::KdTree<PointT>&                                             kdtree,
                             std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);
  void cov_process(const Eigen::Matrix3f& covariance_matrix_point, Eigen::Matrix3f& covariances);

 protected:
  int num_threads_;
  int k_correspondences_;

  RegularizationMethod                                                    regularization_method_;
  std::shared_ptr<pcl::search::KdTree<PointSource>>                       source_kdtree_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs_;

  std::shared_ptr<pcl::search::KdTree<PointSource>>                       target_kdtree_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs_;

  bool inputOctree_;         // whether input octree map directly
  int octree_build_method_;  // 1 先落在栅格 再计算协方差 2. 先计算每个点协方差 再计算栅格 only used when inputOctree_ is right
  OGICP_REG ogicp_reg_      = OGICP_REG::PLANE_TO_PLANE;
  double    min_resulution_ = 0.5;

  std::shared_ptr<octomap::GaussionOcTree>                                       target_octree_;
  std::vector<std::pair<int, octomap::GaussionOcTreeNode::GaussionDistribution>> pair_srcpt_gd_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>        voxel_mahalanobis_;
};
}  // namespace fast_ogicp

#endif
