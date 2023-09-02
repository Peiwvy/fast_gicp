#ifndef FAST_GICP_LSQ_REGISTRATION_DEPTH_HPP
#define FAST_GICP_LSQ_REGISTRATION_DEPTH_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/registration.h>

namespace fast_ogicp {

enum class LSQ_OPTIMIZER_DEPTH_TYPE { GaussNewton, LevenbergMarquardt };

template <typename PointSource, typename PointTarget> class LsqRegistrationDepth : public pcl::Registration<PointSource, PointTarget, float> {
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
  using Ptr      = pcl::shared_ptr<LsqRegistrationDepth<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const LsqRegistrationDepth<PointSource, PointTarget>>;
#else
  using Ptr      = boost::shared_ptr<LsqRegistrationDepth<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const LsqRegistrationDepth<PointSource, PointTarget>>;
#endif

 protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LsqRegistrationDepth();
  virtual ~LsqRegistrationDepth();

  void setDepthChange(bool run);
  void setFirstSearchDepth(int depth);
  void setRotationEpsilon(double eps);
  void setInitialLambdaFactor(double init_lambda_factor);
  void setDebugPrint(bool lm_debug_print);

  const Eigen::Matrix<double, 6, 6>& getFinalHessian() const;

  double evaluateCost(const Eigen::Matrix4f& relative_pose, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr);

  virtual void swapSourceAndTarget() {}
  virtual void clearSource() {}
  virtual void clearTarget() {}

 protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  bool is_converged(const Eigen::Isometry3d& delta) const;

  virtual double
                 linearize(const Eigen::Isometry3d& trans, int depth, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr) = 0;
  virtual double compute_error(const Eigen::Isometry3d& trans, int depth) = 0;

  bool step_optimize_depth(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_gn_depth(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_lm_depth(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);

 protected:
  double rotation_epsilon_;

  LSQ_OPTIMIZER_DEPTH_TYPE lsq_optimizer_type_;
  int                      lm_max_iterations_;
  double                   lm_init_lambda_factor_;
  double                   lm_lambda_;
  bool                     lm_debug_print_;

  Eigen::Matrix<double, 6, 6> final_hessian_;
  int                         initial_depth_;  // 开始搜索的层数  最下层为16 如果选择0.5的尺寸 则 16 0.5，15 1，14 2，13 4 。
  bool                        layer_change_mode_;  // 是否启用层自调整机制
  int                         depth_;              // dont change manually, also reference by fast_ogicp_impl.hpp
};
}  // namespace fast_ogicp

#endif