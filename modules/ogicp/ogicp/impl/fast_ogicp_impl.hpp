#ifndef FAST_OGICP_FAST_OGICP_IMPL_HPP
#define FAST_OGICP_FAST_OGICP_IMPL_HPP

#include "ogicp/fast_ogicp.hpp"
#include "ogicp/so3.hpp"

#include <chrono>
#include <cmath>

namespace fast_ogicp {

template <typename PointSource, typename PointTarget> FastOGICP<PointSource, PointTarget>::FastOGICP() {
  reg_name_ = "FastOGICP";
  source_kdtree_.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree_.reset(new pcl::search::KdTree<PointSource>);
  regularization_method_ = RegularizationMethod::NORMALIZED_SUM;

  min_resulution_      = 1.0;
  k_correspondences_   = 20;
  num_threads_         = 1;
  octree_build_method_ = 1;  // add here only for inilization

  inputOctree_ = false;  // 调用 setInputTargetOctree 时，会自动设置为true

  target_octree_.reset(new octomap::GaussionOcTree(min_resulution_));
}

template <typename PointSource, typename PointTarget> FastOGICP<PointSource, PointTarget>::~FastOGICP() {}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::setDepthChange(bool run) {
  LsqRegistrationDepth<PointSource, PointTarget>::setDepthChange(run);
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::setFirstSearchDepth(int depth) {
  LsqRegistrationDepth<PointSource, PointTarget>::setFirstSearchDepth(depth);
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::setOctreeBuildMethod(int type) {
  octree_build_method_ = type;
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::setRegMode(OGICP_REG mode) {
  ogicp_reg_ = mode;
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::setResolution(double resolution) {
  min_resulution_ = resolution;
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
  target_octree_->clear();
}

template <typename PointSource, typename PointTarget> void FastOGICP<PointSource, PointTarget>::setInputTargetOctree(std::string octree_path) {
  inputOctree_ = true;
  target_octree_->clear();

  octomap::AbstractOcTree* read_tree = octomap::AbstractOcTree::read(octree_path);
  octomap::GaussionOcTree* readtree  = dynamic_cast<octomap::GaussionOcTree*>(read_tree);
  target_octree_.reset(readtree);
}

template <typename PointSource, typename PointTarget>
void FastOGICP<PointSource, PointTarget>::saveOctree(const PointCloudTargetConstPtr& cloud, std::string octree_path) {
  std::cout << "You choose save octree as tree.ot";
  target_octree_.reset(new octomap::GaussionOcTree(min_resulution_));

  if (octree_build_method_ == 1) {
    build_goctomap(cloud);
  } else if (octree_build_method_ == 2) {
    calculate_covariances(target_, *target_kdtree_, target_covs_);
    build_goctomap_new(cloud, *target_kdtree_, target_covs_);
  }

  target_octree_->write("tree.ot");
}

template <typename PointSource, typename PointTarget>
void FastOGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);

  target_kdtree_->setInputCloud(cloud);
  target_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastOGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);

  if (inputOctree_) {
    pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
    // to avoid [No input target dataset was given!]
  }

  if (ogicp_reg_ == OGICP_REG::PLANE_TO_PLANE) {
    source_kdtree_->setInputCloud(cloud);
  }

  source_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastOGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {

  if (ogicp_reg_ == OGICP_REG::PLANE_TO_PLANE) {
    calculate_covariances(input_, *source_kdtree_, source_covs_);
  }

  if (!inputOctree_) {
    if (octree_build_method_ == 1) {
      build_goctomap(target_);
    } else if (octree_build_method_ == 2) {
      calculate_covariances(target_, *target_kdtree_, target_covs_);
      build_goctomap_new(target_, *target_kdtree_, target_covs_);
    }
  }

  LsqRegistrationDepth<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
bool FastOGICP<PointSource, PointTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr&                        cloud,
  pcl::search::KdTree<PointT>&                                             kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances) {
  // notice： kdtree search need set k_correspondences_ 可能引入误差 比如附近的10个 都到了1m以外
  // 为了解决这个问题 可能需要 voxel 特别小 比如 0.5
  covariances.clear();
  covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int>   k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // 计算协方差 提取最近 k_correspondences_ 个点
    Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    // Eigen::Matrix3f cov_processed;
    // cov_process(cov.block<3, 3>(0, 0).cast<float>(), cov_processed);
    // covariances[i].block<3, 3>(0, 0) = cov_processed.cast<double>();

    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double          lambda = 1e-3;
      Eigen::Matrix3d C      = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv  = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      // 对称矩阵 U V相等
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d                   values;

      switch (regularization_method_) {
        default: std::cerr << "here must not be reached" << std::endl; abort();
        case RegularizationMethod::PLANE: values = Eigen::Vector3d(1, 1, 1e-3); break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);  // array 用于操作矩阵内部 matrix vector用于操作外部。 这个语句： 最大的限制为e-3
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MAX: values = svd.singularValues() / svd.singularValues().maxCoeff(); break;
        case RegularizationMethod::NORMALIZED_SUM: values = svd.singularValues() / svd.singularValues().sum(); break;
      }

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  return true;
}

template <typename PointSource, typename PointTarget>
void FastOGICP<PointSource, PointTarget>::cov_process(const Eigen::Matrix3f& covariance_matrix_point, Eigen::Matrix3f& covariances) {
  if (regularization_method_ == RegularizationMethod::NONE) {
    covariances = covariance_matrix_point;
  } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
    float           lambda = 1e-3;
    Eigen::Matrix3f C      = covariance_matrix_point + lambda * Eigen::Matrix3f::Identity();
    Eigen::Matrix3f C_inv  = C.inverse();
    covariances.setZero();
    covariances = (C_inv / C_inv.norm()).inverse();
  } else {
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance_matrix_point, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f                   values;

    switch (regularization_method_) {
      default: std::cerr << "here must not be reached" << std::endl; abort();
      case RegularizationMethod::PLANE: values = Eigen::Vector3f(1, 1, 1e-3); break;
      case RegularizationMethod::MIN_EIG: values = svd.singularValues().array().max(1e-3); break;
      case RegularizationMethod::NORMALIZED_MIN_EIG:
        values = svd.singularValues() / svd.singularValues().maxCoeff();
        values = values.array().max(1e-3);
        break;
      case RegularizationMethod::NORMALIZED_MAX: values = svd.singularValues() / svd.singularValues().maxCoeff(); break;
    }

    covariances.setZero();
    covariances = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
  }
}

template <typename PointSource, typename PointTarget>
bool FastOGICP<PointSource, PointTarget>::build_goctomap(const PointCloudTargetConstPtr& cloud) {
  target_octree_.reset(new octomap::GaussionOcTree(min_resulution_));

  std::unordered_map<octomap::OcTreeKey, std::vector<PointTarget>, octomap::OcTreeKey::KeyHash> my_map;

  // TODO 测试是否可以 omp
  // 1. 遍历目标点云
  for (auto p : (*cloud).points) {
    auto key = target_octree_->coordToKey(p.x, p.y, p.z, 16);
    my_map[key].push_back(p);
  }

  int min_points_per_voxel = min_resulution_ * 20;
  for (auto& kv : my_map) {
    auto  key    = kv.first;
    auto& cloud_ = kv.second;

    if (cloud_.size() < min_points_per_voxel)
      continue;

    // 2.1 计算 点云集分布
    pcl::PointCloud<PointTarget> cloud_pcl;
    cloud_pcl.points.assign(cloud_.begin(), cloud_.end());  // iterator 不同

    Eigen::Vector4f centroid;
    Eigen::Matrix3f covariance_matrix;
    pcl::computeMeanAndCovarianceMatrix(cloud_pcl, covariance_matrix, centroid);
    covariance_matrix = covariance_matrix * cloud_.size() / (cloud_.size() - 1);

    // 2.3 输入octomap 节点
    auto n = target_octree_->updateNode(key, true);
    n->setGaussionDistribution(cloud_.size(), centroid.head<3>(), covariance_matrix);
  }

  // 3. 更新
  target_octree_->updateInnerOccupancy();
  return true;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
bool FastOGICP<PointSource, PointTarget>::build_goctomap_new(const PointCloudTargetConstPtr&                                          cloud,
                                                             pcl::search::KdTree<PointT>&                                             kdtree,
                                                             std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances) {
  target_octree_->clear();
  target_octree_->setResolution(min_resulution_);

  for (size_t i = 0; i < cloud->points.size(); i++) {
    auto x_ = cloud->points[i].x;
    auto y_ = cloud->points[i].y;
    auto z_ = cloud->points[i].z;

    Eigen::Vector3f centroid(x_, y_, z_);
    auto            covariance        = covariances[i].block<3, 3>(0, 0);
    Eigen::Matrix3f covariance_matrix = covariance.cast<float>();

    auto key = target_octree_->coordToKey(x_, y_, z_, 16);
    auto n   = target_octree_->updateNode(key, true);
    target_octree_->averageNodeGaussionDistribution(key, 1, centroid, covariance_matrix);
  }

  target_octree_->updateInnerOccupancy();
  return true;
}

template <typename PointSource, typename PointTarget>
double FastOGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d&     trans,
                                                      int                          depth,
                                                      Eigen::Matrix<double, 6, 6>* H,
                                                      Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);

  double                                                                                          sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < pair_srcpt_gd_.size(); i++) {
    // 1. get A
    const auto&           corr   = pair_srcpt_gd_[i];
    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();

    // 2. get B
    const auto&     gd      = corr.second;
    const auto&     mean_B3 = gd.centroid.cast<double>();
    Eigen::Vector4d mean_B;
    mean_B.setOnes();
    mean_B.block<3, 1>(0, 0) = mean_B3;

    // 3. 计算 error
    Eigen::Vector4d transed_mean_A = trans * mean_A;
    Eigen::Vector4d error          = mean_B - transed_mean_A;

    sum_errors += error.transpose() * voxel_mahalanobis_[i] * error;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0)           = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3)           = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * voxel_mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * voxel_mahalanobis_[i] * error;

    int thread_num = omp_get_thread_num();
    Hs[thread_num] += Hi;
    bs[thread_num] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
void FastOGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  pair_srcpt_gd_.clear();
  voxel_mahalanobis_.clear();

  // 1. 根据 threads 数量 分配容器
  std::vector<std::vector<std::pair<int, octomap::GaussionOcTreeNode::GaussionDistribution>>> corrs(num_threads_);
  for (auto& c : corrs) {
    c.reserve(input_->size() / num_threads_);
  }

  // 2. 遍历输入点云查找 配对
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    const Eigen::Vector4d        mean_A         = input_->at(i).getVector4fMap().template cast<double>();
    Eigen::Vector4d              transed_mean_A = trans * mean_A;
    octomap::GaussionOcTreeNode* node           = target_octree_->search(transed_mean_A(0), transed_mean_A(1), transed_mean_A(2), this->depth_);
    if (node != NULL && node->isGaussionDistributionSet()) {
      auto gd = node->getGaussionDistribution();
      corrs[omp_get_thread_num()].push_back(std::make_pair(i, gd));
    }
  }

  // 3. 不同线程数据合并
  pair_srcpt_gd_.reserve(input_->size());
  for (const auto& c : corrs) {
    pair_srcpt_gd_.insert(pair_srcpt_gd_.end(), c.begin(), c.end());
  }

  // 4. 计算 相互关系矩阵
  voxel_mahalanobis_.resize(pair_srcpt_gd_.size());
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < pair_srcpt_gd_.size(); i++) {
    switch (ogicp_reg_) {
      case OGICP_REG::PLANE_TO_PLANE: {
        const auto& corr = pair_srcpt_gd_[i];
        // get cov A
        const auto& cov_A = source_covs_[corr.first];
        // get cov B
        const auto&     gd      = corr.second;
        const auto&     cov_B3  = gd.covariance_matrix.cast<double>();
        Eigen::Matrix4d cov_B   = Eigen::Matrix4d::Zero();
        cov_B.block<3, 3>(0, 0) = cov_B3;
        // 计算 协方差矩阵
        Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
        // Eigen::Matrix4d RCR = trans.matrix() * cov_A * trans.matrix().transpose();
        // Eigen::Matrix4d RCR = cov_B;
        RCR(3, 3)                   = 1.0;
        voxel_mahalanobis_[i]       = RCR.inverse();
        voxel_mahalanobis_[i](3, 3) = 0.0;
        break;
      }
      case OGICP_REG::POINT_TO_PLANE: {
        const auto& corr                        = pair_srcpt_gd_[i];
        const auto& gd                          = corr.second;
        const auto& cov_B3                      = gd.covariance_matrix.cast<double>();
        voxel_mahalanobis_[i].block<3, 3>(0, 0) = cov_B3;
        voxel_mahalanobis_[i](3, 3)             = 0.0;
        break;
      }
      case OGICP_REG::POINT_TO_POINT: {
        voxel_mahalanobis_[i]       = Eigen::Matrix4d::Identity();
        voxel_mahalanobis_[i](3, 3) = 0.0;
        break;
      }
    }
  }
}

template <typename PointSource, typename PointTarget>
double FastOGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans, int depth) {
  return linearize(trans, depth, nullptr, nullptr);
}

}  // namespace fast_ogicp

#endif