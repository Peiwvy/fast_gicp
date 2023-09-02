#include "ogicp/lsq_registration_depth.hpp"
#include "ogicp/so3.hpp"
#include <boost/format.hpp>

// #define DEBUG

namespace fast_ogicp {

template <typename PointTarget, typename PointSource> LsqRegistrationDepth<PointTarget, PointSource>::LsqRegistrationDepth() {
  this->reg_name_         = "LsqRegistrationDepth";
  max_iterations_         = 64;
  rotation_epsilon_       = 2e-4;
  transformation_epsilon_ = 5e-4;

  lsq_optimizer_type_    = LSQ_OPTIMIZER_DEPTH_TYPE::LevenbergMarquardt;
  lm_debug_print_        = false;
  lm_max_iterations_     = 20;
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_             = -1.0;

  final_hessian_.setIdentity();
  layer_change_mode_ = false;
  initial_depth_     = 13;
}

template <typename PointTarget, typename PointSource> LsqRegistrationDepth<PointTarget, PointSource>::~LsqRegistrationDepth() {}

template <typename PointTarget, typename PointSource> void LsqRegistrationDepth<PointTarget, PointSource>::setDepthChange(bool run) {
  layer_change_mode_ = run;
}

template <typename PointTarget, typename PointSource> void LsqRegistrationDepth<PointTarget, PointSource>::setFirstSearchDepth(int depth) {
  initial_depth_ = depth;
}

template <typename PointTarget, typename PointSource> void LsqRegistrationDepth<PointTarget, PointSource>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistrationDepth<PointTarget, PointSource>::setInitialLambdaFactor(double init_lambda_factor) {
  lm_init_lambda_factor_ = init_lambda_factor;
}

template <typename PointTarget, typename PointSource> void LsqRegistrationDepth<PointTarget, PointSource>::setDebugPrint(bool lm_debug_print) {
  lm_debug_print_ = lm_debug_print;
}

template <typename PointTarget, typename PointSource>
const Eigen::Matrix<double, 6, 6>& LsqRegistrationDepth<PointTarget, PointSource>::getFinalHessian() const {
  return final_hessian_;
}

template <typename PointTarget, typename PointSource>
double LsqRegistrationDepth<PointTarget, PointSource>::evaluateCost(const Eigen::Matrix4f&       relative_pose,
                                                                    Eigen::Matrix<double, 6, 6>* H,
                                                                    Eigen::Matrix<double, 6, 1>* b) {
  return this->linearize(Eigen::Isometry3f(relative_pose).cast<double>(), depth_, H, b);
}

template <typename PointTarget, typename PointSource>
void LsqRegistrationDepth<PointTarget, PointSource>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

  if (lm_debug_print_) {
    std::cout << "********************************************" << std::endl;
    std::cout << "***************** optimize *****************" << std::endl;
    std::cout << "********************************************" << std::endl;
  }

  // 迭代优化 核心是step_optimize
  // x0 是优化后的结果， delta 是最小二乘计算出来的增量
  // below
  if (layer_change_mode_) {
    for (depth_ = initial_depth_; depth_ < 17; depth_++) {
      lm_lambda_ = -1.0;
      converged_ = false;
#ifdef DEBUG
      std::cout << "layer change, inilize_depth at: " << depth_ << std::endl;
#endif
      for (nr_iterations_ = 0; nr_iterations_ < max_iterations_ && !converged_; nr_iterations_++) {
#ifdef DEBUG
        std::cout << "iteration: " << nr_iterations_ << std::endl;
#endif
        Eigen::Isometry3d delta;
        if (!step_optimize_depth(x0, delta)) {
          std::cerr << "lm not converged!!" << std::endl;
          break;
        }

        converged_ = is_converged(delta);
#ifdef DEBUG
        if (converged_) {
          std::cout << "convergence!" << std::endl;
        }
#endif
      }
    }
  } else {
    depth_     = initial_depth_;
    lm_lambda_ = -1.0;
    converged_ = false;
#ifdef DEBUG
    std::cout << "layer not change, inilize_depth at: " << depth_ << std::endl;
#endif
    for (nr_iterations_ = 0; nr_iterations_ < max_iterations_ && !converged_; nr_iterations_++) {
#ifdef DEBUG
      std::cout << "iteration: " << nr_iterations_ << std::endl;
#endif
      Eigen::Isometry3d delta;
      if (!step_optimize_depth(x0, delta)) {
        std::cerr << "lm not converged!!" << std::endl;
        break;
      }

      converged_ = is_converged(delta);
#ifdef DEBUG
      if (converged_) {
        std::cout << "convergence!" << std::endl;
      }
#endif
    }
  }

  final_transformation_ = x0.cast<float>().matrix();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistrationDepth<PointTarget, PointSource>::is_converged(const Eigen::Isometry3d& delta) const {
  double          accum = 0.0;
  Eigen::Matrix3d R     = delta.linear() - Eigen::Matrix3d::Identity();  // linear 就是其 rotation 33
  Eigen::Vector3d t     = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / transformation_epsilon_ * t.array().abs();
  // std::cout << r_delta.maxCoeff() << "   " << t_delta.maxCoeff() << std::endl;
  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistrationDepth<PointTarget, PointSource>::step_optimize_depth(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  switch (lsq_optimizer_type_) {
    case LSQ_OPTIMIZER_DEPTH_TYPE::LevenbergMarquardt: return step_lm_depth(x0, delta);
    case LSQ_OPTIMIZER_DEPTH_TYPE::GaussNewton: return step_gn_depth(x0, delta);
  }

  return step_lm_depth(x0, delta);
}

// delta 是变化微量 每次都会重置 返回
// 核心的计算 进行 最小二乘 高斯牛顿法
template <typename PointTarget, typename PointSource>
bool LsqRegistrationDepth<PointTarget, PointSource>::step_gn_depth(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;  // Hessian
  Eigen::Matrix<double, 6, 1> b;  // Jacobian
  // x0 的地方 开始线性化 要返回 H & J 以及 对应的 y0
  double y0 = linearize(x0, depth_, &H, &b);

  // 求解 H * (delta_x)= -J
  // eigen 内置线性方程组求解 Eigen::LDLT cholesky分解 L：下三角单位矩阵 D为对角阵
  // LS :https://zhuanlan.zhihu.com/p/360495510
  // example:  https://www.cnblogs.com/ymd12103410/p/9705792.html
  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
  Eigen::Matrix<double, 6, 1>              d = solver.solve(-b);
  // d 是角度在前3，xyz在后3
  delta.setIdentity();
  delta.linear()      = so3_exp(d.head<3>()).toRotationMatrix();
  delta.translation() = d.tail<3>();

  x0             = delta * x0;
  final_hessian_ = H;

  return true;
}

// 核心的计算 进行 最小二乘 LM
template <typename PointTarget, typename PointSource>
bool LsqRegistrationDepth<PointTarget, PointSource>::step_lm_depth(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double                      y0 = linearize(x0, depth_, &H, &b);

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * H.diagonal().array().abs().maxCoeff();
  }

  double nu = 2.0;
  for (int i = 0; i < lm_max_iterations_; i++) {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H + lm_lambda_ * Eigen::Matrix<double, 6, 6>::Identity());
    Eigen::Matrix<double, 6, 1>              d = solver.solve(-b);

    delta.setIdentity();
    delta.linear()      = so3_exp(d.head<3>()).toRotationMatrix();  // 旋转矩阵
    delta.translation() = d.tail<3>();                              // 平移

    Eigen::Isometry3d xi = delta * x0;
    double            yi = compute_error(xi, depth_);
    double rho = (y0 - yi) / (d.dot(lm_lambda_ * d - b));  // rho 下降因子 据此 调节lambda 物理意义：相似度 分子是函数 分母是泰勒后
                                                           //  https://zhuanlan.zhihu.com/p/136143299
    // Nielsen 1999
    if (lm_debug_print_) {
      if (i == 0) {
        std::cout << boost::format("--- LM optimization ---\n%5s %15s %15s %15s %15s %15s %5s\n") % "i" % "y0" % "yi" % "rho" % "lambda" % "|delta|"
                       % "dec";
      }
      char dec = rho > 0.0 ? 'x' : ' ';
      std::cout << boost::format("%5d %15g %15g %15g %15g %15g %5c") % i % y0 % yi % rho % lm_lambda_ % d.norm() % dec << std::endl;
    }

    if (rho < 0) {
      if (is_converged(delta)) {
        return true;
      }

      lm_lambda_ = nu * lm_lambda_;
      nu         = 2 * nu;
      continue;
    }

    // rho > 0 结束
    x0             = xi;
    lm_lambda_     = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    final_hessian_ = H;
    nu             = 2.0;

    return true;
  }

  return false;
}

}  // namespace fast_ogicp