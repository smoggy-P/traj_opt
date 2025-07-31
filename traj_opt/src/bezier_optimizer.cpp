/**
 * @file bezier_optimizer.cpp
 * @author Siyuan Wu (siyuanwu99@gmail.com)
 * @brief
 * @version 1.0
 * @date 2022-09-08
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "bernstein/bezier_optimizer.hpp"
using namespace Bernstein;

namespace traj_opt {

void BezierOpt::setConstraints(const std::vector<PolyhedronH>& constraints) {
  constraints_ = constraints;
  // assert(static_cast<int>(constraints_.size()) == M_);
}

void BezierOpt::setTimeAllocation(const std::vector<double>& time_allocation) {
  t_ = time_allocation;
  M_ = t_.size();
}

void BezierOpt::setup(const Eigen::Matrix3d&          start,
                      const Eigen::Matrix3d&          end,
                      const std::vector<double>&      time_allocation,
                      const std::vector<PolyhedronH>& constraints,
                      const double&                   max_vel,
                      const double&                   max_acc) {
  max_vel_ = max_vel;
  max_acc_ = max_acc;
  setTimeAllocation(time_allocation);
  setConstraints(constraints);
  init_ = start;
  goal_ = end;

  DM_ = DIM * M_ * (N_ + 1);
  x_.resize(DM_);
  x_.setZero();
  calcMinJerkCost();
  addConstraints();
}

void BezierOpt::calcCtrlPtsCvtMat() {
  p2v_.resize(DIM * N_, DIM * (N_ + 1));
  v2a_.resize(DIM * (N_ - 1), DIM * N_);
  a2j_.resize(DIM * (N_ - 2), DIM * (N_ - 1));
  p2v_.setZero();
  v2a_.setZero();
  a2j_.setZero();
  for (int i = 0; i < N_; i++) {
    p2v_.block(i * DIM, i * DIM, DIM, DIM)       = -N_ * Eigen::MatrixXd::Identity(DIM, DIM);
    p2v_.block(i * DIM, (i + 1) * DIM, DIM, DIM) = N_ * Eigen::MatrixXd::Identity(DIM, DIM);
  }
  for (int i = 0; i < N_ - 1; i++) {
    v2a_.block(i * DIM, i * DIM, DIM, DIM)       = -(N_ - 1) * Eigen::MatrixXd::Identity(DIM, DIM);
    v2a_.block(i * DIM, (i + 1) * DIM, DIM, DIM) = (N_ - 1) * Eigen::MatrixXd::Identity(DIM, DIM);
  }
  for (int i = 0; i < N_ - 2; i++) {
    a2j_.block(i * DIM, i * DIM, DIM, DIM)       = -(N_ - 2) * Eigen::MatrixXd::Identity(DIM, DIM);
    a2j_.block(i * DIM, (i + 1) * DIM, DIM, DIM) = (N_ - 2) * Eigen::MatrixXd::Identity(DIM, DIM);
  }
}

/**
 * @brief cost = x'Qx = x'P'QPx
 * x are control points
 * P is the control points to jerk conversion matrix
 *
 */
 void BezierOpt::calcMinJerkCost() {
  Q_.resize(DM_, DM_);
  Q_.setZero();
  q_.resize(DM_);
  q_.setZero();
  
  Eigen::Matrix<double, DIM, DIM> I = Eigen::MatrixXd::Identity(DIM, DIM);
  Eigen::MatrixXd p2j = a2j_ * v2a_ * p2v_;
  
  // Dynamically compute P matrix for N-degree Bezier curve
  int n_ctrl_pts = N_ - 2;  // Number of control points for jerk calculation
  Eigen::MatrixXd P(DIM * n_ctrl_pts, DIM * n_ctrl_pts);
  P.setZero();
  
  // Compute jerk cost matrix based on Bezier curve degree
  computeJerkCostMatrix(P, n_ctrl_pts, N_);
  
  Eigen::MatrixXd QM = p2j.transpose() * P * p2j;
  
  for (int i = 0; i < M_; i++) {
    Q_.block(i * DIM * (N_ + 1), i * DIM * (N_ + 1), DIM * (N_ + 1), DIM * (N_ + 1)) = QM;
  }
}

void BezierOpt::computeJerkCostMatrix(Eigen::MatrixXd& P, int n_ctrl_pts, int degree) {
  Eigen::Matrix<double, DIM, DIM> I = Eigen::MatrixXd::Identity(DIM, DIM);
  
  // Pre-computed matrices for common Bezier degrees
  if (degree == 4) {  // Cubic Bezier (your original case)
    P.block(0, 0, DIM, DIM) = I / 3.0;
    P.block(0, DIM, DIM, DIM) = I / 6.0;
    P.block(DIM, 0, DIM, DIM) = I / 6.0;
    P.block(DIM, DIM, DIM, DIM) = I / 3.0;
  }
  else if (degree == 5) {  // Quartic Bezier
    P.block(0, 0, DIM, DIM) = I * 0.8;
    P.block(0, DIM, DIM, DIM) = I * 0.6;
    P.block(0, 2*DIM, DIM, DIM) = I * 0.2;
    P.block(DIM, 0, DIM, DIM) = I * 0.6;
    P.block(DIM, DIM, DIM, DIM) = I * 1.2;
    P.block(DIM, 2*DIM, DIM, DIM) = I * 0.6;
    P.block(2*DIM, 0, DIM, DIM) = I * 0.2;
    P.block(2*DIM, DIM, DIM, DIM) = I * 0.6;
    P.block(2*DIM, 2*DIM, DIM, DIM) = I * 0.8;
  }
  else if (degree == 6) {  // Quintic Bezier
    P.block(0, 0, DIM, DIM) = I * 1.5;
    P.block(0, DIM, DIM, DIM) = I * 1.25;
    P.block(0, 2*DIM, DIM, DIM) = I * 0.75;
    P.block(0, 3*DIM, DIM, DIM) = I * 0.25;
    P.block(DIM, 0, DIM, DIM) = I * 1.25;
    P.block(DIM, DIM, DIM, DIM) = I * 2.0;
    P.block(DIM, 2*DIM, DIM, DIM) = I * 1.5;
    P.block(DIM, 3*DIM, DIM, DIM) = I * 0.75;
    P.block(2*DIM, 0, DIM, DIM) = I * 0.75;
    P.block(2*DIM, DIM, DIM, DIM) = I * 1.5;
    P.block(2*DIM, 2*DIM, DIM, DIM) = I * 2.0;
    P.block(2*DIM, 3*DIM, DIM, DIM) = I * 1.25;
    P.block(3*DIM, 0, DIM, DIM) = I * 0.25;
    P.block(3*DIM, DIM, DIM, DIM) = I * 0.75;
    P.block(3*DIM, 2*DIM, DIM, DIM) = I * 1.25;
    P.block(3*DIM, 3*DIM, DIM, DIM) = I * 1.5;
  }
  else {
    // For other degrees, use the general formula
    for (int i = 0; i < n_ctrl_pts; i++) {
      for (int j = 0; j < n_ctrl_pts; j++) {
        double coeff = computeJerkCoefficient(i, j, degree);
        P.block(i * DIM, j * DIM, DIM, DIM) = coeff * I;
      }
    }
  }
}

// Helper function to compute jerk coefficients
double BezierOpt::computeJerkCoefficient(int i, int j, int n) {
  // For Bezier curves of degree n, the jerk cost matrix coefficients are:
  // P[i,j] = integral_0^1 B''_i(t) * B''_j(t) dt
  // where B''_i(t) is the second derivative of the i-th Bezier basis function
  
  if (n < 3) return 0.0;  // No jerk for curves of degree < 3
  
  // General formula for Bezier jerk cost matrix
  double factorial_n = factorial(n);
  double factorial_n_minus_2 = factorial(n - 2);
  
  // Binomial coefficients
  double binom_i = binomialCoeff(n - 2, i);
  double binom_j = binomialCoeff(n - 2, j);
  
  // Beta function: B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
  double beta_val = beta(i + j + 1, 2*n - i - j - 3);
  
  return (factorial_n * factorial_n) / (factorial_n_minus_2 * factorial_n_minus_2) * 
         binom_i * binom_j * beta_val;
}

// Utility functions (add these to your class)
double BezierOpt::factorial(int n) {
  if (n <= 1) return 1.0;
  double result = 1.0;
  for (int i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

double BezierOpt::binomialCoeff(int n, int k) {
  if (k > n || k < 0) return 0.0;
  if (k == 0 || k == n) return 1.0;
  
  double result = 1.0;
  for (int i = 0; i < k; i++) {
    result = result * (n - i) / (i + 1);
  }
  return result;
}

double BezierOpt::beta(double a, double b) {
  // Beta function: B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
  // For integer arguments, we can use factorials
  // B(m,n) = (m-1)!(n-1)!/(m+n-1)!
  if (a > 0 && b > 0 && a == floor(a) && b == floor(b)) {
    return factorial(a - 1) * factorial(b - 1) / factorial(a + b - 1);
  }
  
  // For non-integer arguments, use gamma function approximation
  // or use standard library if available: std::tgamma
  return std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b);
}

void BezierOpt::addConstraints() {
  int num_const = 0;  // number of constraints
  for (auto& c : constraints_) {
    num_const += c.rows();
  }
  num_const *= N_ + 1;                                    // all points inside the polyhedron
  int num_continuous = (1 + M_) * DIM * 3;                // continuous between segments
  int num_dynamical  = M_ * (DIM * N_ + DIM * (N_ - 1));  // maximum velocity and acceleration
  int num            = num_const + num_continuous + num_dynamical;
  std::cout << "num: " << num_continuous << " | " << num_const << " | " << num_dynamical << " || "
            << num << std::endl;
  A_.resize(num, DM_);
  A_.setZero();
  b_.resize(num);
  b_.setZero();
  lb_.resize(num);
  lb_.setZero();
  idx_ = 0;
  addContinuityConstraints();
  addDynamicalConstraints();
  addSafetyConstraints();
}

/**
 * @brief equality constraints for continuity between segments
 *
 */
void BezierOpt::addContinuityConstraints() {
  double t0 = t_[0], tM = t_[M_ - 1];
  /* position continuity */
  Eigen::Matrix<double, DIM, DIM> I = Eigen::MatrixXd::Identity(DIM, DIM);
  /* initial position */
  A_.block(idx_, 0, DIM, DIM) = I;
  b_.segment(idx_, DIM)       = init_.row(0);
  lb_.segment(idx_, DIM)      = init_.row(0);
  idx_ += DIM;
  for (int i = 1; i < M_; i++) {
    A_.block(idx_, i * DIM * (N_ + 1), DIM, DIM)       = I;
    A_.block(idx_, i * DIM * (N_ + 1) - DIM, DIM, DIM) = -I;
    b_.segment(idx_, DIM)                              = Eigen::Vector3d::Zero();
    lb_.segment(idx_, DIM)                             = Eigen::Vector3d::Zero();
    idx_ += DIM;
  }
  /* final position */
  A_.block(idx_, M_ * DIM * (N_ + 1) - DIM, DIM, DIM) = I;
  b_.segment(idx_, DIM)                               = goal_.row(0);
  lb_.segment(idx_, DIM)                              = goal_.row(0);
  idx_ += DIM;

  /* velocity continuity */
  /* initial velocity */
  A_.block(idx_, 0, DIM, DIM)   = -N_ * I;
  A_.block(idx_, DIM, DIM, DIM) = N_ * I;
  b_.segment(idx_, DIM)         = init_.row(1) * t0;
  lb_.segment(idx_, DIM)        = init_.row(1) * t0;
  idx_ += DIM;
  for (int i = 1; i < M_; i++) {
    double t1  = t_[i];
    double t1_ = t_[i - 1];

    A_.block(idx_, i * DIM * (N_ + 1), DIM, DIM)           = -N_ * I / t1;
    A_.block(idx_, i * DIM * (N_ + 1) + DIM, DIM, DIM)     = N_ * I / t1;
    A_.block(idx_, i * DIM * (N_ + 1) - DIM, DIM, DIM)     = -N_ * I / t1_;
    A_.block(idx_, i * DIM * (N_ + 1) - 2 * DIM, DIM, DIM) = N_ * I / t1_;
    b_.segment(idx_, DIM)                                  = Eigen::Vector3d::Zero();
    lb_.segment(idx_, DIM)                                 = Eigen::Vector3d::Zero();
    idx_ += DIM;
  }
  /* final velocity */
  A_.block(idx_, M_ * DIM * (N_ + 1) - DIM * 2, DIM, DIM) = -N_ * I;
  A_.block(idx_, M_ * DIM * (N_ + 1) - DIM, DIM, DIM)     = N_ * I;
  b_.segment(idx_, DIM)                                   = goal_.row(1) * tM;
  lb_.segment(idx_, DIM)                                  = goal_.row(1) * tM;
  idx_ += DIM;

  /* acceleration continuity */
  constexpr int                    DIM3 = DIM * 3;
  Eigen::Matrix<double, DIM, DIM3> p2a  = (v2a_ * p2v_).block<DIM, DIM3>(0, 0);
  /* initial acceleration */
  A_.block(idx_, 0, DIM, DIM3) = p2a;
  b_.segment(idx_, DIM)        = init_.row(2) * t0 * t0;
  lb_.segment(idx_, DIM)       = init_.row(2) * t0 * t0;
  idx_ += DIM;
  for (int i = 1; i < M_; i++) {
    double t2  = pow(t_[i], 2);
    double t2_ = pow(t_[i - 1], 2);

    A_.block(idx_, i * DIM * (N_ + 1), DIM, DIM3)        = p2a / t2;
    A_.block(idx_, i * DIM * (N_ + 1) - DIM3, DIM, DIM3) = -p2a / t2_;
    b_.segment(idx_, DIM)                                = Eigen::Vector3d::Zero();
    lb_.segment(idx_, DIM)                               = Eigen::Vector3d::Zero();
    idx_ += DIM;
  }
  /* final acceleration */
  A_.block(idx_, M_ * DIM * (N_ + 1) - DIM3, DIM, DIM3) = p2a;
  b_.segment(idx_, DIM)                                 = goal_.row(2) * tM * tM;
  lb_.segment(idx_, DIM)                                = goal_.row(2) * tM * tM;
  idx_ += DIM;

  std::cout << "idx: " << idx_ << std::endl;
}

void BezierOpt::addDynamicalConstraints() {
  constexpr int                    DIM2 = DIM * 2;
  constexpr int                    DIM3 = DIM * 3;
  Eigen::Matrix<double, DIM, DIM2> p2v  = p2v_.block<DIM, DIM2>(0, 0);
  Eigen::Matrix<double, DIM, DIM3> p2a  = (v2a_ * p2v_).block<DIM, DIM3>(0, 0);
  Eigen::Matrix<double, DIM, DIM>  I    = Eigen::MatrixXd::Identity(DIM, DIM);

  Eigen::Vector3d ones = Eigen::Vector3d::Ones();

  /* maximum velocity */
  for (int i = 0; i < M_; i++) {
    double t = t_[i];
    for (int j = 0; j < N_; j++) {
      A_.block(idx_, i * DIM * (N_ + 1) + j * DIM, DIM, DIM2) = p2v;
      b_.segment(idx_, DIM)                                   = max_vel_ * ones * t;
      lb_.segment(idx_, DIM)                                  = -max_vel_ * ones * t;
      idx_ += DIM;
    }
  }

  /* maximum acceleration */
  for (int i = 0; i < M_; i++) {
    double t = t_[i];
    for (int j = 0; j < N_ - 1; j++) {
      A_.block(idx_, i * DIM * (N_ + 1) + j * DIM, DIM, DIM3) = p2a;
      b_.segment(idx_, DIM)                                   = max_acc_ * ones * t * t;
      lb_.segment(idx_, DIM)                                  = -max_acc_ * ones * t * t;
      idx_ += DIM;
    }
  }
  std::cout << "idx: " << idx_ << std::endl;
}

void BezierOpt::addSafetyConstraints() {
  if (constraints_.empty()) {
    std::cout << "No constraints" << std::endl;
    return;
  }

  for (int i = 0; i < M_; i++) {
    auto c = constraints_[i];
    for (int j = 0; j < c.rows(); j++) {
      Eigen::Vector4d p = c.row(j);
      for (int n = 0; n < N_ + 1; n++) {
        // std::cout << idx_ << "|" << i * DIM * (N_ + 1) + n * DIM << "|" <<
        // p.head(DIM).transpose()
        //           << std::endl;
        A_.block(idx_, i * DIM * (N_ + 1) + n * DIM, 1, DIM) = p.head(DIM).transpose();

        b_[idx_]  = -p[3];
        lb_[idx_] = -OSQP_INFTY;
        idx_++;
      }
    }
  }
  std::cout << "idx: " << idx_ << std::endl;
}

bool BezierOpt::optimize() {
  IOSQP                       solver;
  Eigen::SparseMatrix<double> Q = Q_.sparseView();
  Eigen::SparseMatrix<double> A = A_.sparseView();

  Eigen::VectorXd lb = Eigen::VectorXd::Constant(x_.size(), -OSQP_INFTY);

  c_int flag = solver.setMats(Q, q_, A, lb_, b_, 1e-3, 1e-3);

  if (flag != 0) {
    std::cout << "Problem non-convex. " << std::endl;
    return false;
  } else {
    solver.solve();
    c_int status = solver.getStatus();
    std::cout << "STATUS: " << status << std::endl;
    x_ = solver.getPrimalSol();
    if (status == 1 || status == 2) {
      return true;
    } else {
      return false;
    }
  }
}

void BezierOpt::calcBezierCurve() {
  Eigen::MatrixXd p = getOptCtrlPtsMat();
  bc_.reset(new BezierCurve(t_, p));
}

// bool BezierOpt::optimizeSDQP() {
//   double sdqp_tol      = 1e-3;
//   double sdqp_max_iter = 1000;

//   double sdqp_rst = sdqp::sdqp<-1>(Q_, q_, A_, b_, x_);
// }

}  // namespace traj_opt