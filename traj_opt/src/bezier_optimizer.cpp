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
#include <cmath>
#include <limits>
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
  setup(start, end, time_allocation, constraints, max_vel, max_acc, false);
}

void BezierOpt::setup(const Eigen::Matrix3d&          start,
                      const Eigen::Matrix3d&          end,
                      const std::vector<double>&      time_allocation,
                      const std::vector<PolyhedronH>& constraints,
                      const double&                   max_vel,
                      const double&                   max_acc,
                      bool                             enforce_final_dynamics) {
  enforce_final_dynamics_ = enforce_final_dynamics;
  max_vel_                = max_vel;
  max_acc_                = max_acc;
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
  int position_rows  = (1 + M_) * DIM;
  int velocity_rows  = (M_ + (enforce_final_dynamics_ ? 1 : 0)) * DIM;
  int accel_rows     = (M_ + (enforce_final_dynamics_ ? 1 : 0)) * DIM;
  int num_continuous = position_rows + velocity_rows + accel_rows;  // continuous between segments
  int num_dynamical  = M_ * (DIM * N_ + DIM * (N_ - 1));            // maximum velocity and acceleration
  int num            = num_const + num_continuous + num_dynamical;
  std::cout << "num: " << num_continuous << " | " << num_const << " | " << num_dynamical << " || "
            << num << std::endl;
  A_.resize(num, DM_);
  A_.setZero();
  b_.resize(num);
  b_.setZero();
  lb_.resize(num);
  lb_.setZero();
  constraint_labels_.clear();
  constraint_labels_.reserve(num);
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
  for (int d = 0; d < DIM; ++d) {
    constraint_labels_.push_back("continuity:init_pos:dim" + std::to_string(d));
  }
  idx_ += DIM;
  for (int i = 1; i < M_; i++) {
    A_.block(idx_, i * DIM * (N_ + 1), DIM, DIM)       = I;
    A_.block(idx_, i * DIM * (N_ + 1) - DIM, DIM, DIM) = -I;
    b_.segment(idx_, DIM)                              = Eigen::Vector3d::Zero();
    lb_.segment(idx_, DIM)                             = Eigen::Vector3d::Zero();
    for (int d = 0; d < DIM; ++d) {
      constraint_labels_.push_back("continuity:seg" + std::to_string(i - 1) + "->" +
                                   std::to_string(i) + ":pos:dim" + std::to_string(d));
    }
    idx_ += DIM;
  }
  /* final position */
  A_.block(idx_, M_ * DIM * (N_ + 1) - DIM, DIM, DIM) = I;
  b_.segment(idx_, DIM)                               = goal_.row(0);
  lb_.segment(idx_, DIM)                              = goal_.row(0);
  for (int d = 0; d < DIM; ++d) {
    constraint_labels_.push_back("continuity:final_pos:dim" + std::to_string(d));
  }
  idx_ += DIM;

  /* velocity continuity */
  /* initial velocity */
  A_.block(idx_, 0, DIM, DIM)   = -N_ * I;
  A_.block(idx_, DIM, DIM, DIM) = N_ * I;
  b_.segment(idx_, DIM)         = init_.row(1) * t0;
  lb_.segment(idx_, DIM)        = init_.row(1) * t0;
  for (int d = 0; d < DIM; ++d) {
    constraint_labels_.push_back("continuity:init_vel:dim" + std::to_string(d));
  }
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
    for (int d = 0; d < DIM; ++d) {
      constraint_labels_.push_back("continuity:seg" + std::to_string(i - 1) + "->" +
                                   std::to_string(i) + ":vel:dim" + std::to_string(d));
    }
    idx_ += DIM;
  }
  /* final velocity */
  if (enforce_final_dynamics_) {
    A_.block(idx_, M_ * DIM * (N_ + 1) - DIM * 2, DIM, DIM) = -N_ * I;
    A_.block(idx_, M_ * DIM * (N_ + 1) - DIM, DIM, DIM)     = N_ * I;
    b_.segment(idx_, DIM)                                   = goal_.row(1) * tM;
    lb_.segment(idx_, DIM)                                  = goal_.row(1) * tM;
    for (int d = 0; d < DIM; ++d) {
      constraint_labels_.push_back("continuity:final_vel:dim" + std::to_string(d));
    }
    idx_ += DIM;
  }

  /* acceleration continuity */
  constexpr int                    DIM3 = DIM * 3;
  Eigen::Matrix<double, DIM, DIM3> p2a  = (v2a_ * p2v_).block<DIM, DIM3>(0, 0);
  /* initial acceleration */
  A_.block(idx_, 0, DIM, DIM3) = p2a;
  b_.segment(idx_, DIM)        = init_.row(2) * t0 * t0;
  lb_.segment(idx_, DIM)       = init_.row(2) * t0 * t0;
  for (int d = 0; d < DIM; ++d) {
    constraint_labels_.push_back("continuity:init_acc:dim" + std::to_string(d));
  }
  idx_ += DIM;
  for (int i = 1; i < M_; i++) {
    double t2  = pow(t_[i], 2);
    double t2_ = pow(t_[i - 1], 2);

    A_.block(idx_, i * DIM * (N_ + 1), DIM, DIM3)        = p2a / t2;
    A_.block(idx_, i * DIM * (N_ + 1) - DIM3, DIM, DIM3) = -p2a / t2_;
    b_.segment(idx_, DIM)                                = Eigen::Vector3d::Zero();
    lb_.segment(idx_, DIM)                               = Eigen::Vector3d::Zero();
    for (int d = 0; d < DIM; ++d) {
      constraint_labels_.push_back("continuity:seg" + std::to_string(i - 1) + "->" +
                                   std::to_string(i) + ":acc:dim" + std::to_string(d));
    }
    idx_ += DIM;
  }
  /* final acceleration */
  if (enforce_final_dynamics_) {
    A_.block(idx_, M_ * DIM * (N_ + 1) - DIM3, DIM, DIM3) = p2a;
    b_.segment(idx_, DIM)                                 = goal_.row(2) * tM * tM;
    lb_.segment(idx_, DIM)                                = goal_.row(2) * tM * tM;
    for (int d = 0; d < DIM; ++d) {
      constraint_labels_.push_back("continuity:final_acc:dim" + std::to_string(d));
    }
    idx_ += DIM;
  }

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
      for (int d = 0; d < DIM; ++d) {
        constraint_labels_.push_back("dynamical:seg" + std::to_string(i) + ":vel:cp" +
                                     std::to_string(j) + ":dim" + std::to_string(d));
      }
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
      for (int d = 0; d < DIM; ++d) {
        constraint_labels_.push_back("dynamical:seg" + std::to_string(i) + ":acc:cp" +
                                     std::to_string(j) + ":dim" + std::to_string(d));
      }
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
        constraint_labels_.push_back("safety:seg" + std::to_string(i) + ":plane" +
                                     std::to_string(j) + ":cp" + std::to_string(n));
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
      if (status == -3) {
        std::cout << "[BezierOpt][debug] OSQP reports primal infeasible (status -3). "
                  << "Dumping suspect constraints..." << std::endl;
        debugDumpInfeasibleConstraints();
      }
      return false;
    }
  }
}

void BezierOpt::calcBezierCurve() {
  Eigen::MatrixXd p = getOptCtrlPtsMat();
  bc_.reset(new BezierCurve(t_, p));
}

void BezierOpt::debugDumpInfeasibleConstraints() const {
  const double zero_row_eps = 1e-12;
  const double bound_eps    = 1e-9;

  auto labelForRow = [&](int row) -> std::string {
    if (row >= 0 && row < static_cast<int>(constraint_labels_.size())) {
      return constraint_labels_[row];
    }
    return "<unlabeled_row_" + std::to_string(row) + ">";
  };

  auto isSingleVarConstraint = [&](int row, int& col_out, double& coeff_out) -> bool {
    int nz_col = -1;
    double nz_val = 0.0;
    for (int c = 0; c < A_.cols(); ++c) {
      double v = A_(row, c);
      if (std::abs(v) > 1e-9) {
        if (nz_col != -1) return false;  // more than one non-zero
        nz_col = c;
        nz_val = v;
      }
    }
    if (nz_col == -1) return false;
    col_out = nz_col;
    coeff_out = nz_val;
    return true;
  };

  std::cout << "[BezierOpt][debug] total constraints: " << A_.rows()
            << ", variables: " << A_.cols() << std::endl;

  // Quick sanity on time allocation
  double t_min = std::numeric_limits<double>::infinity();
  double t_max = 0.0;
  for (double ti : t_) {
    t_min = std::min(t_min, ti);
    t_max = std::max(t_max, ti);
  }
  std::cout << "[BezierOpt][debug] time allocation min/max: " << t_min << " / " << t_max
            << std::endl;

  int bad_bounds_cnt   = 0;
  int zero_row_cnt     = 0;
  int nan_inf_cnt      = 0;
  int equality_cnt     = 0;
  int inequality_cnt   = 0;
  int single_var_cnt   = 0;

  // Track per-variable feasible interval if constraint only touches one variable
  std::vector<double> var_lb(A_.cols(), -std::numeric_limits<double>::infinity());
  std::vector<double> var_ub(A_.cols(), std::numeric_limits<double>::infinity());
  for (int r = 0; r < A_.rows(); ++r) {
    double lb = (r < lb_.size()) ? lb_[r] : std::numeric_limits<double>::quiet_NaN();
    double ub = (r < b_.size()) ? b_[r] : std::numeric_limits<double>::quiet_NaN();
    double norm_row = A_.row(r).norm();

    bool has_nan_inf = !std::isfinite(lb) || !std::isfinite(ub) ||
                       !A_.row(r).array().isFinite().all();
    bool bound_conflict = (lb > ub + bound_eps);
    bool zero_row_conflict =
        (norm_row < zero_row_eps) && (std::abs(lb) > bound_eps || std::abs(ub) > bound_eps);
    bool equality_row = std::abs(lb - ub) <= bound_eps;
    equality_cnt += equality_row ? 1 : 0;
    inequality_cnt += equality_row ? 0 : 1;

    if (has_nan_inf || bound_conflict || zero_row_conflict) {
      std::cout << "[BezierOpt][debug] row " << r << " (" << labelForRow(r) << ") "
                << "lb=" << lb << ", ub=" << ub
                << ", ||A_row||=" << norm_row;
      if (bound_conflict) std::cout << " [lb>ub]";
      if (zero_row_conflict) std::cout << " [zero_row_nonzero_bound]";
      if (has_nan_inf) std::cout << " [nan/inf]";
      std::cout << std::endl;

      // Print a compact view of non-zero coefficients to help localize variables
      std::cout << "  nz coefficients (col:value): ";
      int printed = 0;
      for (int c = 0; c < A_.cols(); ++c) {
        double v = A_(r, c);
        if (std::abs(v) > 1e-9) {
          std::cout << c << ":" << v << " ";
          if (++printed >= 8) break;  // avoid flooding
        }
      }
      if (printed == 0) std::cout << "<all zero>";
      std::cout << std::endl;
    }

    bad_bounds_cnt += bound_conflict ? 1 : 0;
    zero_row_cnt   += zero_row_conflict ? 1 : 0;
    nan_inf_cnt    += has_nan_inf ? 1 : 0;

    int col = -1;
    double coeff = 0.0;
    if (isSingleVarConstraint(r, col, coeff)) {
      single_var_cnt++;
      double c_lb = lb / coeff;
      double c_ub = ub / coeff;
      if (coeff < 0) std::swap(c_lb, c_ub);
      var_lb[col] = std::max(var_lb[col], c_lb);
      var_ub[col] = std::min(var_ub[col], c_ub);
    }
  }

  std::cout << "[BezierOpt][debug] summary: "
            << "bad_bounds=" << bad_bounds_cnt
            << ", zero_row_nonzero_bound=" << zero_row_cnt
            << ", nan_or_inf=" << nan_inf_cnt
            << ", equality_rows=" << equality_cnt
            << ", inequality_rows=" << inequality_cnt
            << ", single_var_rows=" << single_var_cnt
            << std::endl;

  // Report per-variable interval conflicts (only derived from single-var constraints)
  for (int c = 0; c < A_.cols(); ++c) {
    if (var_lb[c] > var_ub[c] + bound_eps) {
      std::cout << "[BezierOpt][debug] variable " << c
                << " has conflicting single-var bounds: "
                << "[" << var_lb[c] << ", " << var_ub[c] << "]" << std::endl;
    }
  }

  // Check for obvious conflicting duplicate rows (same coefficients, disjoint bounds)
  const double same_row_eps = 1e-8;
  int duplicate_conflicts = 0;
  for (int r1 = 0; r1 < A_.rows(); ++r1) {
    for (int r2 = r1 + 1; r2 < A_.rows(); ++r2) {
      if ((A_.row(r1) - A_.row(r2)).norm() < same_row_eps) {
        double lb1 = lb_[r1], ub1 = b_[r1];
        double lb2 = lb_[r2], ub2 = b_[r2];
        if (lb1 > ub2 + bound_eps || lb2 > ub1 + bound_eps) {
          std::cout << "[BezierOpt][debug] duplicate-row conflict between "
                    << r1 << " (" << labelForRow(r1) << ") and "
                    << r2 << " (" << labelForRow(r2) << ") "
                    << "bounds [" << lb1 << "," << ub1 << "] vs ["
                    << lb2 << "," << ub2 << "]" << std::endl;
          duplicate_conflicts++;
          if (duplicate_conflicts > 8) {
            std::cout << "[BezierOpt][debug] ... more duplicate conflicts omitted"
                      << std::endl;
            r1 = A_.rows();  // break outer
            break;
          }
        }
      }
    }
  }
}

// bool BezierOpt::optimizeSDQP() {
//   double sdqp_tol      = 1e-3;
//   double sdqp_max_iter = 1000;

//   double sdqp_rst = sdqp::sdqp<-1>(Q_, q_, A_, b_, x_);
// }

}  // namespace traj_opt