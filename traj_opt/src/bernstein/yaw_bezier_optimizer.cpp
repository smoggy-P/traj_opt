#include <bernstein/yaw_bezier_optimizer.hpp>

namespace traj_opt {

void YawBezierOpt::setup(const double start_yaw,
                        const double start_yaw_rate,
                        const double end_yaw,
                        const double end_yaw_rate,
                        const std::vector<double>& time_allocation,
                        const double max_yaw_rate,
                        const double max_yaw_acc) {
    M_ = time_allocation.size();
    t_ = time_allocation;
    max_yaw_rate_ = max_yaw_rate;
    max_yaw_acc_ = max_yaw_acc;
    
    // Initialize boundary conditions
    init_ << start_yaw, start_yaw_rate;
    goal_ << end_yaw, end_yaw_rate;
    
    // Calculate dimension of optimization problem
    DM_ = M_ * (N_ + 1);
    
    // Initialize optimization variables
    x_ = Eigen::VectorXd::Zero(DM_);
    
    // Calculate conversion matrices
    calcCtrlPtsCvtMat();
    
    // Calculate cost matrix
    calcMinAccCost();
    
    // Initialize constraint matrices
    A_ = Eigen::MatrixXd::Zero(0, DM_);
    b_ = Eigen::VectorXd::Zero(0);
    ub_ = Eigen::VectorXd::Zero(0);
    lb_ = Eigen::VectorXd::Zero(0);
    
    // Add constraints
    addBoundaryConstraints();
    addContinuityConstraints();
    addDynamicalConstraints();
}

void YawBezierOpt::calcCtrlPtsCvtMat() {
    // Calculate yaw to rate conversion matrix
    y2r_ = Eigen::MatrixXd::Zero(N_, N_ + 1);
    for (int i = 0; i < N_; i++) {
        y2r_(i, i) = -N_;
        y2r_(i, i + 1) = N_;
    }
    
    // Calculate rate to acceleration conversion matrix
    r2a_ = Eigen::MatrixXd::Zero(N_ - 1, N_);
    for (int i = 0; i < N_ - 1; i++) {
        r2a_(i, i) = -(N_ - 1);
        r2a_(i, i + 1) = N_ - 1;
    }
}

void YawBezierOpt::calcMinAccCost() {
    // Calculate cost matrix for minimum acceleration
    Q_ = Eigen::MatrixXd::Zero(DM_, DM_);
    for (int i = 0; i < M_; i++) {
        int idx = i * (N_ + 1);
        Eigen::MatrixXd Qi = Eigen::MatrixXd::Zero(N_ + 1, N_ + 1);
        
        // Calculate acceleration cost for each segment
        for (int j = 0; j <= N_ - 2; j++) {
            double c = (N_ * (N_ - 1)) / (t_[i] * t_[i]);
            Qi(j, j) += c;
            Qi(j, j + 1) -= 2 * c;
            Qi(j, j + 2) += c;
        }
        
        Q_.block(idx, idx, N_ + 1, N_ + 1) = Qi;
    }
}

void YawBezierOpt::addBoundaryConstraints() {
    // Add start position and velocity constraints
    Eigen::MatrixXd A_start = Eigen::MatrixXd::Zero(2, DM_);
    A_start.block(0, 0, 1, N_ + 1) = Eigen::VectorXd::Ones(N_ + 1).transpose();
    A_start.block(1, 0, 1, N_) = y2r_.row(0);
    
    // Add end position and velocity constraints
    Eigen::MatrixXd A_end = Eigen::MatrixXd::Zero(2, DM_);
    A_end.block(0, DM_ - (N_ + 1), 1, N_ + 1) = Eigen::VectorXd::Ones(N_ + 1).transpose();
    A_end.block(1, DM_ - (N_ + 1), 1, N_) = y2r_.row(0);
    
    // Combine constraints
    Eigen::MatrixXd A_new(A_.rows() + 4, DM_);
    Eigen::VectorXd b_new(b_.size() + 4);
    
    A_new << A_,
             A_start,
             A_end;
             
    b_new << b_,
             init_(0),
             init_(1),
             goal_(0),
             goal_(1);
             
    A_ = A_new;
    b_ = b_new;
}

void YawBezierOpt::addContinuityConstraints() {
    // Add continuity constraints for position and velocity at segment boundaries
    Eigen::MatrixXd A_cont = Eigen::MatrixXd::Zero(2 * (M_ - 1), DM_);
    Eigen::VectorXd b_cont = Eigen::VectorXd::Zero(2 * (M_ - 1));
    
    for (int i = 0; i < M_ - 1; i++) {
        int idx1 = i * (N_ + 1);
        int idx2 = (i + 1) * (N_ + 1);
        
        // Position continuity
        A_cont(2 * i, idx1 + N_) = 1.0;
        A_cont(2 * i, idx2) = -1.0;
        
        // Velocity continuity
        A_cont(2 * i + 1, idx1 + N_ - 1) = y2r_(0, N_ - 1);
        A_cont(2 * i + 1, idx1 + N_) = y2r_(0, N_);
        A_cont(2 * i + 1, idx2) = -y2r_(0, 0);
        A_cont(2 * i + 1, idx2 + 1) = -y2r_(0, 1);
    }
    
    // Combine with existing constraints
    Eigen::MatrixXd A_new(A_.rows() + A_cont.rows(), DM_);
    Eigen::VectorXd b_new(b_.size() + b_cont.size());
    
    A_new << A_,
             A_cont;
             
    b_new << b_,
             b_cont;
             
    A_ = A_new;
    b_ = b_new;
}

void YawBezierOpt::addDynamicalConstraints() {
    // Add velocity and acceleration constraints
    int num_vel_constraints = M_ * (N_);
    int num_acc_constraints = M_ * (N_ - 1);
    
    Eigen::MatrixXd A_vel = Eigen::MatrixXd::Zero(num_vel_constraints, DM_);
    Eigen::MatrixXd A_acc = Eigen::MatrixXd::Zero(num_acc_constraints, DM_);
    
    for (int i = 0; i < M_; i++) {
        int idx = i * (N_ + 1);
        
        // Velocity constraints
        A_vel.block(i * N_, idx, N_, N_ + 1) = y2r_;
        
        // Acceleration constraints
        A_acc.block(i * (N_ - 1), idx, N_ - 1, N_ + 1) = r2a_;
    }
    
    // Combine with existing constraints
    Eigen::MatrixXd A_new(A_.rows() + A_vel.rows() + A_acc.rows(), DM_);
    Eigen::VectorXd b_new(b_.size() + A_vel.rows() + A_acc.rows());
    Eigen::VectorXd ub_new(ub_.size() + A_vel.rows() + A_acc.rows());
    Eigen::VectorXd lb_new(lb_.size() + A_vel.rows() + A_acc.rows());
    
    A_new << A_,
             A_vel,
             A_acc;
             
    b_new << b_,
             Eigen::VectorXd::Zero(A_vel.rows() + A_acc.rows());
             
    ub_new << ub_,
              Eigen::VectorXd::Constant(A_vel.rows(), max_yaw_rate_),
              Eigen::VectorXd::Constant(A_acc.rows(), max_yaw_acc_);
              
    lb_new << lb_,
              Eigen::VectorXd::Constant(A_vel.rows(), -max_yaw_rate_),
              Eigen::VectorXd::Constant(A_acc.rows(), -max_yaw_acc_);
              
    A_ = A_new;
    b_ = b_new;
    ub_ = ub_new;
    lb_ = lb_new;
}

void YawBezierOpt::calcBezierCurve() {
    // Convert control points to matrix format
    Eigen::MatrixXd ctrl_pts = Eigen::Map<Eigen::MatrixXd>(x_.data(), 1, DM_).transpose();
    // Create new BezierCurve with time and control points
    bc_.reset(new BezierCurve(t_, ctrl_pts));
}

bool YawBezierOpt::optimize() {
    // Setup QP problem
    Eigen::SparseMatrix<double> Q_sparse = Q_.sparseView();
    Eigen::SparseMatrix<double> A_sparse = A_.sparseView();
    
    // Solve using IOSQP
    IOSQP solver;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(DM_);
    
    c_int flag = solver.setMats(Q_sparse, q, A_sparse, lb_, ub_, 1e-4, 1e-4);
    if (flag != 0) {
        return false;
    }
    
    solver.solve();
    c_int status = solver.getStatus();
    if (status != 1 && status != 2) {
        return false;
    }
    
    x_ = solver.getPrimalSol();
    return true;
}

}  // namespace traj_opt