#include <bernstein/yaw_bezier_optimizer.hpp>

namespace traj_opt {

void YawBezierOpt::setup(const double start_yaw,
                        const double start_yaw_rate,
                        const std::vector<double>& end_yaws,
                        const std::vector<double>& end_yaw_rates,
                        const std::vector<double>& time_allocation,
                        const double max_yaw_rate,
                        const double max_yaw_acc) {
    M_ = time_allocation.size();
    t_ = time_allocation;
    max_yaw_rate_ = max_yaw_rate;
    max_yaw_acc_ = max_yaw_acc;
    
    // Initialize boundary conditions for single dimension
    ROS_INFO("Setting up yaw optimization:");
    ROS_INFO("  Start yaw: %f, Start rate: %f", start_yaw, start_yaw_rate);
    for (int i = 0; i < end_yaws.size(); i++) {
        ROS_INFO("  End yaw: %f, End rate: %f", end_yaws[i], end_yaw_rates[i]);
    }
    ROS_INFO("  Segments: %d, Order: %d", M_, N_);
    
    init_ << start_yaw, start_yaw_rate;
    goals_.clear();
    for (int i = 0; i < end_yaws.size(); i++) {
        goals_.push_back(Eigen::Vector2d(end_yaws[i], end_yaw_rates[i]));
    }
    
    // Calculate dimension of optimization problem
    DM_ = M_ * (N_ + 1);  // Total number of control points
    ROS_INFO("  Total control points: %d", DM_);
    
    // Initialize optimization variables
    x_ = Eigen::VectorXd::Zero(DM_);
    
    // Calculate conversion matrices
    calcCtrlPtsCvtMat();
    
    // Calculate cost matrix
    calcMinAccCost();
    
    // Initialize constraint matrices
    A_ = Eigen::MatrixXd::Zero(0, DM_);
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
    r2a_ = Eigen::MatrixXd::Zero(N_ - 1, N_ + 1);
    for (int i = 0; i < N_ - 1; i++) {
        r2a_(i, i) = -(N_ - 1);
        r2a_(i, i + 1) = (N_ - 1);
        r2a_(i, i + 2) = 0;  // Last column is zero
    }
}

void YawBezierOpt::calcMinAccCost() {
    // Calculate cost matrix for minimum acceleration
    Q_ = Eigen::MatrixXd::Zero(DM_, DM_);
    for (int i = 0; i < M_; i++) {
        int idx = i * (N_ + 1);
        Eigen::MatrixXd Qi = Eigen::MatrixXd::Zero(N_ + 1, N_ + 1);
        Qi = r2a_.transpose() * r2a_;
        Q_.block(idx, idx, N_ + 1, N_ + 1) = Qi;
    }
}

void YawBezierOpt::addBoundaryConstraints() {
    // Add start position and velocity constraints
    Eigen::MatrixXd A_start = Eigen::MatrixXd::Zero(2, DM_);
    A_start.block(0, 0, 1, 1) = Eigen::VectorXd::Ones(1);
    A_start.block(1, 0, 1, N_ + 1) = y2r_.row(0);
    
    // Add end position and velocity constraints
    Eigen::MatrixXd A_end = Eigen::MatrixXd::Zero(2, DM_);
    A_end.block(0, DM_ - 1, 1, 1) = Eigen::VectorXd::Ones(1);
    A_end.block(1, DM_ - (N_ + 1), 1, N_ + 1) = y2r_.row(0);
    
    // Combine constraints
    Eigen::MatrixXd A_new(A_.rows() + 4, DM_);
    Eigen::VectorXd ub_new(ub_.size() + 4);
    Eigen::VectorXd lb_new(lb_.size() + 4);
    A_new << A_,
             A_start,
             A_end;
             
    ub_new << ub_,
             init_(0),  // Start yaw
             init_(1),  // Start yaw rate
             goals_[goals_.size() - 1](0),  // End yaw
             goals_[goals_.size() - 1](1);  // End yaw rate

    lb_new << lb_,
             init_(0),  // Start yaw
             init_(1),  // Start yaw rate
             goals_[goals_.size() - 1](0),  // End yaw
             goals_[goals_.size() - 1](1);  // End yaw rate
             
    A_ = A_new;
    ub_ = ub_new;
    lb_ = lb_new;


    // Add constraints for each waypoint
    Eigen::MatrixXd A_waypoint = Eigen::MatrixXd::Zero(goals_.size(), DM_);
    Eigen::VectorXd ub_waypoint = Eigen::VectorXd::Zero(goals_.size());
    Eigen::VectorXd lb_waypoint = Eigen::VectorXd::Zero(goals_.size());
    for (int i = 0; i < goals_.size() - 1; i++) {
        int idx1 = i * (N_ + 1);
        // int idx2 = (i + 1) * (N_ + 1);
        A_waypoint(i, idx1 + N_) = 1.0;
        // A_waypoint(2 * i + 1, idx1 + N_ - 1) = y2r_(0, N_ - 1);
        ub_waypoint(i) = goals_[i](0);
        lb_waypoint(i) = goals_[i](0);
    }

    Eigen::MatrixXd A_new_waypoint(A_.rows() + A_waypoint.rows(), DM_);
    Eigen::VectorXd ub_new_waypoint(ub_.size() + ub_waypoint.size());
    Eigen::VectorXd lb_new_waypoint(lb_.size() + lb_waypoint.size());
    A_new_waypoint << A_,
             A_waypoint;
        
    ub_new_waypoint << ub_,
             ub_waypoint;
    lb_new_waypoint << lb_,
             lb_waypoint;
             
    A_ = A_new_waypoint;
    ub_ = ub_new_waypoint;
    lb_ = lb_new_waypoint;
    
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
    Eigen::VectorXd ub_new(ub_.size() + b_cont.size());
    Eigen::VectorXd lb_new(lb_.size() + b_cont.size());
    
    A_new << A_,
             A_cont;
             
    ub_new << ub_,
             b_cont;
             
    lb_new << lb_,
             b_cont;
             
    A_ = A_new;
    ub_ = ub_new;
    lb_ = lb_new;
}

void YawBezierOpt::addDynamicalConstraints() {
    // Add velocity and acceleration constraints
    int num_vel_constraints = M_ * N_;
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
    Eigen::VectorXd ub_new(ub_.size() + A_vel.rows() + A_acc.rows());
    Eigen::VectorXd lb_new(lb_.size() + A_vel.rows() + A_acc.rows());
    
    A_new << A_,
             A_vel,
             A_acc;
             
    ub_new << ub_,
              Eigen::VectorXd::Constant(A_vel.rows(), max_yaw_rate_),
              Eigen::VectorXd::Constant(A_acc.rows(), max_yaw_acc_);
              
    lb_new << lb_,
              Eigen::VectorXd::Constant(A_vel.rows(), -max_yaw_rate_),
              Eigen::VectorXd::Constant(A_acc.rows(), -max_yaw_acc_);
              
    A_ = A_new;
    ub_ = ub_new;
    lb_ = lb_new;
}

double YawBezierOpt::getYaw(double t) {
    // Convert control points to matrix format for single-dimensional yaw
    // Each column represents a segment, each row represents a control point
    Eigen::MatrixXd ctrl_pts = Eigen::Map<Eigen::MatrixXd>(x_.data(), 1, M_ * (N_ + 1)).transpose();
    // Calculate yaw at time t according to ctrl_pts and Bernstein polynomial
    // Find which segment time t belongs to
    double elapsed_time = 0.0;
    int segment_idx = 0;
    for (int i = 0; i < M_; i++) {
        if (t <= elapsed_time + t_[i]) {
            segment_idx = i;
            break;
        }
        elapsed_time += t_[i];
    }
    
    // Calculate normalized time within the segment
    double s = (t - elapsed_time) / t_[segment_idx];
    
    // Get control points for this segment
    Eigen::VectorXd segment_ctrl_pts = x_.segment(segment_idx * (N_ + 1), N_ + 1);
    
    // Calculate Bernstein basis functions for order N_
    Eigen::VectorXd basis(N_ + 1);
    for (int i = 0; i <= N_; i++) {
        // Binomial coefficient C(N_, i)
        double binomial_coeff = 1.0;
        for (int j = 1; j <= i; j++) {
            binomial_coeff *= (double)(N_ - j + 1) / j;
        }
        
        // Bernstein basis: C(N_, i) * s^i * (1-s)^(N_-i)
        basis(i) = binomial_coeff * std::pow(s, i) * std::pow(1.0 - s, N_ - i);
    }
    
    // Calculate yaw as dot product of control points and basis functions
    double yaw = segment_ctrl_pts.dot(basis);
    
    return yaw;
}

bool YawBezierOpt::optimize() {
    // Setup QP problem
    Eigen::SparseMatrix<double> Q_sparse = Q_.sparseView();
    Eigen::SparseMatrix<double> A_sparse = A_.sparseView();
    
    // Solve using IOSQP
    IOSQP solver;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(DM_);
    ROS_INFO("size of Q_sparse: %d x %d", Q_sparse.rows(), Q_sparse.cols());
    ROS_INFO("size of A_sparse: %d x %d", A_sparse.rows(), A_sparse.cols());
    ROS_INFO("size of q: %d", q.size());
    ROS_INFO("size of lb: %d", lb_.size());
    ROS_INFO("size of ub: %d", ub_.size());
    c_int flag = solver.setMats(Q_sparse, q, A_sparse, lb_, ub_, 1e-4, 1e-4);
    if (flag != 0) {
        ROS_ERROR("Failed to setup IOSQP solver");
        return false;
    }
    
    solver.solve();
    c_int status = solver.getStatus();
    if (status != 1 && status != 2) {
        ROS_ERROR("IOSQP solver failed with status: %d", status);
        return false;
    }
    
    x_ = solver.getPrimalSol();
    return true;
}

}  // namespace traj_opt