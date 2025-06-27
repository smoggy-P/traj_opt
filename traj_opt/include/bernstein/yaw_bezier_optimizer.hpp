/**
 * @file yaw_bezier_optimizer.hpp
 * @brief Yaw angle planning using Bezier curves with minimum acceleration
 * @version 1.0
 * @date 2024-03-19
 */

#ifndef _YAW_BEZIER_OPTIMIZER_H_
#define _YAW_BEZIER_OPTIMIZER_H_

#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <traj_utils/bernstein.hpp>
#include <iosqp.hpp>
#include <ros/ros.h>

namespace traj_opt {

typedef Bernstein::Bezier BezierCurve;

/**
 * @brief Yaw angle planning using Bezier curves with minimum acceleration
 * This class optimizes a single-dimensional yaw angle trajectory using Bezier curves.
 * Unlike the position Bezier curve which handles 3D (x,y,z), this handles only yaw angle.
 */
class YawBezierOpt {
public:
    YawBezierOpt() {
        N_ = Bernstein::ORDER;  // Default order
        calcCtrlPtsCvtMat();
    }
    
    YawBezierOpt(const int& N) {
        N_ = N;
        calcCtrlPtsCvtMat();
    }
    
    ~YawBezierOpt() {}

    /* Main API */
    void setup(const double start_yaw,              // Initial yaw angle
              const double start_yaw_rate,          // Initial yaw rate
              const std::vector<double>& end_yaws,  // Final yaw angles
              const std::vector<double>& end_yaw_rates,  // Final yaw rates
              const std::vector<double>& time_allocation,  // Time allocation for each segment
              const double max_yaw_rate = 1.0,      // Maximum yaw rate constraint
              const double max_yaw_acc = 1.0);      // Maximum yaw acceleration constraint

    bool optimize();

    /* Getters */
    inline Eigen::VectorXd getOptCtrlPts() { return x_; }
    inline BezierCurve getOptBezier() {
        // calcBezierCurve();
        return *bc_;
    }
    double getYaw(double t);

private:
    BezierCurve::Ptr bc_;  // Bezier curve for single-dimensional yaw

    int M_;    // number of segments
    int N_;    // order of the polynomial
    int DM_;   // dimension of the optimization problem (M_ * (N_ + 1))
    double max_yaw_rate_;
    double max_yaw_acc_;

    std::vector<double> t_;            // time allocation
    Eigen::Vector2d init_;      // [yaw; yaw_rate] for single dimension
    std::vector<Eigen::Vector2d> goals_;  

    Eigen::MatrixXd Q_;   // cost matrix for minimum acceleration
    Eigen::MatrixXd A_;   // constraint matrix
    Eigen::VectorXd b_;   // bound vector
    Eigen::VectorXd ub_;  // upper bound vector
    Eigen::VectorXd lb_;  // lower bound vector
    Eigen::VectorXd x_;   // vector of control points for single-dimensional yaw

    Eigen::MatrixXd y2r_;  // yaw to rate conversion matrix (N_ x (N_+1))
    Eigen::MatrixXd r2a_;  // rate to acceleration conversion matrix ((N_-1) x (N_+1))

    void calcCtrlPtsCvtMat();    // Calculate conversion matrices for single dimension
    void calcMinAccCost();       // Calculate minimum acceleration cost
    void addContinuityConstraints();  // Add continuity constraints
    void addDynamicalConstraints();   // Add dynamical constraints
    void addBoundaryConstraints();    // Add boundary constraints

public:
    typedef std::shared_ptr<YawBezierOpt> Ptr;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace traj_opt

#endif  // _YAW_BEZIER_OPTIMIZER_H_ 