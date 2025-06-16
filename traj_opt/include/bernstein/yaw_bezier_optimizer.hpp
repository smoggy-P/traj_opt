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

namespace traj_opt {

typedef Bernstein::Bezier BezierCurve;

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
    void setup(const double start_yaw,
              const double start_yaw_rate,
              const double end_yaw,
              const double end_yaw_rate,
              const std::vector<double>& time_allocation,
              const double max_yaw_rate = 1.0,
              const double max_yaw_acc = 1.0);

    bool optimize();

    /* Getters */
    inline Eigen::VectorXd getOptCtrlPts() { return x_; }
    inline BezierCurve getOptBezier() {
        calcBezierCurve();
        return *bc_;
    }

private:
    BezierCurve::Ptr bc_;

    int M_;    // number of segments
    int N_;    // order of the polynomial
    int DM_;   // dimension of the optimization problem
    double max_yaw_rate_;
    double max_yaw_acc_;

    std::vector<double> t_;            // time allocation
    Eigen::Vector4d init_, goal_;      // [yaw; yaw_rate]

    Eigen::MatrixXd Q_;   // cost matrix
    Eigen::MatrixXd A_;   // constraint matrix
    Eigen::VectorXd b_;   // bound vector
    Eigen::VectorXd ub_;  // upper bound vector
    Eigen::VectorXd lb_;  // lower bound vector
    Eigen::VectorXd x_;   // vector of control points

    Eigen::MatrixXd y2r_;  // yaw control points to rate control points
    Eigen::MatrixXd r2a_;  // rate control points to acceleration control points

    void calcCtrlPtsCvtMat();
    void calcMinAccCost();
    void calcBezierCurve();
    void addContinuityConstraints();
    void addDynamicalConstraints();
    void addBoundaryConstraints();

public:
    typedef std::shared_ptr<YawBezierOpt> Ptr;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace traj_opt

#endif  // _YAW_BEZIER_OPTIMIZER_H_ 