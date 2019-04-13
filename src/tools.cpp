#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * DONE: Calculate the RMSE here.
   */

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if ((estimations.size() == 0) || (estimations.size() != ground_truth.size())) {
      std::cout << "Wrong size." << std::endl;
  }

  // DONE: accumulate squared residuals
  VectorXd residuals(4);
  for (unsigned int i=0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse/estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * DONE:
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float pxy = px*px + py*py;

  // check division by zero
  if (pxy < 1e-7) {
      std::cout << "CalculateJacobian() - Error - Division by Zero" << std::endl;
      return Hj;
  }

  // compute the Jacobian matrix
  Hj << px/sqrt(pxy), py/sqrt(pxy), 0, 0,
        -py/pxy, px/pxy, 0, 0,
        py*(vx*py-vy*px)/pow(pxy, 1.5), px*(vy*px-vx*py)/pow(pxy, 1.5), px/sqrt(pxy), py/sqrt(pxy);

  return Hj;
}
