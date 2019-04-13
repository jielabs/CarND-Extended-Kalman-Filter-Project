#include <cmath>
#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * DONE: predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateWithY(const VectorXd &y) {
  /**
   * DONE: update the state by using Kalman Filter equations
   */
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}


void KalmanFilter::Update(const VectorXd &z) {
  /**
   * DONE: update the state by using Kalman Filter equations
   */
  VectorXd y = z - H_ * x_;
  UpdateWithY(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * DONE: update the state by using Extended Kalman Filter equations
   */
  VectorXd z_pred(3);
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  float ro = sqrt(px * px + py * py);
  float theta = 0.0;
  if (px != 0 && py != 0) {
    theta = atan2(py, px);
  }
  float ro_dot = 0.0;
  if (ro > 1e-7) {
    ro_dot = (px * vx + py * vy) / ro;
  }
  z_pred << ro, theta, ro_dot;
  VectorXd y(3);
  y = z - z_pred;
  // Theta must be very small.
  if (y[1] >= M_PI) {
    y[1] = y[1] - 2 * M_PI;
  } else if (y[1] <= -1 * M_PI) {
    y[1] = y[1] + 2 * M_PI;
  }
  UpdateWithY(y);
}

// Calculate process covariance Q based on dt and noise of acceleration.
void KalmanFilter::UpdateQ(float dt, float noise_ax, float noise_ay) {
  float dt4 = pow(dt, 4) / 4.0;
  float dt3 = pow(dt, 3) / 2.0;
  float dt2 = pow(dt, 2);
  float ax2 = noise_ax;
  float ay2 = noise_ay;
  Q_ << dt4 * ax2, 0, dt3 * ax2, 0,
        0, dt4 * ay2, 0, dt3 * ay2,
        dt3 * ax2, 0, dt2 * ax2, 0,
        0, dt3 * ay2, 0, dt2 * ay2;
}

