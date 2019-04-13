#include "FusionEKF.h"
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * DONE: Finish initializing the FusionEKF.
   * DONE: Set the process and measurement noises
   */
  // Initialize x_in, P_in, F_in, H_in, R_in, Q_in.
  VectorXd x(4);
  MatrixXd P(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;
  MatrixXd F(4, 4);
  F << 1, 0, 1, 0,
       0, 1, 0, 1,
       0, 0, 1, 0,
       0, 0, 0, 1;
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  noise_ax_ = 9.0;
  noise_ay_ = 9.0;
  MatrixXd Q(4, 4);

  ekf_.Init(x, P, F, H_laser_, R_laser_, Q);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

VectorXd FusionEKF::GetXFromRadarMeasurement(
    const MeasurementPackage &measurement_pack) {
  float ro = measurement_pack.raw_measurements_[0];
  float theta = measurement_pack.raw_measurements_[1];
  float ro_dot = measurement_pack.raw_measurements_[2];
  float px = ro * cos(theta);
  float py = ro * sin(theta);
  float vx = ro_dot * cos(theta);
  float vy = ro_dot * sin(theta);
  VectorXd x(4);
  x << px, py, vx, vy;
  return x;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  if ((dt < 0) || (dt > 100000000)) {
    // If we switch from Dataset 1 to Dataset 2 in the simulator, the timestamp
    // could be wrong, so we need to reinitialize.
    dt = 0.05;
    is_initialized_ = false;
  }
  previous_timestamp_ = measurement_pack.timestamp_;

  if (!is_initialized_) {
    /**
     * DONE: Initialize the state ekf_.x_ with the first measurement.
     * DONE: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // DONE: Convert radar from polar to cartesian coordinates
      //         and initialize state.

      ekf_.x_ = GetXFromRadarMeasurement(measurement_pack);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // DONE: Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0,
                 0;
    }
    // Assume the time interval is 50 millisecond.
    ekf_.UpdateQ(dt, noise_ax_, noise_ay_);

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  /**
   * Prediction
   */

  /**
   * DONE: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * DONE: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  ekf_.UpdateF(dt);
  ekf_.UpdateQ(dt, noise_ax_, noise_ay_);
  ekf_.Predict();

  /**
   * Update
   */

  /**
   * DONE:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // DONE: Radar updates
    Hj_ = tools_.CalculateJacobian(ekf_.x_);
    ekf_.UpdateH(Hj_);
    ekf_.UpdateR(R_radar_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // DONE: Laser updates
    ekf_.UpdateH(H_laser_);
    ekf_.UpdateR(R_laser_);
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
