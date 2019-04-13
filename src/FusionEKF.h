#ifndef FusionEKF_H_
#define FusionEKF_H_

#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "kalman_filter.h"
#include "measurement_package.h"
#include "tools.h"

class FusionEKF {
 public:
  /**
   * Constructor.
   */
  FusionEKF();

  /**
   * Destructor.
   */
  virtual ~FusionEKF();

  /**
   * Run the whole flow of the Kalman Filter from here.
   */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
   * Kalman Filter update and prediction math lives in here.
   */
  KalmanFilter ekf_;

 private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  // tool object used to compute Jacobian and RMSE
  Tools tools_;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd Hj_;  // Jacobian Matrix for Radar.
  float noise_ax_;
  float noise_ay_;

  /**
   * Convert radar from polar to cartesian coordinates (px, py, vx, vy)
   */
  Eigen::VectorXd GetXFromRadarMeasurement(
      const MeasurementPackage &measurement_pack);
};

#endif // FusionEKF_H_
