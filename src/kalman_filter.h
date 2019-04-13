#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "Eigen/Dense"

class KalmanFilter {
 public:
  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
            Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict();

  /**
   * Updates the state by using standard Kalman Filter equations with y,
   * so this method can be used for both Update and UpdateEKF.
   * @param y The residual between actual measurement at k+1 and the predicted
   *          measurement
   */
  void UpdateWithY(const Eigen::VectorXd &y);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(const Eigen::VectorXd &z);

  /**
   * Updates the transition matrix F_ by delta t
   * @param dt The delta t (elapsed time since the last measurement)
   */
  void UpdateF(float dt) {
    F_(0, 2) = dt;
    F_(1, 3) = dt;
  }

  /**
   * Updates the process covariance matrix Q_ by delta t and process noise.
   * @param dt The delta t (elapsed time since the last measurement)
   * @param noise_ax The acceleration variance on x direction
   * @param noise_ay The acceleration variance on y direction
   */
  void UpdateQ(float dt, float noise_ax, float noise_ay);

  /**
   * Updates the Process covariance matrix R_ by R. LiDar and RaDar have
   * different R.
   * @param R The matrix R to update R_
   */
  void UpdateR(Eigen::MatrixXd R) {
    R_ = R;
  }

  /**
   * Updates the Measurement matrix H_ by H. LiDar and RaDar have different H.
   * @param H The matrix H to update H_
   */
  void UpdateH(Eigen::MatrixXd H) {
    H_ = H;
  }

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // process covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;
};

#endif // KALMAN_FILTER_H_
