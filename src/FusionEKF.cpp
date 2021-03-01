#include "FusionEKF.h"
#include <iostream>
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
FusionEKF::FusionEKF() 
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  //create 4D state vector
  ekf_.x_ = VectorXd(4);
  
  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  ekf_.P_ = MatrixXd(4,4);
  ekf_.H_ = MatrixXd(2,4);
  ekf_.R_ = MatrixXd(2,2);
  ekf_.F_ = MatrixXd(4,4);
  ekf_.Q_ = MatrixXd(4,4);
  
  //initial transition matrix
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
  
  //state covariance matrix
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
  
  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  noise_ax = 9;
  noise_ay = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
  /**
   * Initialization
   */
  if (!is_initialized_) 
  {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ << measurement_pack.raw_measurements_(0), 
               measurement_pack.raw_measurements_(1), 
               0, 
               0;
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
      float ro = measurement_pack.raw_measurements_(0);
      float theta = measurement_pack.raw_measurements_(1);
      float ro_dot = measurement_pack.raw_measurements_(2);
      
      float px = cos(theta) * ro;
      float py = sin(theta) * ro;
      float vx = cos(theta) * ro_dot;
      float vy = sin(theta) * ro_dot;
      
      ekf_.x_ << px, py, vx, vy;
    }
    
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  
  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  
  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
             0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
             dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
             0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  
  // predict
  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
  {
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    
  } 
  else 
  {
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
