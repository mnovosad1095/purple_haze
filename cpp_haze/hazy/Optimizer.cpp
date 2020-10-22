//
// Created by spike on 21.10.20.
//

#include "Optimizer.h"

Eigen::MatrixXd
Optimizer::wlsOptimization(Eigen::MatrixXd &tlb, Eigen::MatrixXd &radius_std, Image &guidance, double lambda)
{
  double err = 0.00001;
  ImageConverter::convertToGrayScale(guidance);
//  guidance.write_png_file((char*)"output2.png");

  return Eigen::MatrixXd();
}
