//
// Created by spike on 21.10.20.
//

#ifndef HAZY_OPTIMIZER_H
#define HAZY_OPTIMIZER_H

#include "Image.h"
#include "ImageConverter.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

class Optimizer {
private:
public:
    typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> RowMajorMatrixXd;
    static RowMajorMatrixXd wlsOptimization(Eigen::Matrix<double, -1,-1, Eigen::RowMajor>& tlb , Eigen::MatrixXd& radius_std,
                                    Image& guidance, double lambda);
    static Eigen::MatrixXd diffY(Eigen::MatrixXd& matrix);
    static Eigen::MatrixXd diffX(Eigen::MatrixXd& matrix);
};


#endif //HAZY_OPTIMIZER_H
