//
// Created by spike on 21.10.20.
//

#ifndef HAZY_OPTIMIZER_H
#define HAZY_OPTIMIZER_H

#include "Image.h"
#include "ImageConverter.h"
#include <eigen3/Eigen/Dense>

class Optimizer {
public:
    static Eigen::MatrixXd wlsOptimization(Eigen::MatrixXd& tlb, Eigen::MatrixXd& radius_std,
                                    Image& guidance, double lambda);

};


#endif //HAZY_OPTIMIZER_H
