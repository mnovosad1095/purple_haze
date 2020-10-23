//
// Created by spike on 21.10.20.
//

#include "Optimizer.h"
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/Eigen/SparseQR>
#include <eigen3/Eigen/OrderingMethods>

typedef std::vector<Eigen::Array<double, -1,-1, Eigen::RowMajor>> diags_vec;

template <typename Scalar>
Eigen::SparseMatrix<Scalar, Eigen::RowMajor> spdiags(const diags_vec& diags,
                                    const std::vector<int>& subs, size_t m, size_t n)
{
  Eigen::SparseMatrix<Scalar> A(m,n);

  typedef Eigen::Triplet<Scalar> T;
  std::vector<T> triplets;
  triplets.reserve(std::min(m,n)*subs.size());

  size_t i_min, i_max;
  for (int k=0; k<subs.size();++k) {
    i_min = std::max(0, -subs[k]);
    i_max = std::min(n-1, n - subs[k] - 1);
    auto& diag = diags[k];

    for(;i_min < i_max; ++i_min) {
      triplets.emplace_back(i_min, i_min+subs[k], diag(std::min(i_min, i_min+subs[k])));
    }
  }

  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

Eigen::SparseMatrix<double, Eigen::RowMajor> insert_column(const Eigen::Matrix<double, -1,-1, Eigen::RowMajor>& matrix)
{
  auto size = matrix.size();
  Eigen::SparseMatrix<double, Eigen::RowMajor> sm(size, size);
  typedef Eigen::Triplet<double> T;
  std::vector<T> triplets;
  triplets.reserve(size);
  auto& arr = matrix.array();

  for (size_t i =0; i < size; ++i)
  {
    triplets.emplace_back(i,0,arr(i));
  }
  sm.setFromTriplets(triplets.begin(), triplets.end());

  return sm;
}

Optimizer::RowMajorMatrixXd
Optimizer::wlsOptimization(Eigen::Matrix<double, -1,-1, Eigen::RowMajor>&tlb, Eigen::MatrixXd &radius_std, Image &guidance, double lambda)
{
  size_t k = guidance.rows()*guidance.cols();
//  size_t k = 9;
  double err = 0.00001;

  ImageConverter::convertToGrayScale(guidance);

  guidance.write_png_file((char *) "guidance.png");
  Eigen::Matrix<double, -1,-1, Eigen::RowMajor> dy = diffY(guidance.imgMat[0]);
  dy =( (dy.array().square() + err).pow(-1) * (-lambda) ).matrix();
  dy.row(dy.rows()-1).setZero();

  Eigen::Matrix<double, -1,-1, Eigen::RowMajor> dx = diffX(guidance.imgMat[0]);
  dx =( (dx.array().square() + err).pow(-1) * (-lambda) ).matrix();
  dx.col(dx.cols()-1).setZero();

  Eigen::SparseMatrix<double, Eigen::RowMajor>
  asmooth = spdiags<double>(diags_vec {dx.array(), dy.array()},
                        std::vector<int>{(int)-guidance.imgMat[0].rows(), -1}, k, k);

  dx = -(dx+dy);
  asmooth = asmooth + Eigen::SparseMatrix<double, Eigen::RowMajor>(asmooth.transpose().conjugate())
          + spdiags<double>(diags_vec {dx.array()}, std::vector<int> {0}, k, k);

  radius_std = radius_std.unaryExpr([a = radius_std.minCoeff()] (double x){return x-a;});
  radius_std /= radius_std.maxCoeff() + err;

  radius_std.row(0) = radius_std.row(0).unaryExpr([](double x) { return x > 0.6?x: 0.8; });

  auto inrow = tlb.colwise().minCoeff();
  auto tlb_row = tlb.row(0);
  auto radius_row = radius_std.row(0);

  tlb_row = Eigen::MatrixXd::NullaryExpr(tlb_row.rows(), tlb_row.cols(),[&tlb_row, &inrow, &radius_row] (Eigen::Index i)
          {return radius_row(i) == 0.8? inrow(i): tlb_row(i);});


  auto aData = spdiags<double>(diags_vec {tlb.array()}, std::vector<int>{0}, k,k);
  Eigen::SparseMatrix<double, Eigen::RowMajor> stlb = insert_column(tlb);
  Eigen::SparseVector<double> b = (aData*stlb).col(0);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>solver;

  solver.compute(aData+asmooth);
  auto solution = Optimizer::RowMajorMatrixXd (solver.solve(b));
  solution.resize(guidance.rows(), guidance.cols());
  return solution;
}

Eigen::MatrixXd Optimizer::diffY(Eigen::MatrixXd& matrix) {
  Eigen::Matrix<double, -1,-1, Eigen::RowMajor> differences(matrix.rows(), matrix.cols());
  long r = matrix.rows()-1;
  long c = matrix.cols();

  differences.row(r).setZero();
  differences.block(0,0,r, c) = matrix.block(1,0, r, c) -
                                                matrix.block(0,0, r, c);
  return differences;
}


Eigen::MatrixXd Optimizer::diffX(Eigen::MatrixXd& matrix) {
  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> differences(matrix.rows(), matrix.cols());
  long r = matrix.rows();
  long c = matrix.cols()-1;

  differences.col(c).setZero();
  differences.block(0,0,r, c) = matrix.block(0,1, r, c) -
                                matrix.block(0,0, r, c);
  return differences;
}
