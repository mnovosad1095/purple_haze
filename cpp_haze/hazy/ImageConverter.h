//
// Created by spike on 20.10.20.
//

#ifndef HAZY_IMAGECONVERTER_H
#define HAZY_IMAGECONVERTER_H
#include "Image.h"
class ImageConverter{
private:
    static void arctan(Eigen::MatrixXd& y, const Eigen::MatrixXd& x)
    {
      for (int j = 0; j < y.cols(); ++j) {
        for (int i = 0; i < y.rows(); ++i) {
          y(i,j) = atan2(y(i,j), x(i,j));
        }
      }
    }

public:
    static void shift(const point<double,3> pixel, Image& image)
    {
      auto& mat = image.getMatrix();
      for (int i = 0; i < 3; ++i)
      {
        mat[i] = mat[i].unaryExpr([a = pixel.get(i)] (double x) {return x-a;});
      }
    }

    static void dehaze(Image& image, Eigen::Matrix<double, -1,-1, Eigen::RowMajor>& t,
                       double tmin, double leave_haze, const point<double, 3>& airlight)
    {
      for (int r = 0; r < image.rows(); ++r) {
        for (int c = 0; c < image.cols(); ++c) {
          auto pixel = image.getPixel(r,c);
          std::array<double,3> coords{};
          auto trans = t(r,c);
          for (int i=0;i<3;++i) coords[i] = (pixel.get(i) - (1.0 - trans*leave_haze)*airlight.get(i))
                                            /std::max(trans, tmin);
          image.set_pixel(point<double,3>(coords), r,c);

        }

      }
    }

    static void convertToSpherical(Image& image)
    {
      ImageMatrix& mat = image.getMatrix();
      auto red = mat[0];
      auto green = mat[1];
      auto blue = mat[2];
      mat[0] = (red.array().square() + green.array().square() + blue.array().square())
              .cwiseSqrt()
              .matrix();
      mat[0] /= mat[0].maxCoeff();

      arctan(mat[2], red);
      mat[2] /= std::fabs(mat[2].maxCoeff());

      mat[1] = (red.array().square() + blue.array().square()).cwiseSqrt().matrix();
      arctan(mat[1], green);
      mat[1] /= std::fabs(mat[1].maxCoeff());
    }

    static void convertToGrayScale(Image& image)
    {
      std::cout << image.end().curRow<< "\t" << image.end().curCol << std::endl;
      for (auto it = image.begin(); it != image.end(); ++it)
      {
        auto& pixel = *it;
        double wb = 0.33*pixel.get(0) + 0.34*pixel.get(1) + 0.33*pixel.get(2);
        image.set_pixel(point<double,3>{wb,wb,wb}, it.curRow, it.curCol);
      }
    }

    static void divide(Image& image,point<double,3> point)
    {
      auto& imgMat = image.getMatrix();
      for (int i=0; i<3; ++i) imgMat[i] / point.get(i);
    }
};
#endif //HAZY_IMAGECONVERTER_H
