//
// Created by spike on 19.10.20.
//

#include "Image.h"
#include "ImageConverter.h"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <list>
#include <cmath>
#include <algorithm>
#include "KDTree.h"
#include "Optimizer.h"
#include <chrono>
#include <atomic>

inline std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  auto res_time = std::chrono::high_resolution_clock::now();
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return res_time;
}

template<class D>
inline long long to_s(const D &d) {
  return std::chrono::duration_cast<std::chrono::seconds>(d).count();
}

struct cmpPoints {
    bool operator()(const point<double,3>& a, const point<double,3>& b) const {
      return a.get(0) < b.get(0);
    }
};

typedef std::map<point<double, 3>, std::list<double>, cmpPoints> hazeLineMap;

std::vector<point<double, 3>> getTessalation(const std::string& filename, int n_points)
{
  std::vector<point<double,3>> points;
  points.reserve(n_points);
  double a, b, c;

  std::ifstream infile(filename);
  while (infile >> a >> b >> c)
  {
    points.emplace_back(std::array<double, 3>{a,b,c});
  }

  return points;
}

template <typename It, typename T>
double get_std(It begin, It end, int size)
{
  T sum = std::accumulate(begin, end, 0.0);
  T mean = sum / size;

  std::vector<double> diff(size);
  std::transform(begin, end, diff.begin(), [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  return std::sqrt(sq_sum / size);
}

void createHazelines(Image& img, hazeLineMap& hazelines, std::vector<point<double,3>>& points, kdtree<double,3>& kdtree)
{
  for (auto pixel: img) {
    auto point = kdtree.nearest(pixel);

    points.push_back(point);
    auto r = pixel.get(0);

    auto search = hazelines.find(point);
    if (search != hazelines.end())
      search->second.push_back(r);
    else
      hazelines.insert(std::make_pair(point, std::list<double>{r}));
  }
}

std::map<point<double,3>, double, cmpPoints> getLinesMax(hazeLineMap& hazelines)
{
  std::map<point<double,3>, double, cmpPoints> hazeline_max;
  for (auto & hazeline : hazelines) {
    auto r = *std::max_element(hazeline.second.cbegin(), hazeline.second.cend());
    hazeline_max.insert(std::make_pair(hazeline.first, r));
  }
  return hazeline_max;
}

std::map<point<double,3>, double, cmpPoints> getLinesStDev(hazeLineMap& hazelines) {
  std::map<point<double, 3>, double, cmpPoints> hazeline_std;
  for (auto &hazeline : hazelines) {
    auto r_std = get_std<std::list<double>::iterator, double>(
            hazeline.second.begin(), hazeline.second.end(),
            hazeline.second.size());
    hazeline_std.insert(std::make_pair(hazeline.first, r_std));
  }
  return hazeline_std;
}


Eigen::MatrixXd mapToMatrix(std::vector<point<double,3>>& points,
                            std::map<point<double,3>, double, cmpPoints> map, int rows, int cols)
{
  Eigen::MatrixXd matrix(rows, cols);

  for (int row = 0; row < matrix.rows(); ++row) {
    for (int col = 0; col < matrix.cols(); ++col) {
      double val = map.find(points[col + row*cols])->second;
      matrix(row, col) = val;
    }
  }

  return matrix;
}



int main(int argc, char **argv)
{
  Image original_image("cityscape_input.png");
  auto img = original_image;
  auto& mat = img.imgMat;
  point<double,3> airlight{0.81, 0.81, 0.82};

  ImageConverter::shift(airlight, img);
  ImageConverter::convertToSpherical(img);

  auto tes = getTessalation("TR1000.txt", 1000);
  kdtree<double, 3> kdtree(tes.begin(), tes.end());

  std::map<point<double, 3>, std::list<double>, cmpPoints> hazelines;
  std::vector<point<double, 3>> points;

  createHazelines(img, hazelines, points, kdtree);
  auto map = getLinesMax(hazelines);
  auto radius_max = mapToMatrix(points, map, img.rows(), img.cols());

  auto transmission_est = mat[0];
  transmission_est = (transmission_est.array() / radius_max.array()).cwiseMax(0.1).cwiseMin(1).matrix();


  auto img_copy = original_image;
  ImageConverter::divide(img_copy, point<double,3>{airlight});
  auto& tlb_matrix = img_copy.getMatrix();
  auto& tlb_min = tlb_matrix[0].array().min(tlb_matrix[1].array().min(tlb_matrix[2].array()));
  Optimizer::RowMajorMatrixXd tlb =(-1*tlb_min+1).max(transmission_est.array()).matrix();

  auto hazeline_std = getLinesStDev(hazelines);

  Eigen::MatrixXd radius_std = mapToMatrix(points, map, img_copy.rows(), img_copy.cols());


  double lambda = 0.1;
  auto start = get_current_time_fenced();
  std::cout << radius_std.minCoeff() << "\t" << radius_std.maxCoeff() << std::endl;
  img_copy = original_image;
  auto transmission = Optimizer::wlsOptimization(tlb, radius_std, img_copy, lambda);

  std::cout << transmission.maxCoeff() << "\t" << transmission.minCoeff() << std::endl;
  auto dehazed = original_image;
  transmission = transmission.array().cwiseMax(0.2).cwiseMin(0.9).matrix();
  ImageConverter::dehaze(dehazed,
                         transmission,
                         0.1, 1.0, airlight);
//  img.write_png_file((char*)"output.png");
  original_image.write_png_file("original.png");
  dehazed.write_png_file((char*)"out.png");
  return 0;
}