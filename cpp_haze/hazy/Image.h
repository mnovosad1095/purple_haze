//
// Created by spike on 20.10.20.
//

#ifndef HAZY_IMAGE_H
#define HAZY_IMAGE_H

#include "KDTree.h"
#include <iterator>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

typedef std::vector<Eigen::MatrixXd> ImageMatrix;

class Image
{
private:
    int width, height;
    unsigned char color_type;
    unsigned char bit_depth;

    static void abort_(const char * s, ...)
    {
      std::cerr<< s << std::endl;

    }

public:
    ImageMatrix imgMat;

      template<typename ValueType>
      class ImageIterator: public std::iterator<std::input_iterator_tag, point<double,3>>
      {
          friend class Image;
          friend class ImageConverter;
      protected:
          ImageIterator(Image* img, int row, int col);

      public:

          ImageIterator(const ImageIterator &it) ;

          bool operator!=(ImageIterator const& other) const ;
          bool operator==(ImageIterator const& other) const ;
          typename ImageIterator::reference operator*() const;

          ImageIterator& operator++();

      protected:
          Image* img;
          int curRow;
          int curCol;
          point<double,3>* ptr;
          point<double,3>  p{};
      };

    typedef ImageIterator<point<double,3>> iterator;

    Image() = delete;
    explicit Image(const std::string& filename) : imgMat(3)
    {
      read_png_file(filename.c_str());
    }

    ~Image() = default;

    ImageMatrix& getMatrix()
    {
      return imgMat;
    }

    iterator begin() { return ImageIterator<point<double,3>>(this, 0,0); }
    iterator   end() { return ImageIterator<point<double,3>>(this, width,height); }

    void set_pixel(const unsigned char * pixel, const int x,const int y)
    {
      for (int i = 0; i < 3; ++i) imgMat[i](x,y) =  (double)pixel[i] / 255.0;
    }

    void set_pixel(const point<double,3> point, const int x, const int y)
    {
      for (int i = 0; i < 3; ++i) imgMat[i](x,y) =  point.get(i);
    }

    point<double, 3> getPixel(int row, int col) const
    {
      return point<double, 3> {imgMat[0](row,col), imgMat[1](row,col), imgMat[2](row,col)};
    }

    void create_matrix(unsigned char** row_pointers)
    {
      for (int i =0; i<3;++i) imgMat[i].resize(width, height);
      std::cout << "resized" << std::endl;
      for (int y=0; y<height; ++y) {
        unsigned char * row = row_pointers[y];
        for (int x=0; x<width; ++x) {
          unsigned char * ptr = &(row[x*3]);
          set_pixel(ptr, x, y);
        }
      }
    }

    void matrixToRowPointers(unsigned char** row_pointers)
    {
      for (int y=0; y<height; y++) {
        unsigned char* row = row_pointers[y];
        for (int x=0; x<width; x++) {
          unsigned char* ptr = &(row[x*3]);
          for (int i = 0; i < 3; ++i)
            ptr[i] = (unsigned char) (std::min(1.0,std::max(imgMat[i](x,y), 0.0))*255.0);
        }
      }
    }

    void setMatrix(const ImageMatrix& m)
    {
      imgMat = m;
    }

    int cols() const { return imgMat[0].cols(); }
    int rows() const { return imgMat[0].rows();}

    void read_png_file(const std::string& file_name);

    void write_png_file(const std::string& file_name);

};

template<typename ValueType>
Image::ImageIterator<ValueType>::ImageIterator(Image* img, int row, int col) : img(img), curRow(row), curCol(col) {
  if (row == img->rows() && col == img->cols())
    p = point<double,3>({0,0,0});
  else
    p = img->getPixel(curRow, curCol);
  ptr = &p;
}

template<typename ValueType>
Image::ImageIterator<ValueType>::ImageIterator(const ImageIterator &it) {
  img = it.img;
  curCol = it.curCol;
  curRow = it.curRow;
  p = it.p;
  ptr = it.ptr;
}
template<typename ValueType>
bool Image::ImageIterator<ValueType>::operator!=(ImageIterator const& other) const
{
  return curRow != other.curRow || curCol != other.curCol;
}

template<typename ValueType>
bool Image::ImageIterator<ValueType>::operator==(ImageIterator const& other) const { return curRow == other.curRow && curCol == other.curCol;}

template<typename ValueType>
typename Image::ImageIterator<ValueType>::reference Image::ImageIterator<ValueType>::operator*() const {
  return *ptr;
}

template<typename ValueType>
Image::ImageIterator<ValueType>& Image::ImageIterator<ValueType>::operator++() {
  curCol++;

  if (curCol > img->cols()-1) {
    curCol = 0;
    curRow++;
  }
  if (curRow == img->rows()) {
    curCol = img->cols();
    p = point<double,3> ({0,0,0});
  } else
    p = img->getPixel(curRow, curCol);
  ptr = &p;
  return *this;
}

#endif //HAZY_IMAGE_H
