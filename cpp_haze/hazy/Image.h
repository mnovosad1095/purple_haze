//
// Created by spike on 20.10.20.
//

#ifndef HAZY_IMAGE_H
#define HAZY_IMAGE_H

#include <png.h>
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
    png_byte color_type;
    png_byte bit_depth;

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

    ~Image() {}

    ImageMatrix& getMatrix()
    {
      return imgMat;
    }

    iterator begin() { return ImageIterator<point<double,3>>(this, 0,0); }
    iterator   end() { return ImageIterator<point<double,3>>(this, width,height); }

    void set_pixel(const png_byte* pixel, const int x,const int y)
    {
      for (int i = 0; i < 3; ++i) imgMat[i](x,y) =  (double)pixel[i] / 255.0;
    }

    void set_pixel(const point<double,3> point, const int x, const int y)
    {
      for (int i = 0; i < 3; ++i) imgMat[i](x,y) =  point.get(i);
    }

    point<double, 3> getPixel(int row, int col)
    {
      return point<double, 3> {imgMat[0](row,col), imgMat[1](row,col), imgMat[2](row,col)};
    }

    void create_matrix(png_bytep * row_pointers)
    {
      for (int i =0; i<3;++i) imgMat[i].resize(width, height);
      for (int y=0; y<height; ++y) {
        png_byte* row = row_pointers[y];
        for (int x=0; x<width; ++x) {
          png_byte* ptr = &(row[x*3]);
          set_pixel(ptr, x, y);
        }
      }
    }

    void matrixToRowPointers(png_bytep * row_pointers)
    {
      for (int y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (int x=0; x<width; x++) {
          png_byte* ptr = &(row[x*3]);
          for (int i = 0; i < 3; ++i)
            ptr[i] = (png_byte) (std::min(1.0,std::max(imgMat[i](x,y), 0.0))*255.0);
        }
      }
    }

    void setMatrix(const ImageMatrix& m)
    {
      imgMat = m;
    }

    int cols() const { return imgMat[0].cols(); }
    int rows() const { return imgMat[0].rows();}

    void read_png_file(const char* file_name)
    {
      char header[8];    // 8 is the maximum size that can be checked

      /* open file and test for it being a png */
      FILE *fp = fopen(file_name, "rb");
      if (!fp)
        abort_("[read_png_file] File %s could not be opened for reading", file_name);
      fread(header, 1, 8, fp);
      /* initialize stuff */
      auto png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

      if (!png_ptr)
        abort_("[read_png_file] png_create_read_struct failed");

      auto info_ptr = png_create_info_struct(png_ptr);
      if (!info_ptr)
        abort_("[read_png_file] png_create_info_struct failed");

      if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during init_io");

      png_init_io(png_ptr, fp);
      png_set_sig_bytes(png_ptr, 8);

      png_read_info(png_ptr, info_ptr);

      width = png_get_image_width(png_ptr, info_ptr);
      height = png_get_image_height(png_ptr, info_ptr);
      color_type = png_get_color_type(png_ptr, info_ptr);
      bit_depth = png_get_bit_depth(png_ptr, info_ptr);

      int number_of_passes = png_set_interlace_handling(png_ptr);
      png_read_update_info(png_ptr, info_ptr);


      /* read file */
      if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during read_image");

      png_bytep * row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
      for (int y=0; y<height; ++y)
        row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

      png_read_image(png_ptr, row_pointers);
      create_matrix(row_pointers);
      fclose(fp);

      for (int y=0; y<height; ++y)
        free(row_pointers[y]);
      free(row_pointers);
    }

    void write_png_file(char* file_name)
    {
      /* create file */
      FILE *fp = fopen(file_name, "wb");
      if (!fp)
        abort_("[write_png_file] File %s could not be opened for writing", file_name);

      /* initialize stuff */
      auto png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

      if (!png_ptr)
        abort_("[write_png_file] png_create_write_struct failed");

      auto info_ptr = png_create_info_struct(png_ptr);
      if (!info_ptr)
        abort_("[write_png_file] png_create_info_struct failed");

      if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during init_io");

      png_init_io(png_ptr, fp);


      /* write header */
      if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing header");

      png_set_IHDR(png_ptr, info_ptr, width, height,
                   bit_depth, color_type, PNG_INTERLACE_NONE,
                   PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

      png_write_info(png_ptr, info_ptr);

      /* write bytes */
      if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing bytes");

      auto * row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
      for (int y=0; y<height; ++y)
        row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));
      matrixToRowPointers(row_pointers);
      png_write_image(png_ptr, row_pointers);


      /* end write */
      if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during end of write");

      png_write_end(png_ptr, nullptr);

      /* cleanup heap allocation */
      for (int y=0; y<height; ++y)
        free(row_pointers[y]);
      free(row_pointers);

      fclose(fp);
    }

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
