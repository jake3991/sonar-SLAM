#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <algorithm>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

MatrixXb ca(const MatrixXf &img, int train_hs, int guard_hs, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float sum_train = 0;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if (std::abs(i - row) > guard_hs)
          sum_train += img(i, col);
      }
      ret(row, col) = img(row, col) > tau * sum_train / (2.0 * train_hs);
    }
  }
  return ret;
}

MatrixXb soca(const MatrixXf &img, int train_hs, int guard_hs, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float leading_sum = 0.0, lagging_sum = 0.0;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if ((i - row) > guard_hs)
          lagging_sum += img(i, col);
        else if ((i - row) < -guard_hs)
          leading_sum += img(i, col);
      }
      float sum_train = std::min(leading_sum, lagging_sum);
      ret(row, col) = img(row, col) > tau * sum_train / train_hs;
    }
  }
  return ret;
}

MatrixXb goca(const MatrixXf &img, int train_hs, int guard_hs, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float leading_sum = 0.0, lagging_sum = 0.0;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if ((i - row) > guard_hs)
          lagging_sum += img(i, col);
        else if ((i - row) < -guard_hs)
          leading_sum += img(i, col);
      }
      float sum_train = std::max(leading_sum, lagging_sum);
      ret(row, col) = img(row, col) > tau * sum_train / train_hs;
    }
  }
  return ret;
}

MatrixXb os(const MatrixXf &img, int train_hs, int guard_hs, int k, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float leading_sum = 0.0, lagging_sum = 0.0;
      std::vector<float> train;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if (std::abs(i - row) > guard_hs)
          train.push_back(img(i, col));
      }
      std::nth_element(train.begin(), train.begin() + k, train.end());
      ret(row, col) = img(row, col) > tau * train[k];
    }
  }
  return ret;
}

std::pair<MatrixXb, MatrixXf> ca2(const MatrixXf &img, int train_hs, int guard_hs, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());
  MatrixXf ret2 = MatrixXf::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float sum_train = 0;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if (std::abs(i - row) > guard_hs)
          sum_train += img(i, col);
      }
      ret(row, col) = img(row, col) > tau * sum_train / (2.0 * train_hs);
      ret2(row, col) = tau * sum_train / (2.0 * train_hs);
    }
  }
  return std::make_pair(ret, ret2);
}

std::pair<MatrixXb, MatrixXf> soca2(const MatrixXf &img, int train_hs, int guard_hs, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());
  MatrixXf ret2 = MatrixXf::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float leading_sum = 0.0, lagging_sum = 0.0;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if ((i - row) > guard_hs)
          lagging_sum += img(i, col);
        else if ((i - row) < -guard_hs)
          leading_sum += img(i, col);
      }
      float sum_train = std::min(leading_sum, lagging_sum);
      ret(row, col) = img(row, col) > tau * sum_train / train_hs;
      ret2(row, col) = tau * sum_train / train_hs;
    }
  }
  return std::make_pair(ret, ret2);
}

std::pair<MatrixXb, MatrixXf> goca2(const MatrixXf &img, int train_hs, int guard_hs, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());
  MatrixXf ret2 = MatrixXf::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float leading_sum = 0.0, lagging_sum = 0.0;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if ((i - row) > guard_hs)
          lagging_sum += img(i, col);
        else if ((i - row) < -guard_hs)
          leading_sum += img(i, col);
      }
      float sum_train = std::max(leading_sum, lagging_sum);
      ret(row, col) = img(row, col) > tau * sum_train / train_hs;
      ret2(row, col) = tau * sum_train / train_hs;
    }
  }
  return std::make_pair(ret, ret2);
}

std::pair<MatrixXb, MatrixXf> os2(const MatrixXf &img, int train_hs, int guard_hs, int k, double tau)
{
  MatrixXb ret = MatrixXb::Zero(img.rows(), img.cols());
  MatrixXf ret2 = MatrixXf::Zero(img.rows(), img.cols());

  for (int col = 0; col < img.cols(); ++col)
  {
    for (int row = train_hs + guard_hs; row < img.rows() - train_hs - guard_hs; ++row)
    {
      float leading_sum = 0.0, lagging_sum = 0.0;
      std::vector<float> train;
      for (int i = row - train_hs - guard_hs; i < row + train_hs + guard_hs + 1; ++i)
      {
        if (std::abs(i - row) > guard_hs)
          train.push_back(img(i, col));
      }
      std::nth_element(train.begin(), train.begin() + k, train.end());
      ret(row, col) = img(row, col) > tau * train[k];
      ret2(row, col) = tau * train[k];
    }
  }
  return std::make_pair(ret, ret2);
}

PYBIND11_MODULE(cfar, m)
{
  m.def("ca", &ca);
  m.def("soca", &soca);
  m.def("goca", &goca);
  m.def("os", &os);
  m.def("ca2", &ca2);
  m.def("soca2", &soca2);
  m.def("goca2", &goca2);
  m.def("os2", &os2);
}