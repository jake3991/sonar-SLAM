#include <pointmatcher/PointMatcher.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <fstream>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>

namespace py = pybind11;
typedef PointMatcher<float> PM;
typedef PM::Matrix Matrix;
typedef PM::IntMatrix IntMatrix;
typedef PM::DataPoints DP;
typedef std::shared_ptr<PM::DataPoints> DPPtr;

DPPtr fromEigen(const Matrix &mat)
{
  DP::Labels labels;
  labels.push_back(DP::Label("x", 1));
  labels.push_back(DP::Label("y", 1));
  if (mat.cols() == 3)
    labels.push_back(DP::Label("z", 1));
  labels.push_back(DP::Label("pad", 1));

  Matrix padded_mat = Matrix::Ones(labels.size(), mat.rows());
  padded_mat.block(0, 0, labels.size() - 1, mat.rows()) = mat.transpose();

  DPPtr pc(new DP(padded_mat, labels));
  return pc;
}

DPPtr fromEigen(const Matrix &mat, const Matrix &desc)
{
  DP::Labels labels;
  labels.push_back(DP::Label("x", 1));
  labels.push_back(DP::Label("y", 1));
  if (mat.cols() == 3)
    labels.push_back(DP::Label("z", 1));
  labels.push_back(DP::Label("pad", 1));

  Matrix padded_mat = Matrix::Ones(labels.size(), mat.rows());
  padded_mat.block(0, 0, labels.size() - 1, mat.rows()) = mat.transpose();

  DP::Labels desc_labels;
  for (int col = 0; col < desc.cols(); ++col)
    desc_labels.push_back(DP::Label("desc" + std::to_string(col), 1));

  DPPtr pc(new DP(padded_mat, labels, desc.transpose(), desc_labels));
  return pc;
}

Matrix remove_outlier(const Matrix &mat_in, double radius, int min_points)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>(mat_in.rows(), 1));
  for (int row = 0; row < mat_in.rows(); ++row)
  {
    cloud_in->at(row).x = mat_in(row, 0);
    cloud_in->at(row).y = mat_in(row, 1);
    if (mat_in.cols() == 3)
      cloud_in->at(row).z = mat_in(row, 2);
  }

  pcl::RadiusOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_in);
  sor.setRadiusSearch(radius);
  sor.setMinNeighborsInRadius(min_points);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
  sor.filter(*cloud_out);
  Matrix mat_out = cloud_out->getMatrixXfMap();
  return mat_out.topRows(mat_in.cols()).transpose();
}

Matrix density_filter(const Matrix &mat_in, int knn, float min_density, float max_density)
{
  if (mat_in.rows() == 0)
    return mat_in;

  PointMatcherSupport::Parametrizable::Parameters params1;
  params1["knn"] = std::to_string(knn);
  params1["keepNormals"] = "0";
  params1["keepDensities"] = "1";

  std::shared_ptr<PM::DataPointsFilter> filter =
      PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter", params1);
  DPPtr cloud_in = fromEigen(mat_in);
  filter->inPlaceFilter(*cloud_in);

  PointMatcherSupport::Parametrizable::Parameters params2;
  params2["minDensity"] = std::to_string(min_density);
  params2["maxDensity"] = std::to_string(max_density);

  filter = PM::get().DataPointsFilterRegistrar.create("MaxDensityDataPointsFilter", params2);
  filter->inPlaceFilter(*cloud_in);
  return cloud_in->features.topRows(mat_in.cols()).transpose();
}

std::pair<Matrix, Matrix> density_filter(const Matrix &mat_in, const Matrix &desc_in, int knn, float min_density,
                                         float max_density)
{
  if (mat_in.rows() == 0)
    return std::make_pair(mat_in, desc_in);

  PointMatcherSupport::Parametrizable::Parameters params1;
  params1["knn"] = std::to_string(knn);
  params1["keepNormals"] = "0";
  params1["keepDensities"] = "1";

  std::shared_ptr<PM::DataPointsFilter> filter =
      PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter", params1);
  DPPtr cloud_in = fromEigen(mat_in, desc_in);
  filter->inPlaceFilter(*cloud_in);

  PointMatcherSupport::Parametrizable::Parameters params2;
  params2["minDensity"] = std::to_string(min_density);
  params2["maxDensity"] = std::to_string(max_density);

  filter = PM::get().DataPointsFilterRegistrar.create("MaxDensityDataPointsFilter", params2);
  filter->inPlaceFilter(*cloud_in);

  Matrix mat_out = cloud_in->features.topRows(mat_in.cols()).transpose();
  Matrix desc_out = cloud_in->descriptors.transpose();
  return std::make_pair(mat_out, desc_out);
}

Matrix downsample(const Matrix &mat_in, float resolution)
{
  if (mat_in.rows() == 0)
    return mat_in;

  PointMatcherSupport::Parametrizable::Parameters params;
  params["maxSizeByNode"] = std::to_string(resolution);
  params["samplingMethod"] = "3";
  std::shared_ptr<PM::DataPointsFilter> filter =
      PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter", params);
  DPPtr cloud_in = fromEigen(mat_in);
  filter->inPlaceFilter(*cloud_in);
  return cloud_in->features.topRows(mat_in.cols()).transpose();
}

std::pair<Matrix, Matrix> downsample(const Matrix &mat_in, const Matrix &desc_in, float resolution)
{
  if (mat_in.rows() == 0)
    return std::make_pair(mat_in, desc_in);

  PointMatcherSupport::Parametrizable::Parameters params;
  params["maxSizeByNode"] = std::to_string(resolution);
  params["samplingMethod"] = "3";
  std::shared_ptr<PM::DataPointsFilter> filter =
      PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter", params);
  DPPtr cloud_in = fromEigen(mat_in, desc_in);
  filter->inPlaceFilter(*cloud_in);

  Matrix mat_out = cloud_in->features.topRows(mat_in.cols()).transpose();
  Matrix desc_out = cloud_in->descriptors.transpose();
  return std::make_pair(mat_out, desc_out);
}

std::pair<IntMatrix, Matrix> match(const Matrix &mat_ref, const Matrix &mat_in, int knn, float max_dist)
{
  PointMatcherSupport::Parametrizable::Parameters params;
  params["knn"] = std::to_string(knn);
  params["maxDist"] = std::to_string(max_dist);
  std::shared_ptr<PM::Matcher> matcher = PM::get().MatcherRegistrar.create("KDTreeMatcher", params);

  DPPtr cloud_ref = fromEigen(mat_ref);
  DPPtr cloud_in = fromEigen(mat_in);

  matcher->init(*cloud_ref);
  PM::Matches matches = matcher->findClosests(*cloud_in);
  return std::make_pair(matches.ids, matches.dists);
}

PYBIND11_MODULE(pcl, m)
{
  m.def("remove_outlier", &remove_outlier);
  m.def("density_filter", (Matrix(*)(const Matrix &, int, float, float)) & density_filter);
  m.def("density_filter",
        (std::pair<Matrix, Matrix>(*)(const Matrix &, const Matrix &, int, float, float)) & density_filter);
  m.def("downsample", (Matrix(*)(const Matrix &, float)) & downsample);
  m.def("downsample", (std::pair<Matrix, Matrix>(*)(const Matrix &, const Matrix &, float)) & downsample);
  m.def("match", &match);
  py::class_<PM::ICP>(m, "ICP")
      .def(py::init<>())
      .def("loadFromYaml",
           [](PM::ICP &icp, const std::string &filename) {
             std::ifstream ifs(filename);
             if (!ifs.is_open())
             {
               std::cout << "Failed to load " << filename << ". Use default configuration." << std::endl;
               icp.setDefault();
             }
             else
               icp.loadFromYaml(ifs);
           })
      .def("compute",
           [](PM::ICP &icp, const Matrix &source, const Matrix &target, const Matrix &guess) {
             DPPtr pc_source = fromEigen(source);
             DPPtr pc_target = fromEigen(target);
             Matrix T = guess;
             try
             {
               T = icp(*pc_source, *pc_target, guess);
             }
             catch (PM::ConvergenceError &e)
             {
               return std::make_pair(e.what(), T);
             }
             return std::make_pair("success", T);
           })
      .def("getCovariance", [](const PM::ICP &icp) { return icp.errorMinimizer->getCovariance(); });
}
