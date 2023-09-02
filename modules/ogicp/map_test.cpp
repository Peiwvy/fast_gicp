#include <iostream>
#include <string>

#include <pcl/console/time.h>  // TicToc
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
// #include <fast_gicp/gicp/fast_ogicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

// #include "../modules/octree/GaussionOcTree.h"
// #include <octomap/octomap.h>

typedef pcl::PointXYZ           PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;
bool next_scan      = false;

void print4x4Matrix(const Eigen::Matrix4d& matrix) {
  printf("Rotation matrix :\n");
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
  printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
  printf("Translation vector :\n");
  printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing) {
  if (event.getKeySym() == "space" && event.keyDown()) {
    next_iteration = true;
  }

  if (event.getKeySym() == "s" && event.keyDown()) {
    std::cout << "press button S " << std::endl;
    next_scan = true;
  }
}

void readScan(const std::string folder_path, const int scan_num, PointCloudT::Ptr& read_cloud) {
  read_cloud->clear();
  pcl::io::loadPCDFile(folder_path + "/test_pcd" + std::to_string(scan_num) + ".pcd", *read_cloud);
  std::cout << "Read： " << folder_path + "/test_pcd" + std::to_string(scan_num) + ".pcd" << std::endl;
}

int main(int argc, char* argv[]) {
  int    scan_num   = 25;
  size_t iterations = 1;

  PointCloudT::Ptr scan_in(new PointCloudT);    // inpput scan
  PointCloudT::Ptr scan_icp(new PointCloudT);   // registrationed ICP
  PointCloudT::Ptr cloud_map(new PointCloudT);  // input ot map

  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

  pcl::console::TicToc time;

  std::string path_octomap = argv[1];  // 输入 ot 路径
  std::string path_scan    = argv[2];  // 输入 scan 路径
  readScan(path_scan, scan_num++, scan_in);

  // 读进来数据 首先进行平移旋转 到初始位置
  if (argc == 3) {
    double degree               = 60;                   // 正 逆时针
    double theta                = (degree)*M_PI / 180;  // The angle of rotation in radians
    transformation_matrix(0, 0) = std::cos(theta);
    transformation_matrix(0, 1) = -sin(theta);
    transformation_matrix(1, 0) = sin(theta);
    transformation_matrix(1, 1) = std::cos(theta);

    transformation_matrix(0, 3) = 8;
    transformation_matrix(1, 3) = 6;
    std::cout << std::endl << "[" << transformation_matrix(0, 3) << " " << transformation_matrix(1, 3) << " " << degree << "]" << std::endl;

    print4x4Matrix(transformation_matrix);

    pcl::transformPointCloud(*scan_in, *scan_in, transformation_matrix);
    *scan_icp = *scan_in;
  } else {
    return (-1);
  }

  time.tic();

  // method : OGICP Setting
  fast_gicp::FastOGICP<PointT, PointT> registration;
  registration.setResolution(1.0);
  registration.setNumThreads(4);
  registration.setOctreeBuildMethod(1);  // 1 先落在栅格 再计算协方差 2. 先计算每个点协方差 再计算栅格
  registration.setDepthChange(true);     // 用于设置是否开启 层切换机制 默认未开启
  registration.setFirstSearchDepth(14);  // 用于设置初始搜索层数

  registration.setInputSource(scan_icp);            // input randomly to avoid
  registration.setInputTarget(scan_icp);            // input randomly to avoid
  registration.setInputTargetOctree(path_octomap);  // path_octomap 作为一个参数输入

  // 读取 octomap
  std::shared_ptr<octomap::GaussionOcTree> octree_ptr_;
  octomap::AbstractOcTree*                 read_tree = octomap::AbstractOcTree::read(path_octomap);
  octomap::GaussionOcTree*                 readtree  = dynamic_cast<octomap::GaussionOcTree*>(read_tree);
  readtree->updateInnerOccupancy();
  octree_ptr_.reset(readtree);
  for (auto it = octree_ptr_->begin_leafs(15), end = octree_ptr_->end_leafs(); it != end; ++it) {
    auto gd = it->getGaussionDistribution();
    cloud_map->push_back(pcl::PointXYZ(gd.centroid(0), gd.centroid(1), gd.centroid(2)));
  }

  // Visualization
  pcl::visualization::PCLVisualizer viewer("ICP demo");
  // Create two vertically separated viewports
  int v1(0);
  int v2(1);
  viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
  viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

  // The color we will be using
  float bckgr_gray_level = 0.0;  // Black
  float txt_gray_lvl     = 1.0 - bckgr_gray_level;

  // Original point cloud is white
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(
    cloud_map, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl);
  viewer.addPointCloud(cloud_map, cloud_in_color_h, "cloud_in_v1", v1);
  viewer.addPointCloud(cloud_map, cloud_in_color_h, "cloud_in_v2", v2);

  // Transformed point cloud is green
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h(scan_in, 20, 180, 20);
  viewer.addPointCloud(scan_in, cloud_tr_color_h, "cloud_tr_v1", v1);

  // ICP aligned point cloud is red
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h(scan_icp, 180, 20, 20);
  viewer.addPointCloud(scan_icp, cloud_icp_color_h, "cloud_icp_v2", v2);

  // Adding text descriptions in each viewport
  viewer.addText(
    "White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
  viewer.addText("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

  std::stringstream ss;
  ss << iterations;
  std::string iterations_cnt = "ICP iterations = " + ss.str();
  viewer.addText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

  // Set background color
  viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
  viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

  // Set camera position and orientation
  viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
  viewer.setSize(1280, 1024);  // Visualiser  window size

  // Register keyboard callback :
  viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);

  // Display the visualiser
  while (!viewer.wasStopped()) {
    viewer.spinOnce();

    // The user pressed "space" :
    if (next_iteration) {
      // The Iterative Closest Point algorithm
      time.tic();
      // transformation_matrix used as predict pose
      // registration.align(*scan_icp, transformation_matrix);

      registration.align(*scan_icp);
      std::cout << "Applied ICP iteration in " << time.toc() << " ms" << std::endl;

      std::cout << "Iiterations: " << iterations++ << std::endl;
      transformation_matrix *= registration.getFinalTransformation().cast<double>();

      print4x4Matrix(transformation_matrix);

      ss.str("");
      ss << iterations;
      std::string iterations_cnt = "registration iterations = " + ss.str();
      viewer.updateText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt");
      viewer.updatePointCloud(scan_icp, cloud_icp_color_h, "cloud_icp_v2");
    }

    // The user pressed "s" :
    if (next_scan) {
      readScan(path_scan, scan_num++, scan_in);
      pcl::transformPointCloud(*scan_in, *scan_in, transformation_matrix);
      viewer.updatePointCloud(scan_in, cloud_tr_color_h, "cloud_tr_v1");

      std::string new_msg = "Read new scan " + std::to_string(scan_num);
      scan_icp            = scan_in;
      registration.setInputSource(scan_icp);
      viewer.updateText(new_msg, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt");
      viewer.updatePointCloud(scan_icp, cloud_icp_color_h, "cloud_icp_v2");
      iterations = 0;
    }

    next_scan      = false;
    next_iteration = false;
  }
  return (0);
}