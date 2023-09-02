#include "../GaussionOcTree.h"

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>

#define maxdepth 16   // unit: layer
#define resolution 1  // unit: m

int main(int argc, char** argv) {
  std::string octomap_name = "simple_tree_gussion.ot";

  // Part1: Read origin point cloud
  Eigen::Vector4f                     centroid;
  Eigen::Matrix3f                     covariance_matrix;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);

  std::cout << " --------------- "
            << " initial         "
            << " --------------- " << std::endl;

  if (argc == 2) {
    std::cout << "use input data" << std::endl;

    if (pcl::io::loadPCDFile(argv[1], *cloud_in)) {
      std::cerr << "failed to open " << argv[1] << std::endl;
      return 1;
    }

    pcl::computeMeanAndCovarianceMatrix(*cloud_in, covariance_matrix, centroid);

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 3)           = -centroid(0);
    transform(1, 3)           = -centroid(1);
    transform(2, 3)           = -centroid(2);

    pcl::transformPointCloud(*cloud_in, *cloud_in, transform);
  } else {
    std::cout << "use builtin data" << std::endl;
    cloud_in->push_back(pcl::PointXYZ(1, 1.5, 0));
    cloud_in->push_back(pcl::PointXYZ(1, 0.5, 0));
    cloud_in->push_back(pcl::PointXYZ(0.5, 1, 0));
    cloud_in->push_back(pcl::PointXYZ(1.5, 1, 0));

    cloud_in->push_back(pcl::PointXYZ(3, 1.5, 0));
    cloud_in->push_back(pcl::PointXYZ(3, 0.5, 0));
    cloud_in->push_back(pcl::PointXYZ(2.5, 1, 0));
    cloud_in->push_back(pcl::PointXYZ(3.5, 1, 0));

    cloud_in->push_back(pcl::PointXYZ(1, 2.5, 0));
    cloud_in->push_back(pcl::PointXYZ(1, 3.5, 0));
    cloud_in->push_back(pcl::PointXYZ(0.5, 3, 0));
    cloud_in->push_back(pcl::PointXYZ(1.5, 3, 0));

    cloud_in->push_back(pcl::PointXYZ(3, 2.5, 0));
    cloud_in->push_back(pcl::PointXYZ(3, 3.5, 0));
    cloud_in->push_back(pcl::PointXYZ(2.5, 3, 0));
    cloud_in->push_back(pcl::PointXYZ(3.5, 3, 0));
  }

  std::cout << "Input " << cloud_in->size() << " pts. " << std::endl;

  // Part2: Construct Gaussion Octomap
  std::cout << " --------------- "
            << " compute tree    "
            << " --------------- " << std::endl;

  octomap::GaussionOcTree                                                                 save_tree(resolution);
  std::unordered_multimap<octomap::OcTreeKey, pcl::PointXYZ, octomap::OcTreeKey::KeyHash> unorderedMultiMap;
  std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>                     set;

  for (auto p : (*cloud_in).points) {
    auto key = save_tree.coordToKey(p.x, p.y, p.z, 16);
    unorderedMultiMap.emplace(key, p);
    set.emplace(key);
  }

  for (auto iter = set.begin(); iter != set.end(); ++iter) {
    auto key   = *iter;
    auto range = unorderedMultiMap.equal_range(key);
    int  i     = 0;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (auto it = range.first; it != range.second; ++it) {
      cloud.push_back(it->second);
      i++;
    }

    if (cloud.size() == 1)
      continue;
    pcl::computeMeanAndCovarianceMatrix(cloud, covariance_matrix, centroid);
    covariance_matrix = covariance_matrix * i / (i - 1);
    auto n            = save_tree.updateNode(key, true);
    n->setGaussionDistribution(i, centroid.head<3>(), covariance_matrix);
  }

  save_tree.updateInnerOccupancy();

  std::cout << " --------------- "
            << " itr test before "
            << " --------------- " << std::endl;

  std::cout << "tree size: " << save_tree.size() << std::endl;

  int write_count_leaf_[maxdepth] = {0};

  for (int ite = 0; ite < maxdepth; ite++) {
    bool firstin_ = true;
    for (auto it = save_tree.begin_leafs(ite), end = save_tree.end_leafs(); it != end; ++it) {
      if (firstin_) {
        std::cout << " ------- " << it.getDepth() << "/" << save_tree.getTreeDepth() << " layer ------- " << std::endl;
        std::cout << " coordinate: " << it.getCoordinate() << "  size: " << it.getSize() << std::endl;
        firstin_ = false;
      }

      // if (ite == 0) {  // 打印最低层的所有分布
      //   std::cout << "第" << write_count_leaf_[ite] << "个分布， "
      //             << " " << it->getGaussionDistribution() << std::endl;
      // }

      // if (ite == maxdepth - 1) {  // 打印次低层的所有分布
      //   std::cout << "第" << write_count_leaf_[ite] << "个分布， "
      //             << " " << it->getGaussionDistribution() << std::endl;
      // }

      // if (ite == 1) { // 打印最高分布
      //   std::cout << "第" << write_count_leaf_[ite] << "个分布， "
      //             << " " << it->getGaussionDistribution() << std::endl;
      // }
      write_count_leaf_[ite]++;
    }

    std::cout << " num of leafs :: " << write_count_leaf_[ite] << std::endl << std::endl;
  }

  // Part3: Save Gaussion Octomap
  save_tree.write(octomap_name);

  // Part4: Read Gaussion Octomap
  cloud_in->clear();

  octomap::AbstractOcTree* read_tree = octomap::AbstractOcTree::read(octomap_name);
  octomap::GaussionOcTree* readtree  = dynamic_cast<octomap::GaussionOcTree*>(read_tree);

  std::cout << " --------------- "
            << " itr test  after "
            << " --------------- " << std::endl;

  std::cout << "tree size: " << readtree->size() << std::endl;

  int read_count_leaf_[maxdepth] = {0};
  for (int ite = 0; ite < maxdepth; ite++) {
    bool firstin_ = true;
    for (auto it = readtree->begin_leafs(ite), end = readtree->end_leafs(); it != end; ++it) {
      if (firstin_) {
        std::cout << " ------- " << it.getDepth() << "/" << readtree->getTreeDepth() << " layer ------- " << std::endl;
        std::cout << " coordinate: " << it.getCoordinate() << "  size: " << it.getSize() << std::endl;
        firstin_ = false;
      }

      // if (ite == 0) {  // 打印最低层的所有分布
      //   std::cout << "第" << read_count_leaf_[ite] << "个分布， "
      //             << " " << it->getGaussionDistribution() << std::endl;
      // }

      // if (ite == maxdepth - 1) {  // 打印次低层的所有分布
      //   std::cout << "第" << read_count_leaf_[ite] << "个分布， "
      //             << " " << it->getGaussionDistribution() << std::endl;
      // }

      // if (ite == 1) {
      //   std::cout << "第" << read_count_leaf_[ite] << "个分布， "
      //             << " " << it->getGaussionDistribution() << std::endl;
      // }
      read_count_leaf_[ite]++;
    }

    std::cout << " num of leafs :: " << read_count_leaf_[ite] << std::endl << std::endl;
  }

  std::cout << "read file " << octomap_name << " done" << std::endl << std::endl;
}
