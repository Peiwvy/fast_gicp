#ifndef OCTOMAP_GAUSSION_OCTREE_H
#define OCTOMAP_GAUSSION_OCTREE_H

#include <octomap/OcTreeNode.h>
#include <octomap/OccupancyOcTreeBase.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

namespace octomap {

// forward declaraton for "friend"
class GaussionOcTree;

// node definition
class GaussionOcTreeNode : public OcTreeNode {
 public:
  friend class GaussionOcTree;  // needs access to node children (inherited)

  class GaussionDistribution {
   public:
    GaussionDistribution() : n(0), centroid(Eigen::Vector3f::Zero()), covariance_matrix(Eigen::Matrix3f::Zero()) {}
    GaussionDistribution(uint32_t _n, Eigen::Vector3f _centroid, Eigen::Matrix3f _covariance_matrix)
      : n(_n), centroid(_centroid), covariance_matrix(_covariance_matrix) {}
    inline bool operator==(const GaussionDistribution& other) const {
      return (n == other.n && centroid == other.centroid && covariance_matrix == other.covariance_matrix);
    }
    inline bool operator!=(const GaussionDistribution& other) const {
      return (n != other.n || centroid != other.centroid || covariance_matrix != other.covariance_matrix);
    }

    uint32_t        n;
    Eigen::Vector3f centroid;
    Eigen::Matrix3f covariance_matrix;
  };

 public:
  GaussionOcTreeNode() : OcTreeNode() {}

  GaussionOcTreeNode(const GaussionOcTreeNode& rhs) : OcTreeNode(rhs), gd(rhs.gd) {}

  bool operator==(const GaussionOcTreeNode& rhs) const {
    return (rhs.value == value && rhs.gd == gd);
  }

  void copyData(const GaussionOcTreeNode& from) {
    OcTreeNode::copyData(from);
    this->gd = from.getGaussionDistribution();
  }

  inline GaussionDistribution getGaussionDistribution() const {
    return gd;
  }
  inline void setGaussionDistribution(GaussionDistribution c) {
    this->gd = c;
  }
  inline void setGaussionDistribution(uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix) {
    this->gd = GaussionDistribution(n, centroid, covariance_matrix);
  }

  GaussionDistribution& getGaussionDistribution() {
    return gd;
  }

  inline bool isGaussionDistributionSet() const {
    return ((gd.n != 0));
  }

  GaussionOcTreeNode::GaussionDistribution IncreUpdate(const GaussionDistribution A, const GaussionDistribution B) const;
  GaussionOcTreeNode::GaussionDistribution AdditiveUpdate(const GaussionDistribution A, const GaussionDistribution B) const;

  void updateGaussionDistributionChildren();

  GaussionOcTreeNode::GaussionDistribution getChildGaussionDistribution() const;

  // file I/O
  std::istream& readData(std::istream& s);
  std::ostream& writeData(std::ostream& s) const;

 protected:
  GaussionDistribution gd;
};

// tree definition
class GaussionOcTree : public OccupancyOcTreeBase<GaussionOcTreeNode> {
 public:
  /// Default constructor, sets resolution of leafs
  GaussionOcTree(double resolution);

  /// virtual constructor: creates a new object of same type
  /// (Covariant return type requires an up-to-date compiler)
  GaussionOcTree* create() const {
    return new GaussionOcTree(resolution);
  }

  std::string getTreeType() const {
    return "GaussionOcTree";
  }

  /**
   * Prunes a node when it is collapsible. This overloaded
   * version only considers the node occupancy for pruning,
   * different GaussionDistributions of child nodes are ignored.
   * @return true if pruning was successful
   */
  virtual bool pruneNode(GaussionOcTreeNode* node);

  virtual bool isNodeCollapsible(const GaussionOcTreeNode* node) const;

  // set node GaussionDistribution at given key or coordinate. Replaces previous
  // GaussionDistribution.
  GaussionOcTreeNode* setNodeGaussionDistribution(const OcTreeKey& key, uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix);

  GaussionOcTreeNode*
  setNodeGaussionDistribution(float x, float y, float z, uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix) {
    OcTreeKey key;
    if (!this->coordToKeyChecked(point3d(x, y, z), key))
      return NULL;
    return setNodeGaussionDistribution(key, n, centroid, covariance_matrix);
  }

  // integrate GaussionDistribution measurement at given key or coordinate.
  // Average with previous GaussionDistribution
  GaussionOcTreeNode* averageNodeGaussionDistribution(const OcTreeKey& key, uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix);

  GaussionOcTreeNode*
  averageNodeGaussionDistribution(float x, float y, float z, uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix) {
    OcTreeKey key;
    if (!this->coordToKeyChecked(point3d(x, y, z), key))
      return NULL;
    return averageNodeGaussionDistribution(key, n, centroid, covariance_matrix);
  }

  // integrate GaussionDistribution measurement at given key or coordinate.
  // Average with previous GaussionDistribution
  // GaussionOcTreeNode* integrateNodeGaussionDistribution(
  //     const OcTreeKey& key, uint32_t n, Eigen::Vector3f centroid,
  //     Eigen::Matrix3f covariance_matrix);

  // GaussionOcTreeNode* integrateNodeGaussionDistribution(
  //     float x, float y, float z, uint32_t n, Eigen::Vector3f centroid,
  //     Eigen::Matrix3f covariance_matrix) {
  //   OcTreeKey key;
  //   if (!this->coordToKeyChecked(point3d(x, y, z), key)) return NULL;
  //   return integrateNodeGaussionDistribution(key, n, centroid,
  //                                            covariance_matrix);
  // }

  // update inner nodes, sets GaussionDistribution to average child
  // GaussionDistribution
  void updateInnerOccupancy();

 protected:
  void updateInnerOccupancyRecurs(GaussionOcTreeNode* node, unsigned int depth);

  /**
   * Static member object which ensures that this OcTree's prototype
   * ends up in the classIDMapping only once. You need this as a
   * static member in any derived octree class in order to read .ot
   * files through the AbstractOcTree factory. You should also call
   * ensureLinking() once from the constructor.
   */
  class StaticMemberInitializer {
   public:
    StaticMemberInitializer() {
      GaussionOcTree* tree = new GaussionOcTree(0.1);
      tree->clearKeyRays();
      AbstractOcTree::registerTreeType(tree);
    }

    /**
     * Dummy function to ensure that MSVC does not drop the
     * StaticMemberInitializer, causing this tree failing to register.
     * Needs to be called from the constructor of this octree.
     */
    void ensureLinking(){};
  };
  /// static member to ensure static initialization (only once)
  static StaticMemberInitializer GaussionOcTreeMemberInit;
};

//! user friendly output in format (n centroid covariance_matrix)
std::ostream& operator<<(std::ostream& out, GaussionOcTreeNode::GaussionDistribution const& c);

}  // namespace octomap

#endif