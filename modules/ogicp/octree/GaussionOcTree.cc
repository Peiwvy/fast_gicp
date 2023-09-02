#include "GaussionOcTree.h"

namespace octomap {

// node implementation  --------------------------------------
std::ostream& GaussionOcTreeNode::writeData(std::ostream& s) const {
  s.write((const char*)&value, sizeof(value));

  s.write((const char*)&gd.n, sizeof(GaussionDistribution::n));
  s.write((const char*)&gd.centroid, sizeof(GaussionDistribution::centroid));
  s.write((const char*)&gd.covariance_matrix, sizeof(GaussionDistribution::covariance_matrix));

  return s;
}

std::istream& GaussionOcTreeNode::readData(std::istream& s) {
  s.read((char*)&value, sizeof(value));

  s.read((char*)&gd.n, sizeof(GaussionDistribution::n));
  s.read((char*)&gd.centroid, sizeof(GaussionDistribution::centroid));
  s.read((char*)&gd.covariance_matrix, sizeof(GaussionDistribution::covariance_matrix));

  return s;
}

GaussionOcTreeNode::GaussionDistribution GaussionOcTreeNode::IncreUpdate(const GaussionOcTreeNode::GaussionDistribution A,
                                                                         const GaussionOcTreeNode::GaussionDistribution B) const {
  uint32_t        new_n        = A.n + B.n;
  Eigen::Vector3f new_centroid = (A.n * A.centroid + B.n * B.centroid) / new_n;
  Eigen::Vector3f mean_diff    = A.centroid - B.centroid;
  Eigen::Matrix3f new_covariance_matrix =
    (A.covariance_matrix * (A.n - 1) + B.covariance_matrix * (B.n - 1) + (mean_diff * mean_diff.transpose()) * A.n * B.n / new_n) / (new_n - 1);

  GaussionOcTreeNode::GaussionDistribution new_(new_n, new_centroid, new_covariance_matrix);

  return new_;
}

GaussionOcTreeNode::GaussionDistribution GaussionOcTreeNode::AdditiveUpdate(const GaussionOcTreeNode::GaussionDistribution A,
                                                                            const GaussionOcTreeNode::GaussionDistribution B) const {
  uint32_t        new_n                 = A.n + B.n;
  Eigen::Vector3f new_centroid          = (A.n * A.centroid + B.n * B.centroid) / new_n;
  Eigen::Matrix3f new_covariance_matrix = (A.n * A.covariance_matrix + B.n * B.covariance_matrix) / new_n;

  GaussionOcTreeNode::GaussionDistribution new_(new_n, new_centroid, new_covariance_matrix);

  return new_;
}

GaussionOcTreeNode::GaussionDistribution GaussionOcTreeNode::getChildGaussionDistribution() const {
  GaussionOcTreeNode::GaussionDistribution parents_ds;
  parents_ds.n                 = 0;
  parents_ds.centroid          = Eigen::Vector3f::Zero();
  parents_ds.covariance_matrix = Eigen::Matrix3f::Zero();

  if (children != NULL) {
    for (int i = 0; i < 8; i++) {
      GaussionOcTreeNode* child = static_cast<GaussionOcTreeNode*>(children[i]);

      if (child != NULL && child->isGaussionDistributionSet()) {
        auto child_ds = child->getGaussionDistribution();
        // child_ds must have data in this place.
        if (parents_ds.n == 0) {
          parents_ds = child_ds;
        } else if (child_ds.n == 0) {
          parents_ds = parents_ds;
        } else {
          auto new_node = IncreUpdate(child_ds, parents_ds);
          parents_ds    = new_node;
        }
      }
    }
  }

  return parents_ds;
}

void GaussionOcTreeNode::updateGaussionDistributionChildren() {
  gd = getChildGaussionDistribution();
}

// tree implementation  --------------------------------------
GaussionOcTree::GaussionOcTree(double in_resolution) : OccupancyOcTreeBase<GaussionOcTreeNode>(in_resolution) {
  GaussionOcTreeMemberInit.ensureLinking();
}

GaussionOcTreeNode*
GaussionOcTree::setNodeGaussionDistribution(const OcTreeKey& key, uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix) {
  GaussionOcTreeNode* node = search(key, 0);
  if (node != NULL) {
    node->setGaussionDistribution(n, centroid, covariance_matrix);
  }
  return node;
}

bool GaussionOcTree::pruneNode(GaussionOcTreeNode* node) {
  if (!isNodeCollapsible(node))
    return false;

  // set value to children's values (all assumed equal)
  node->copyData(*(getNodeChild(node, 0)));

  if (node->isGaussionDistributionSet())  // TODO check
    node->setGaussionDistribution(node->getChildGaussionDistribution());

  // delete children
  for (unsigned int i = 0; i < 8; i++) {
    deleteNodeChild(node, i);
  }
  delete[] node->children;
  node->children = NULL;

  return true;
}

bool GaussionOcTree::isNodeCollapsible(const GaussionOcTreeNode* node) const {
  // all children must exist, must not have children of
  // their own and have the same occupancy probability
  if (!nodeChildExists(node, 0))
    return false;

  const GaussionOcTreeNode* firstChild = getNodeChild(node, 0);
  if (nodeHasChildren(firstChild))
    return false;

  for (unsigned int i = 1; i < 8; i++) {
    // compare nodes only using their occupancy, ignoring GaussionDistribution
    // for pruning
    if (!nodeChildExists(node, i) || nodeHasChildren(getNodeChild(node, i)) || !(getNodeChild(node, i)->getValue() == firstChild->getValue()))
      return false;
  }

  return true;
}

GaussionOcTreeNode*
GaussionOcTree::averageNodeGaussionDistribution(const OcTreeKey& key, uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix) {
  GaussionOcTreeNode* node = search(key, 0);

  if (node != NULL) {
    if (node->isGaussionDistributionSet()) {
      GaussionOcTreeNode::GaussionDistribution added(n, centroid, covariance_matrix);
      // 这个地方的增量更新 应该是平均值
      auto new_ = node->AdditiveUpdate(node->getGaussionDistribution(), added);
      node->setGaussionDistribution(new_);
    } else {
      node->setGaussionDistribution(n, centroid, covariance_matrix);
    }
  }

  return node;
}

// // TODO
// GaussionOcTreeNode*
// GaussionOcTree::integrateNodeGaussionDistribution(const OcTreeKey& key,
// uint32_t n, Eigen::Vector3f centroid, Eigen::Matrix3f covariance_matrix) {
//   GaussionOcTreeNode* node = search(key);
//   if (node != 0) {
//     if (node->isGaussionDistributionSet()) {
//       GaussionOcTreeNode::GaussionDistribution prev_gd   =
//       node->getGaussionDistribution(); double node_prob =
//       node->getOccupancy(); uint32_t                                  new_r =
//       (uint32_t)((double)prev_gd.r * node_prob + (double)r * (0.99 -
//       node_prob)); uint32_t                                  new_g     =
//       (uint32_t)((double)prev_gd.g * node_prob + (double)g * (0.99 -
//       node_prob)); uint32_t                                  new_b     =
//       (uint32_t)((double)prev_gd.b * node_prob + (double)b * (0.99 -
//       node_prob)); node->setGaussionDistribution(n, centroid,
//       covariance_matrix);
//     } else {
//       node->setGaussionDistribution(n, centroid, covariance_matrix);
//     }
//   }
//   return node;
// }

void GaussionOcTree::updateInnerOccupancy() {
  this->updateInnerOccupancyRecurs(this->root, 0);
}

void GaussionOcTree::updateInnerOccupancyRecurs(GaussionOcTreeNode* node, unsigned int depth) {
  // only recurse and update for inner nodes:
  if (nodeHasChildren(node)) {
    // return early for last level:
    if (depth < this->tree_depth) {
      for (unsigned int i = 0; i < 8; i++) {
        if (nodeChildExists(node, i)) {
          updateInnerOccupancyRecurs(getNodeChild(node, i), depth + 1);
        }
      }
    }
    node->updateOccupancyChildren();
    node->updateGaussionDistributionChildren();
  }
}

std::ostream& operator<<(std::ostream& out, GaussionOcTreeNode::GaussionDistribution const& distribution) {
  return out << "cout: " << (unsigned int)distribution.n << ", mean: " << distribution.centroid.x() << " " << distribution.centroid.y() << " "
             << distribution.centroid.z() << " ,cov_matrix:  " << std::endl
             << (Eigen::Matrix3f)distribution.covariance_matrix;
}

GaussionOcTree::StaticMemberInitializer GaussionOcTree::GaussionOcTreeMemberInit;

}  // namespace octomap