#ifndef FAST_OGICP_GICP_SETTINGS_HPP
#define FAST_OGICP_GICP_SETTINGS_HPP

namespace fast_ogicp {

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, NORMALIZED_MAX, NORMALIZED_SUM, PLANE, FROBENIUS };

enum class NeighborSearchMethod { DIRECT27, DIRECT7, DIRECT1, /* supported on only VGICP_CUDA */ DIRECT_RADIUS };

enum class VoxelAccumulationMode { ADDITIVE, ADDITIVE_WEIGHTED, MULTIPLICATIVE };
}  // namespace fast_ogicp

#endif