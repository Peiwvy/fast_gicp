#include "ogicp/lsq_registration_depth.hpp"
#include "ogicp/impl/lsq_registration_impl_depth.hpp"

template class fast_ogicp::LsqRegistrationDepth<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_ogicp::LsqRegistrationDepth<pcl::PointXYZI, pcl::PointXYZI>;