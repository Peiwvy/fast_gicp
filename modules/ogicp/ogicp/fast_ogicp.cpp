#include "ogicp/fast_ogicp.hpp"
#include "ogicp/impl/fast_ogicp_impl.hpp"

template class fast_ogicp::FastOGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_ogicp::FastOGICP<pcl::PointXYZI, pcl::PointXYZI>;