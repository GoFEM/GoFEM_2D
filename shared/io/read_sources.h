#ifndef READ_SOURCES_H
#define READ_SOURCES_H

#include <vector>
#include <string>

#include "survey/physical_source.h"

std::vector<PhysicalSourcePtr> read_sources(const std::string sources_file);

#endif // READ_SOURCES_H
