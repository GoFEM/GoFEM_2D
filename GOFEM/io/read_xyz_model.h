#ifndef READ_TENSORGRID_MODEL_H
#define READ_TENSORGRID_MODEL_H

#include <vector>
#include <string>

void read_tensor_grid_model(const std::string file, std::vector<std::vector<double>>& cell_sizes,
                            std::vector<double>& origin, std::vector<unsigned> &material_id,
                            std::vector<unsigned> &parameter_id);

#endif // READ_TENSORGRID_MODEL_H
