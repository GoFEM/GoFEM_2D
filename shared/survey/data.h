/**
 * CMAINV, Geophysical data inversion using Covariance Matrix Adaptation
 * Copyright (c) 2015 Alexander Grayver <agrayver@erdw.ethz.ch>
 *
 * This file is part of CMAINV.
 *
 * CMAINV is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CMAINV is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CMAINV. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _DATA_H
#define _DATA_H

#include <vector>
#include <istream>
#include <unordered_map>
#include <unordered_set>
#include <set>

#include "common.h"
#include "survey/datum.h"
#include "survey/receiver.h"
#include "survey/physical_source.h"

const unsigned any_index = 444555222;

/*
 * Triple index to locate data easily using
 * frequency#, transmitter#, receiver#
 */
struct IndexKey
{
  IndexKey(unsigned fidx, unsigned tidx, unsigned ridx):
    _fidx(fidx), _sidx(tidx), _ridx(ridx)
  {}

  bool operator==(const IndexKey &other) const
  {
    bool equal = true;
    equal &= _fidx == other._fidx;
    equal &= _sidx == other._sidx;
    equal &= _ridx == other._ridx;

    return equal;
  }

  unsigned _fidx, _sidx, _ridx;
};

/*
 * To be able to use index above in unordered map,
 * we need to define a hash function for it
 */
struct IndexHash
{
  std::size_t operator()(const IndexKey& k) const
  {
    return ((std::hash<unsigned>()(k._fidx)
           ^(std::hash<unsigned>()(k._sidx) << 1)) >> 1)
           ^(std::hash<unsigned>()(k._ridx) << 1);
  }
};

struct DataTypeHash
{
  std::size_t operator()(const RealDataType& k) const
  {
    return std::hash<unsigned>()(k);
  }
};

typedef std::unordered_map<RealDataType, Datum, DataTypeHash> DataMap;
typedef std::unordered_map<IndexKey, DataMap, IndexHash> DataTable;
typedef std::set<std::pair<SurveyMethod, ForwardType>> MethodCalculatorSet;

/*
 * This class provides interface to the survey layout
 */
class SurveyData
{
public:
  SurveyData();

  unsigned n_instances() const;
  unsigned n_sources() const;
  unsigned n_receivers() const;
  unsigned n_data() const;

  const PhysicalSourcePtr& physical_source(unsigned tx_index) const;
  const PhysicalSourcePtr& physical_source(const std::string &name) const;

  const Receiver& receiver(unsigned index) const;
  const Receiver& receiver(const std::string &name) const;

  // Return instance (time or instance) by index
  double instance(unsigned index) const;

  // Sets all data values to zero
  void zero_data();

  // Sets status of the method data to a predefined value
  void set_data_status(const ForwardType &forward_type, const DatumStatus &status);

  // Add synthetic noise to data and set error floors
  void contaminate_with_noise(double relative_noise, double relative_floor,
                              double absolute_floor, double freq_factor);

  // Set data of the given types at specified instance, transmitter and receiver
  void set_data_at(unsigned instance_index, unsigned source_index, unsigned receiver_index,
                   const std::vector<RealDataType> &types, const dvector &values);

  // Retrieves data at specified instance, transmitter and receiver
  void data_at(unsigned instance_index, unsigned source_index, unsigned receiver_index,
               std::vector<RealDataType> &types, dvector &values) const;

  void data_at(unsigned instance_index, unsigned source_index, unsigned receiver_index,
               std::vector<RealDataType> &types, dvector &values, dvector &errors) const;

  // Retrieves data types at specified instance, transmitter and receiver
  void data_types_at(unsigned instance_index, unsigned source_index,
                     unsigned receiver_index, std::vector<RealDataType> &types) const;

  // Retrieves receiver indices for specified instance and transmitter
  void receivers_at(unsigned instance_index, unsigned source_index, std::unordered_set<unsigned> &indices) const;

  // Retrieves instances (frequencies or times) for a given source and receiver
  void instances_at(unsigned source_index, unsigned receiver_index, std::unordered_set<unsigned> &indices) const;

  // Returns indices of transmitters and instance at which data really exist
  // instance means frequency or time depending on the method under consideration
  const std::set<unsigned> &used_sources(const SurveyMethod method) const;
  const std::set<unsigned> &used_instances(const SurveyMethod method) const;
  const std::set<unsigned> &used_receivers(const SurveyMethod method) const;

  const std::set<unsigned> &used_sources(const ForwardType type) const;
  const std::set<unsigned> &used_instances(const ForwardType method) const;
  const std::set<unsigned> &used_receivers(const ForwardType method) const;

  // Calculates ||(this - other)/err||_l where other and l are given as input parameters
  // the method also returns misfit normalized by a number of particular method data
  void misfit(const SurveyData &other, const double &norm_power,
              double &misfit, double &misfit_normalized) const;

  // Reads and writes data
  friend std::istream& operator>> (std::istream &in, SurveyData &data);
  friend std::ostream& operator<< (std::ostream &in, SurveyData &data);

  // Although data format is flexible, sometimes more
  // stupid and plain output is needed.
  void write_plain(const std::string &filename) const;

  void write_plain(std::ofstream &ofs) const;

  MethodCalculatorSet get_methods_list() const;

  void frequency_source_pairs(const ForwardType &type,
                              std::vector<std::pair<double, std::string>> &pairs) const;

  // Sets linear indexing of data using the following ordering:
  // Method <- Instance (frequency or time) <- Source <- Receiver
  void create_linear_index();

private:
  void read_receivers(const std::string &file_name);
  void read_sources(const std::string &file_name);

private:
  std::string srcs_file, instances_file, recvs_file;

  std::vector<double> instances;
  std::vector<PhysicalSourcePtr> physical_sources;
  std::vector<Receiver> receivers;

  // Indices of transmitters and instances (frequencies or times)
  // which one can find in the data table according to the method
  typedef std::map<SurveyMethod, std::set<unsigned>> MethodIndexMap;
  MethodIndexMap source_indices_per_method,
                 receiver_indices_per_method,
                 instance_indices_per_method;

  // Indices of transmitters and instances (frequencies or times) which one
  // can find in the data table according to the forward calculator type
  typedef std::map<ForwardType, std::set<unsigned>> CalculatorIndexMap;
  CalculatorIndexMap source_indices_per_calculator,
                     receiver_indices_per_calculator,
                     instance_indices_per_calculator;

  std::map<std::string, unsigned> receiver_index_map,
                                  source_index_map;

  std::map<std::pair<SurveyMethod, ForwardType>, unsigned> data_per_method;

  unsigned n_real_data;

  // Note the key here is a triple of indices: (freq#, trx#, rx#)
  DataTable data_table;

  static std::set<unsigned> empty_set;
};

#endif // _DATA_H
