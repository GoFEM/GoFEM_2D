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
#include "data.h"

#include <stdexcept>
#include <limits>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>

#include "read_list.h"

#include "survey/dipole_source.h"
#include "survey/spherical_harmonics_source.h"
#include "survey/external_source.h"

std::set<unsigned> SurveyData::empty_set = std::set<unsigned>();

bool get_next_line(std::istream &in, std::string &line)
{
  while(!in.eof())
  {
    std::getline(in, line);

    // Trim string
    line.erase(0, line.find_first_not_of(" \t\r\n"));

    // Skip empty lines and comments
    if (line.length() < 1 || line[0] == '!' || line[0] == '%' || line[0] == '#')
      continue;

    return true;
  }

  return false;
}

std::istream& operator>> (std::istream &in, SurveyData &data)
{
  std::string line, line_us;

  data.n_real_data = 0;

  while(!in.eof())
  {
    get_next_line(in, line);
    line_us = line;

    std::transform(line_us.begin(), line_us.end(), line_us.begin(), ::toupper);

    // Parse sections of the file
    if(line_us.find("TRANSMITTERS:") != std::string::npos ||
       line_us.find("SOURCES:") != std::string::npos)
    {
      data.srcs_file = trim(line.substr(line.find(':') + 1));
      if(data.srcs_file.length() > 1)
        data.read_sources(data.srcs_file);
      continue;
    }

    if(line_us.find("FREQUENCIES:") != std::string::npos ||
       line_us.find("INSTANCES:") != std::string::npos ||
       line_us.find("TIMES:") != std::string::npos)
    {
      data.instances_file = trim(line.substr(line.find(':') + 1));
      data.instances = read_list<double> (data.instances_file);
      continue;
    }

    if(line_us.find("RECEIVERS:") != std::string::npos)
    {
      data.recvs_file = trim(line.substr(line.find(':') + 1));
      if(data.recvs_file.length() > 1)
        data.read_receivers(data.recvs_file);
      continue;
    }

    if(line_us.find("DATA:") != std::string::npos)
    {
      std::stringstream ss(line.substr(line.find(':') + 1));
      std::string method_name, calculator_name;
      unsigned n_method_data;
      ss >> n_method_data >> method_name >> calculator_name;

      auto itm = method_type_conversion.find(method_name);
      if(itm == method_type_conversion.end())
        throw std::runtime_error("Unknown method name " + method_name);

      auto itc = calculator_type_conversion.find(calculator_name);
      if(itc == calculator_type_conversion.end())
        throw std::runtime_error("Unknown calculator name " + calculator_name);

      const SurveyMethod &method_type = itm->second;
      const ForwardType &calculator_type = itc->second;

      if(data.data_per_method.find(std::make_pair(method_type, calculator_type)) != data.data_per_method.end())
        throw std::runtime_error("You defined combination " + method_name + " " + calculator_name + " more than once. This is not allowed.");

      data.data_per_method[std::make_pair(method_type, calculator_type)] = n_method_data;

      if(data.data_table.size() == 0)
        data.data_table.reserve(n_method_data);

      Datum d;
      unsigned fidx, ridx, tidx;
      std::string type_str, rec_str, src_str;

      for(unsigned i = 0; i < n_method_data; ++i)
      {
        get_next_line(in, line);
        std::stringstream ss(line);

        type_str = "";
        ss >> type_str >> fidx >> src_str >> rec_str >> d.value >> d.stderr;

        d.type = Datum::convert_string_to_type(type_str);
        d.method = method_type;
        d.calculator_type = calculator_type;
        d.status = DatumObserved;

        if(istrcompare(rec_str, "none"))
        {
          ridx = any_index;
        }
        else
        {
          auto itr = data.receiver_index_map.find(rec_str);
          if(itr == data.receiver_index_map.end())
            throw std::runtime_error("No receiver " + rec_str + " found in the receiver file.");
          ridx = itr->second;
        }

        if(istrcompare(src_str, "none"))
        {
          tidx = any_index;
        }
        else
        {
          auto its = data.source_index_map.find(src_str);
          if(its == data.source_index_map.end())
            throw std::runtime_error("No source " + src_str + " found in the source file.");
          tidx = its->second;
        }

        if(method_data_types_table[method_type].find(d.type) ==
           method_data_types_table[method_type].end())
          throw std::runtime_error("Type " + type_str + " us not supported within method " + method_name);

        data.data_table[IndexKey(fidx - 1, tidx, ridx)].insert(std::make_pair(d.type, d));

        data.instance_indices_per_method[method_type].insert(fidx - 1);
        data.receiver_indices_per_method[method_type].insert(ridx);
        data.source_indices_per_method[method_type].insert(tidx);

        data.instance_indices_per_calculator[calculator_type].insert(fidx - 1);
        data.receiver_indices_per_calculator[calculator_type].insert(ridx);
        data.source_indices_per_calculator[calculator_type].insert(tidx);

        ++data.n_real_data;
      }

      continue;
    }
  }

  return in;
}

std::ostream& operator<< (std::ostream &os, SurveyData &data)
{
  os << std::setprecision(10);

  os << "Transmitters: " << data.srcs_file << "\n";
  os << "Instances: " << data.instances_file << "\n";
  os << "Receivers: " << data.recvs_file << "\n";

  auto methods = data.get_methods_list();
  for(auto &p: methods)
  {
    const SurveyMethod &method_type = p.first;
    const ForwardType &calculator_type = p.second;

    os << "# Number of data\tMethod\tForward calculator\n";
    os << "Data: " << data.data_per_method[std::make_pair(method_type, calculator_type)] << "\t"
                   << type_method_conversion[method_type] << "\t"
                   << type_calculator_conversion[calculator_type] << "\n";
    os << "#	DataType	Instance#	SrcName	RecName	Data	Std_Error\n";
    for(auto& data_pair: data.data_table)
    {
      const DataMap &data_map = data_pair.second;
      const IndexKey &key = data_pair.first;

      for(DataMap::const_iterator itd = data_map.begin(); itd != data_map.end(); ++itd)
      {
        const Datum &datum = itd->second;

        if(method_data_types_table[method_type].find(datum.type) ==
           method_data_types_table[method_type].end())
          continue;

        if(datum.method != method_type || datum.calculator_type != calculator_type)
          continue;

        os << Datum::convert_type_to_string(datum.type) << "\t"
           << key._fidx + 1 << "\t"
           << (key._sidx == any_index ? "none" : data.physical_sources[key._sidx]->get_name()) << "\t"
           << (key._ridx == any_index ? "none" : data.receivers[key._ridx].get_name()) << "\t"
           << datum.value << "\t"
           << datum.stderr << "\n";
      }
    }
  }

  return os;
}

SurveyData::SurveyData()
{}

unsigned SurveyData::n_instances() const
{
  return instances.size();
}

unsigned SurveyData::n_sources() const
{
  return physical_sources.size();
}

unsigned SurveyData::n_receivers() const
{
  return receivers.size();
}

unsigned SurveyData::n_data() const
{
  return n_real_data;
}

const PhysicalSourcePtr &SurveyData::physical_source(unsigned tx_index) const
{
  if(tx_index >= physical_sources.size())
    throw std::out_of_range("You tried to access transmitter with index " + std::to_string(tx_index) +
                            " whereas there are only " + std::to_string(physical_sources.size()) + " transmitters.");

  return physical_sources[tx_index];
}

const PhysicalSourcePtr &SurveyData::physical_source(const std::string &name) const
{
  for(auto &src: physical_sources)
    if(name == src->get_name())
      return src;

  throw std::runtime_error("Cannot find source " + name);
}

double SurveyData::instance(unsigned index) const
{
  if(index >= instances.size())
    throw std::out_of_range("You tried to access frequency with index " + std::to_string(index) +
                            " whereas there are only " + std::to_string(instances.size()) + " frequencies.");

  return instances[index];
}

const Receiver &SurveyData::receiver(unsigned index) const
{
  if(index >= receivers.size())
    throw std::out_of_range("You tried to access receiver with index " + std::to_string(index) +
                            " whereas there are only " + std::to_string(receivers.size()) + " receivers.");

  return receivers[index];
}

const Receiver &SurveyData::receiver(const std::string &name) const
{
  for(auto &rec: receivers)
    if(name == rec.get_name())
      return rec;
}

void SurveyData::zero_data()
{
  for(auto &p: data_table)
  {
    DataTable::mapped_type &data = p.second;
    for(auto &d: data)
    {
      Datum &datum = d.second;
      datum.value = 0.;
      datum.status = DatumEmpty;
    }
  }
}

void SurveyData::contaminate_with_noise(double relative_noise, double relative_floor,
                                        double absolute_floor, double freq_factor)
{
  // Seed with a real random value
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::normal_distribution<double> distribution(0., 1.);

  dvector frequency_scaling(instances.size(), 1.);

  if(freq_factor != 1.)
  {
    auto minmax_it = std::minmax_element (instances.begin(), instances.end());

    const double dx = log10(*minmax_it.second)-log10(*minmax_it.first);
    const double df = 1-freq_factor;

    for(size_t i = 0; i < instances.size(); ++i)
    {
      frequency_scaling[i] = 1. + df/dx*log10(instances[i])-df/dx*log10(*minmax_it.second);
      //std::cout << i << "\t" << frequency_scaling[i] << std::endl;
    }
  }

  for(auto &p: data_table)
  {
    const DataTable::key_type &key = p.first;
    DataTable::mapped_type &data = p.second;
    for(auto &d: data)
    {
      Datum &datum = d.second;
      datum.stderr = absolute_floor;

      if(relative_floor > 0.)
      {
//        if(Datum::convert_type_to_string(datum.type).find("Phs") != std::string::npos)
//          datum.stderr = relative_floor * frequency_scaling[key._fidx] * 180. / M_PI;
//        else
          datum.stderr += fabs(datum.value * relative_floor * frequency_scaling[key._fidx]);
      }

      if(relative_noise > 0.)
      {
//        if(Datum::convert_type_to_string(datum.type).find("Phs") != std::string::npos)
//          datum.value += relative_noise * distribution(generator) * 180. / M_PI;
//        else
          datum.value += fabs(datum.value * relative_noise) * distribution(generator);
      }
    }
  }
}

void SurveyData::set_data_at(unsigned instance_index, unsigned source_index, unsigned receiver_index,
                             const std::vector<RealDataType> &types, const dvector &values)
{
  DataTable::iterator it =
      data_table.find(IndexKey(instance_index, source_index, receiver_index));

  if(it != data_table.end())
  {
    DataMap &datum_map = it->second;

    for(size_t i = 0; i < types.size(); ++i)
    {
      DataMap::iterator itd = datum_map.find(types[i]);
      if(itd != datum_map.end())
      {
        Datum &datum = itd->second;
        datum.value = values[i];
        datum.status = DatumPredicted;
      }
    }
  }
  else
    throw std::runtime_error("You have tried to set non-existent data.");
}

void SurveyData::data_at(unsigned instance_index, unsigned source_index, unsigned receiver_index,
                         std::vector<RealDataType> &types, dvector &values) const
{
  dvector errors;
  data_at(instance_index, source_index, receiver_index, types, values, errors);
}

void SurveyData::data_at(unsigned instance_index, unsigned source_index, unsigned receiver_index,
                         std::vector<RealDataType> &types, dvector &values, dvector &errors) const
{
  DataTable::const_iterator it =
      data_table.find(IndexKey(instance_index, source_index, receiver_index));

  types.clear();
  values.clear();
  errors.clear();

  if(it != data_table.end())
  {
    const DataMap &data_map = it->second;
    for(const auto &d: data_map)
    {
      types.push_back(d.first);
      values.push_back(d.second.value);
      errors.push_back(d.second.stderr);
    }
  }
}

void SurveyData::data_types_at(unsigned instance_index, unsigned source_index,
                               unsigned receiver_index, std::vector<RealDataType> &types) const
{
  DataTable::const_iterator it =
      data_table.find(IndexKey(instance_index, source_index, receiver_index));

  types.clear();

  if(it != data_table.end())
  {
    const DataMap &data_map = it->second;
    for(const auto &d: data_map)
      types.push_back(d.first);
  }
}

void SurveyData::receivers_at(unsigned instance_index, unsigned source_index,
                              std::unordered_set<unsigned> &indices) const
{
  indices.clear();

  for(const auto &p: data_table)
  {
    const IndexKey &key = p.first;
    if(key._fidx == instance_index && key._sidx == source_index)
      indices.insert(key._ridx);
  }
}

void SurveyData::instances_at(unsigned source_index, unsigned receiver_index,
                              std::unordered_set<unsigned> &indices) const
{
  indices.clear();

  for(const auto &p: data_table)
  {
    const IndexKey &key = p.first;
    if(key._ridx == receiver_index && key._sidx == source_index)
      indices.insert(key._fidx);
  }
}

const std::set<unsigned> &SurveyData::used_sources(const SurveyMethod method) const
{
  auto it = source_indices_per_method.find(method);
  if(it == source_indices_per_method.end())
    return empty_set;
  else
    return it->second;
}

const std::set<unsigned> &SurveyData::used_instances(const SurveyMethod method) const
{
  auto it = instance_indices_per_method.find(method);
  if(it == instance_indices_per_method.end())
    return empty_set;
  else
    return it->second;
}

const std::set<unsigned> &SurveyData::used_receivers(const SurveyMethod method) const
{
  auto it = receiver_indices_per_method.find(method);
  if(it == receiver_indices_per_method.end())
    return empty_set;
  else
    return it->second;
}

const std::set<unsigned> &SurveyData::used_sources(const ForwardType type) const
{
  auto it = source_indices_per_calculator.find(type);
  if(it == source_indices_per_calculator.end())
    return empty_set;
  else
    return it->second;
}

const std::set<unsigned> &SurveyData::used_instances(const ForwardType type) const
{
  auto it = instance_indices_per_calculator.find(type);
  if(it == instance_indices_per_calculator.end())
    return empty_set;
  else
    return it->second;
}

const std::set<unsigned> &SurveyData::used_receivers(const ForwardType type) const
{
  auto it = receiver_indices_per_calculator.find(type);
  if(it == receiver_indices_per_calculator.end())
    return empty_set;
  else
    return it->second;
}

void SurveyData::misfit(const SurveyData &other, const double &norm_power,
                        double &misfit, double &misfit_normalized) const
{
  if(data_per_method != other.data_per_method)
    throw std::runtime_error("You are trying to compare survey data objects with different amount of data!");

  misfit = 0;
  misfit_normalized = 0;

  std::map<std::pair<SurveyMethod, ForwardType>, double> factors, mmap;

  for(auto p: data_per_method)
    factors[p.first] = 1. / p.second;

  for(DataTable::const_iterator it = data_table.begin(); it != data_table.end(); ++it)
  {
    const DataMap &data_map = it->second;

    DataTable::const_iterator it_other = other.data_table.find(it->first);
    if(it_other == other.data_table.end())
      throw std::runtime_error("Cannot find data in the given survey to compare with.");

    const DataMap &data_map_other = it_other->second;
    for(DataMap::const_iterator itd = data_map.begin(); itd != data_map.end(); ++itd)
    {
      const Datum &datum = itd->second;

      DataMap::const_iterator itd_other = data_map_other.find(datum.type);
      if(itd_other == data_map_other.end())
        throw std::runtime_error("Cannot find data in the given survey to compare with.");

      const Datum &datum_other = itd_other->second;

      if(datum.status == DatumEmpty || datum_other.status == DatumEmpty)
        throw std::runtime_error("Some data measurements are empty during calculation of misfit.");

      if(datum.stderr < 1e-18)
        throw std::runtime_error("Some data uncertanties are smaller than 1e-18. Check your input data.");

      const double val = pow(fabs(datum.value - datum_other.value) / datum.stderr, norm_power);
      misfit += val;

      // Since data come from different methods, normalize misfit with # of data per method
      const auto it_factor = factors.find(std::make_pair(datum.method, datum.calculator_type));
      if(it_factor == factors.end())
        throw std::runtime_error("Cannot find method " + type_method_conversion[datum.method]);

      misfit_normalized += val * it_factor->second;

//      mmap[it_factor->first] += val * it_factor->second;
    }
  }

//  for(auto p: mmap)
//    std::cout << type_calculator_conversion[p.first.second] << "\t" << p.second << std::endl;
}

void SurveyData::write_plain(const std::string &filename) const
{
  std::ofstream ofs(filename);

  if(!ofs.is_open())
    throw std::runtime_error("Cannot open file " + filename);

  write_plain(ofs);
}

void SurveyData::write_plain(std::ofstream &ofs) const
{
  MethodCalculatorSet methods = get_methods_list();

  for(auto p: methods)
  {
    auto itf = instance_indices_per_method.find(p.first);
    const std::set<unsigned> &findices = itf->second;

    auto its = source_indices_per_method.find(p.first);
    const std::set<unsigned> &tindices = its->second;

    for(unsigned fidx: findices)
    {
      // Loop over transmitter
      for(unsigned tidx: tindices)
      {
        std::unordered_set<unsigned> receiver_indices;
        receivers_at(fidx, tidx, receiver_indices);

        if(receiver_indices.size() != 1)
          continue;

        const unsigned ridx = *receiver_indices.begin();

        std::vector<RealDataType> retrieved_types;
        dvector values, errors;
        data_at(fidx, tidx, ridx, retrieved_types, values, errors);

        ofs << instance(fidx) << "\t" << tidx << "\t" << ridx;
        for(double v: values)
          ofs << "\t" << v;
        for(double e: errors)
          ofs << "\t" << e;
        ofs << "\n";
      }
    }
  }
}

MethodCalculatorSet SurveyData::get_methods_list() const
{
  MethodCalculatorSet methods;

  for(auto &p: data_table)
  {
    const DataTable::mapped_type &data = p.second;
    for(auto &d: data)
    {
      const Datum &datum = d.second;
      methods.insert(std::make_pair(datum.method, datum.calculator_type));
    }
  }

  return methods;
}

void SurveyData::frequency_source_pairs(const ForwardType &type,
                                        std::vector<std::pair<double, std::string> > &pairs) const
{
  std::set<std::pair<unsigned, unsigned> > unique_indices;
  for(auto &p: data_table)
  {
    const DataTable::key_type &key = p.first;
    const DataTable::mapped_type &data = p.second;
    for(auto &d: data)
    {
      const Datum &datum = d.second;

      if(datum.calculator_type == type)
        unique_indices.insert(std::make_pair(key._fidx, key._sidx));
    }
  }

  pairs.clear();

  for(auto &p: unique_indices)
    pairs.push_back(std::make_pair(instances[p.first], physical_sources[p.second]->get_name()));
}

void SurveyData::create_linear_index()
{
  MethodCalculatorSet methods = get_methods_list();

  unsigned index = 0;
  for(auto p: methods) // sort by forward operator type, frequency, transmitter, receiver
  {
    const std::set<unsigned> frequencies = used_instances(p.second);
    const std::set<unsigned> transmitters = used_sources(p.second);
    const std::set<unsigned> receivers = used_receivers(p.second);

    for(unsigned fidx: frequencies)
    {
      for(unsigned tidx: transmitters)
      {
        for(unsigned ridx: receivers)
        {
          DataTable::iterator it = data_table.find(IndexKey(fidx, tidx, ridx));

          DataMap &data_map = it->second;
          for(DataMap::iterator itd = data_map.begin(); itd != data_map.end(); ++itd)
          {
            Datum &datum = itd->second;
            datum.linear_index = index++;
          }
        }
      }
    }
  }
}

void SurveyData::set_data_status(const ForwardType &forward_type, const DatumStatus &status)
{
  for(auto &p: data_table)
  {
    DataTable::mapped_type &data = p.second;
    for(auto &d: data)
    {
      Datum &datum = d.second;

      if(datum.calculator_type == forward_type)
        datum.status = status;
    }
  }
}

void SurveyData::read_receivers(const std::string &file_name)
{
  receivers = read_list<Receiver> (file_name);

  receiver_index_map.clear();
  for(size_t i = 0; i < receivers.size(); ++i)
    receiver_index_map.insert(std::make_pair(receivers[i].get_name(), i));
}

void SurveyData::read_sources(const std::string &file_name)
{
  physical_sources.clear();

  std::ifstream ifs(file_name);

  if (!ifs.is_open ())
    throw std::ios_base::failure ("Can not open file " + file_name);

  std::string clean, line;

  // read in all data skipping comments and empty lines
  while (!ifs.eof())
  {
    std::getline (ifs, line);
    // Trim string
    line.erase(0, line.find_first_not_of(" \t\r\n"));

    // Skip empty lines and comments
    if (line.length() < 1 || line[0] == '!' || line[0] == '%' || line[0] == '#')
      continue;

    clean += line + "\n";
  }

  std::stringstream ss(clean);

  while(ss.good())
  {
    std::streampos pos = ss.tellg();

    std::string source_type;
    ss >> source_type;

    if(trim(source_type).length() == 0)
      continue;

    ss.seekg(pos);

    if(istrcompare(source_type, string_to_source_type[SphericalHarmonic]))
    {
      SphericalHarmonicsSource src;
      ss >> src;
      physical_sources.push_back(PhysicalSourcePtr(new SphericalHarmonicsSource(src)));
    }
    else if(istrcompare(source_type, string_to_source_type[Dipole]))
    {
      DipoleSource src;
      ss >> src;
      physical_sources.push_back(PhysicalSourcePtr(new DipoleSource(src)));
    }
    else if(istrcompare(source_type, string_to_source_type[ExternalSourceFile]))
    {
      ExternalFileSource src;
      ss >> src;
      physical_sources.push_back(PhysicalSourcePtr(new ExternalFileSource(src)));
    }
    else
      throw std::runtime_error("Unsupported source type " + source_type);
  }

  source_index_map.clear();
  for(size_t i = 0; i < physical_sources.size(); ++i)
    source_index_map.insert(std::make_pair(physical_sources[i]->get_name(), i));
}
