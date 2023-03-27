#include "survey.h"

#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <type_traits>

#include "io/read_list.h"
#include "source_name.h"

SurveyData::SurveyData ()
{}

SurveyData::SurveyData(const std::string data_file, const std::string receiver_file,
                       const std::string source_file, SurveyMethod survey_method):
  input_data_file (data_file), receiver_definition_file(receiver_file),
  source_definition_file(source_file), method(survey_method)
{
  read_data ();
}

size_t SurveyData::n_real_data () const
{
  size_t n_data = 0;

  // Loop over sources
  for(typename SurveyMap::const_iterator it = survey.begin(); it != survey.end(); ++it)
  {
    // Loop over sources
    n_data += it->second.n_real_data ();
  }

  return n_data;
}


void SurveyData::nullify()
{
  // Loop over sources
  for(typename SurveyMap::iterator it = survey.begin(); it != survey.end(); ++it)
  {
    SourceData& source = it->second;
    for(auto &p: source.receivers_data)
      p.second.nullify ();
  }
}


unsigned SurveyData::data_offset(double frequency) const
{
  unsigned offset = std::numeric_limits<unsigned>::max();
  for(typename SurveyMap::const_iterator it = survey.begin(); it != survey.end(); ++it)
  {
    const SourceData& source_data = it->second;
    if(fabs(frequency - source_data.frequency) < 1e-8)
      offset = std::min(source_data.data_offset(), offset);
  }

  return offset;
}

void SurveyData::get_sources(double frequency, std::vector<SourceData*> &sources)
{
  std::map<std::string, SourceData*> source_map;
  for(auto &p: survey)
  {
    SourceData& source_data = p.second;
    if(fabs(frequency - source_data.frequency) / frequency < 1e-5)
      source_map[source_data.source_name] = &source_data;
  }

  sources.clear();
  for(auto s: source_map)
    sources.push_back(s.second);
}


std::vector<double> SurveyData::unique_frequencies() const
{
  std::vector<double> frequencies;

  std::set<double, weak_compare> freqs_set;
  // Loop over sources
  for(const auto &p: survey)
  {
    const SourceData& source_data = p.second;
    freqs_set.insert(source_data.frequency);
  }

  frequencies.clear();
  frequencies.resize(freqs_set.size());
  std::copy(freqs_set.begin(), freqs_set.end(), frequencies.begin());

  return frequencies;
}

std::vector<std::string> SurveyData::unique_receiver_names() const
{
  std::set<std::string> names_set;
  // Loop over sources
  for(const auto &p: survey)
  {
    const SourceData& source_data = p.second;
    for(auto& p: source_data.receivers)
      names_set.insert(p.get_name());
  }

  std::vector<std::string> names(names_set.size());
  std::copy(names_set.begin(), names_set.end(), names.begin());

  return names;
}

std::vector<dvec3d> SurveyData::unique_receiver_positions() const
{
  std::map<std::string, dvec3d> positions_map;
  // Loop over sources
  for(const auto &p: survey)
  {
    const SourceData& source_data = p.second;
    for(auto& p: source_data.receivers)
      positions_map.insert({p.get_name(), p.position<dvec3d>(0)});
  }

  std::vector<dvec3d> positions;
  for(auto& p: positions_map)
    positions.push_back(p.second);

  return positions;
}

void SurveyData::get_datum_summary(const unsigned index, double &frequency,
                                   std::string &srcname, std::string &recname,
                                   RealDataType &data_type)
{
  std::vector<double> frequencies = unique_frequencies();

  for(double f: frequencies)
  {
    std::vector<SourceData*> sources;
    get_sources(f, sources);

    for(auto s: sources)
    {
      for(auto& p: s->receivers_data)
      {
        ReceiverData &r = p.second;

        if(!r.data_present())
          continue;

        for(auto type: r.get_measured_types ())
        {
          if(index == r.get_datum_index(type))
          {
            frequency = f;
            srcname = s->source_name;
            recname = p.first;
            data_type = type;

            return;
          }
        }
      }
    }
  }

  throw std::runtime_error("Datum with index " + std::to_string(index) + " was not found.");
}

void SurveyData::reindex_data()
{
  const std::vector<double> frequencies = unique_frequencies();

  unsigned index = 0;
  for(double f: frequencies)
  {
    std::vector<SourceData*> sources;
    get_sources(f, sources);

    for(auto s: sources)
    {
      for(auto& p: s->receivers_data)
      {
        ReceiverData &r = p.second;

        if(!r.data_present())
          continue;

        for(auto type: r.get_measured_types ())
        {
          r.set_index(type, index);
          ++index;
        }
      }
    }
  }
}

#ifndef NO_MPI

void SurveyData::construct_petsc_data_vector (Vec data, Mat errors)
{
  PetscInt vsize, low, high;
  VecGetSize (data, &vsize);

  Vec errors_vec;
  if(errors != NULL)
    VecDuplicate(data, &errors_vec);

  if (static_cast<unsigned>(vsize) != n_real_data())
    throw std::runtime_error ("The size of the vector given to the routine is not compatible with input data set.");

  VecGetOwnershipRange (data, &low, &high);

  // Loop over sources
  for (const auto& s: survey)
  {
    // Loop over sources
    const SourceData& source = s.second;
    for (const auto& p: source.receivers_data)
    {
      const ReceiverData &rdata = p.second;

      std::vector<std::pair<unsigned, double> > rec_data, rec_errors;
      rdata.get_measured_values (rec_data);
      if (errors != NULL)
        rdata.get_measurement_errors (rec_errors);

      for (size_t i = 0; i < rec_data.size(); ++i)
        if (rdata.data_present() &&
            static_cast<int>(rec_data[i].first) >= low &&
            static_cast<int>(rec_data[i].first) < high)
        {
          VecSetValue (data, rec_data[i].first, rec_data[i].second, INSERT_VALUES);
          if (errors != nullptr)
            VecSetValue (errors_vec, rec_errors[i].first, rec_errors[i].second, INSERT_VALUES);
        }
    }
  }

  VecAssemblyBegin(data);
  VecAssemblyEnd(data);

  if (errors != NULL)
  {
    VecAssemblyBegin(errors_vec);
    VecAssemblyEnd(errors_vec);

    VecReciprocal(errors_vec);

    MatDiagonalSet(errors, errors_vec, INSERT_VALUES);

    VecDestroy(&errors_vec);
  }
}


void SurveyData::broadcast_data(MPI_Comm communicator)
{
  // Loop over sources
  for (auto& s: survey)
  {
    SourceData& source = s.second;
    // Loop over receivers
    for (auto &p: source.receivers_data)
      p.second.broadcast_data(communicator);
  }
}
#endif

void SurveyData::write(std::string filename) const
{
  // Read inversion input data
  std::ofstream ofs (filename.c_str());

  if(ofs.fail ())
    throw std::ios::failure ("Could not open file " + filename);

  for(const auto& s: survey)
  {
    const SourceData& sdata = s.second;
    for(const auto& p: sdata.receivers_data)
    {
      const ReceiverData &rdata = p.second;
      const std::string &rname = p.first;

      if(!rdata.data_present())
        continue;

      std::stringstream ss;
      ss << std::scientific << std::setprecision(8) << sdata.frequency << "\t"
         << sdata.source_name << "\t" << rname;

      auto data_types = rdata.get_measured_types ();
      for(RealDataType type: data_types)
      {
        double val = rdata.get_value(type);
        double err = rdata.get_datum_error(type);
        ofs << std::scientific << std::setprecision(8)
            << Datum::convert_type_to_string(type) << "\t" << ss.str()
            << "\t" << val << "\t" << err << "\n";
      }
    }
  }
}

SurveyMethod SurveyData::survey_method() const
{
  return method;
}

void SurveyData::read_data ()
{
  // Create associated array for receivers
  std::vector<Receiver> receivers = read_list<Receiver> (receiver_definition_file);

  std::set<std::string> receiver_names;
  for(size_t i = 0; i < receivers.size(); ++i)
    receiver_names.insert(receivers[i].get_name());

  if(receiver_names.size() != receivers.size())
    throw std::runtime_error("Receiver file " + receiver_definition_file +
                             " contains multiple receivers with the same name.");

  std::set<std::string> source_names;
  if(!source_definition_file.empty() &&
     method != MT &&
     method != MTSphere)
  {
    const auto sources = read_list<SourceName> (source_definition_file);
    for(size_t i = 0; i < sources.size(); ++i)
      source_names.insert(sources[i].name);

    if(source_names.size() != sources.size())
      throw std::runtime_error("Source file " + source_definition_file +
                               " contains multiple sources with the same name.");
  }

  // Read inversion input data
  std::ifstream ifs (input_data_file.c_str());

  if(ifs.fail ())
    throw std::ios::failure ("Could not open file " + input_data_file);

  std::string receiver_name, source_name, data_type;
  double frequency, value, err;
  std::string line;

  while (!ifs.eof())
  {
    line.erase ();
    std::getline (ifs, line);

    line.erase(0, line.find_first_not_of(" \n\r\t"));

    if(line.length() == 0 || line[0] == '#')
      continue;

    std::istringstream ss(line);
    ss >> data_type >> frequency
       >> source_name >> receiver_name
       >> value >> err;

    if(fabs(value) < std::numeric_limits<float>::min())
    {
      std::cout << "Value " << value << " is too small and will treated as zero.\n";
      continue;
    }

    if(err < std::numeric_limits<float>::min())
    {
      std::stringstream ss;
      ss << "Negative error " << err << " is detected in the input data file. "
         << "Measurement uncertainty represent standard deviations and cannot be negative.\n";

      throw std::runtime_error(ss.str());
    }

    if(source_name.find("Plane_wave") == std::string::npos)
    {
      if(source_names.size() == 0)
      {
        throw std::runtime_error("You need to provide valid source name in data "
                                 "file or specify source definition file.");
      }
      else if(source_names.find(source_name) == source_names.end())
      {
        throw std::runtime_error("Source " + source_name +
                                 " has not been defined in file " +
                                 source_definition_file);
      }
    }

    if(receiver_names.find(receiver_name) == receiver_names.end())
    {
      throw std::runtime_error("Receiver " + receiver_name +
                               " has not been defined in file " +
                               receiver_definition_file);
    }

    // Create source for receiver if it does not exist yet
    if(source_names.size() == 0)
      source_name = "Plane_wave";

    std::stringstream sstream;
    sstream << std::fixed << std::setprecision(10) << source_name << "_" << frequency << "_Hz";
    std::string sname = sstream.str();

    if (survey.find(sname) == survey.end())
    {
      survey[sname] = SourceData();
      survey[sname].source_name = source_name;
      survey[sname].frequency = frequency;

      if(source_names.size() == 0)
        survey[sname].is_plane_wave = true;
      else
        survey[sname].is_plane_wave = false;
    }

    survey[sname].receivers_data[receiver_name].add_value(Datum::convert_string_to_type(data_type), value, err, -1);
    survey[sname].receivers_data[receiver_name].set_data_present(true);

    bool receiver_found = false;
    for(const Receiver &rec: survey[sstream.str()].receivers)
    {
      if(rec.get_name() == receiver_name)
        receiver_found = true;
    }

    if(!receiver_found)
    {
      for(const Receiver &rec: receivers)
      {
        if(rec.get_name() == receiver_name)
          survey[sname].receivers.push_back(rec);
      }
    }
  }

  // Make proper, contiguous indexing of the data read
  reindex_data();
}
