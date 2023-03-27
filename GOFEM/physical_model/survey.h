#ifndef SURVEY_H
#define SURVEY_H

#include <map>
#include <limits>

#ifndef NO_MPI
#include <petscvec.h>
#include <petscmat.h>
#endif

#include "survey/receiver.h"
#include "receiver_data.h"

class weak_compare : public std::binary_function<double,double,bool>
{
public:
  weak_compare( double arg_ = 1e-7 ) : epsilon(arg_) {}
  bool operator()( const double &left, const double &right  ) const
  {
    // you can choose other way to make decision
    // (The original version is: return left < right;)
    return (fabs(left - right) > epsilon) && (left < right);
  }
  const double epsilon;
};

/*
 * Stores a transmitter and all measurements associated with it
 */
struct SourceData
{
  std::string source_name;
  bool is_plane_wave;

  double frequency;

  std::vector<Receiver> receivers;
  std::map<std::string, ReceiverData> receivers_data;

  size_t n_real_data () const
  {
    size_t n_data = 0;
    for(auto &rdata: receivers_data)
      n_data += rdata.second.n_real_data ();

    return n_data;
  }

  const Receiver& get_receiver(const std::string& name) const
  {
    for(auto &r: receivers)
    {
      if(name == r.get_name())
        return r;
    }

    throw std::runtime_error("Cannot find receiver with name " + name);
  }

  /*
   * Return the smallest index of the data value from
   * this source in the global vector from this source
   */
  unsigned data_offset () const
  {
    unsigned offset = std::numeric_limits<unsigned>::max();
    for(auto &rdata: receivers_data)
      offset = std::min(rdata.second.get_minimum_index(), offset);
    return offset;
  }

};

class SurveyData
{
public:
  SurveyData ();
  SurveyData (const std::string data_file, const std::string receiver_file,
              const std::string source_file, SurveyMethod survey_method);

  // Returns number of real data values in the whole dataset
  size_t n_real_data () const;

  // Preserves structure, but sets all data values to zero
  void nullify ();

  // Returns a vector of unique frequencies contained in the survey
  // (for time domain data always returns empty vector)
  std::vector<double> unique_frequencies() const;

  std::vector<std::string> unique_receiver_names() const;
  std::vector<dvec3d> unique_receiver_positions() const;

  void get_datum_summary(const unsigned index, double &frequency,
                         std::string &srcname, std::string &recname,
                         RealDataType &data_type);

  // Returns a global data index for a given frequency
  // Note: data in the global layout are sorted by frequency/time,
  // then by source, then by receiver and finally by data type
  unsigned data_offset(double frequency) const;

  // Get all sources at a given frequency/time
  void get_sources(double frequency, std::vector<SourceData*> &sources);

#ifndef NO_MPI
  // Constructs petsc vector of observed data and weighting matrix (if required)
  void construct_petsc_data_vector (Vec data, Mat errors = PETSC_NULL);

  /*
   * Distributes data over all processes in the communicator. After calling
   * this method all processes own the full copy of data.
   */
  void broadcast_data (MPI_Comm communicator);
#endif

  // Writes survey data
  void write(std::string filename) const;

  SurveyMethod survey_method() const;

private:
  // Read observed data and errors from the input file
  void read_data ();

  // Assign every datum an index such that the data is ordered
  // over frequencies(times)/sources/receivers/datatypes.
  void reindex_data();

  /*
   * Whathever the input format is, we order data over
   * frequencies(times)/sources/receivers/datatypes.
   * This decision is made since we parallelize over
   * frequencies or groups thereof.
   */
  typedef std::map<std::string, SourceData> SurveyMap;
  SurveyMap survey;

  // File path to the observed data file
  std::string input_data_file;
  std::string receiver_definition_file;
  std::string source_definition_file;

  SurveyMethod method;
};

#endif // SURVEY_H
