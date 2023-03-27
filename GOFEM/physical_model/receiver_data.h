#ifndef RECEIVER_DATA_H
#define RECEIVER_DATA_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <complex>
#include <iostream>

#ifndef NO_MPI
#include "mpi.h"
#endif

#include "survey/datum.h"

/*
 * Stores data for a receiver.
 */
class ReceiverData
{
public:
  ReceiverData ():
    is_data_present(false)
  {}

  /*
   * Returns number of real data values, i.e. every complex
   * measurement counts as two real ones.
   */
  size_t n_real_data () const;

  /*
   * Replaces data values with zero and sets respective flag to false
   */
  void nullify ();
  bool data_present () const;
  void set_data_present (bool f);

  /*
   * Sets value/error of the given type. Note: the old value of the same type gets replaced.
   */
  void set_value(RealDataType type, const double value);
  void set_value(ComplexDataType type, const std::complex<double> &value);
  void set_error(RealDataType type, double error);
  void set_index(RealDataType type, unsigned index);

  /*
   * Adds new measured value
   */
  void add_value(RealDataType type, double value, double error, unsigned index);

  /*
   * Returns pairs of observed data with corresponding indices in the global data vector
   * Note: complex data values get split into real and imaginary parts
   */
  void get_measured_values (std::vector<std::pair<unsigned, double>>& rec_data) const;
  void get_measurement_errors (std::vector<std::pair<unsigned, double>>& rec_errors) const;
  const std::set<RealDataType>& get_measured_types () const;

  /*
   * This method returns a counterpart for a real data type
   * if it is part of a complex type. For instance:
   * for RealZxy it will return ImagZxy and so on.
   * For inherently real data types it returns invalid type.
   */
  RealDataType get_complex_complement(const RealDataType type) const;

  /*
   * Return the complex datum's value or index for a given data type.
   */
  double get_value (RealDataType type) const;
  std::complex<double> get_value (ComplexDataType type) const;
  double get_datum_error (RealDataType type) const;

  /*
   * Returns index within global observed data vector. If not such datum exists,
   * returns std::numeric_limits<unsigned>::max()
   */
  unsigned get_datum_index (RealDataType type) const;
  unsigned get_minimum_index () const;

  void copy_from(const ReceiverData &other);

#ifndef NO_MPI
  void broadcast_data (MPI_Comm communicator);
#endif

private:
  /*
   * List of the measured data types
   */
  std::set<RealDataType> measured_data;

  /*
   * List of the linear indices for data values
   */
  std::map<RealDataType, unsigned> indices;

  /*
   * We save all values as MeasurementType, which is typically either complex or real.
   * Complex values are split later into real and imaginary parts.
   * Note: for modelled data this array does no have to match measured_data set,
   * but latter has to be a subset of the former.
   * For instance, 3D MT requires calculating additional data for Jacobian.
   */
  typedef std::map<RealDataType, double> ValuesMap;
  ValuesMap values;

  /*
   * Estimated uncertainty of the measurements.
   */
  std::map<RealDataType, double> error_estimates;

  /*
   * The flag is true only in two cases:
   * (i) if observed data has been read in for this receiver
   * (ii) if data has been calculated for this receiver on this MPI process
   */
  unsigned is_data_present;
};

#endif // RECEIVER_DATA_H
