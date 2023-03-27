#include "receiver_data.h"

#include <limits>
#include <stdexcept>

#ifndef NO_MPI
#include "mpi/mpi_error.h"
#endif

size_t ReceiverData::n_real_data () const
{
  return measured_data.size();
}

void ReceiverData::nullify ()
{
  for (RealDataType v: measured_data)
  {
    values[v] = 0.0;
    error_estimates[v] = 0.0;
  }

  is_data_present = false;
}

bool ReceiverData::data_present () const
{
  return is_data_present;
}

void ReceiverData::set_data_present (bool f)
{
  is_data_present = f;
}

void ReceiverData::set_value (RealDataType type, double value)
{
  is_data_present = true;
  values[type] = value;
}

void ReceiverData::set_value(ComplexDataType type, const std::complex<double> &value)
{
  is_data_present = true;

  auto real_types_pair = cmplx_to_real_table.find(type);

  if(real_types_pair == cmplx_to_real_table.end())
    throw std::runtime_error("Could not find real types of a complex type value");

  values[real_types_pair->second.first] = value.real();
  values[real_types_pair->second.second] = value.imag();
}

void ReceiverData::set_error(RealDataType type, double error)
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to set data to receiver that has been nullified!");

  error_estimates[type] = error;
}

void ReceiverData::set_index(RealDataType type, unsigned index)
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to request data from receiver that has been nullified!");

  auto it = indices.find(type);
  if (it != indices.end())
    it->second = index;
  else
    throw std::runtime_error("You have tried to request non-existent data from receiver.");
}

void ReceiverData::add_value (RealDataType type, double value, double error, unsigned index)
{
  values[type] = value;
  error_estimates[type] = error;
  indices[type] = index;
  measured_data.insert(type);

  is_data_present = true;
}

void ReceiverData::get_measured_values (std::vector<std::pair<unsigned, double> >& rec_data) const
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to request data from receiver that has been nullified!");

  rec_data.reserve (measured_data.size () * 2);

  for (RealDataType v: measured_data)
  {
    if(values.find(v) == values.end())
      throw std::runtime_error("No requested value found!");

    unsigned index = indices.find(v)->second;
    double value = values.find(v)->second;

    rec_data.push_back (std::make_pair (index, value));
  }
}

void ReceiverData::get_measurement_errors (std::vector<std::pair<unsigned, double> >& rec_errors) const
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to request data from receiver that has been nullified!");

  rec_errors.reserve (measured_data.size () * 2);

  for (RealDataType v: measured_data)
  {
    const unsigned index = indices.find(v)->second;
    const double error = error_estimates.find(v)->second;
    rec_errors.push_back (std::make_pair (index, error));
  }
}

RealDataType ReceiverData::get_complex_complement(const RealDataType type) const
{
  auto it = cmplx_counterparts_table.find(type);

  if(it == cmplx_counterparts_table.end())
    return InvalidType;
  else if(measured_data.find(it->second) == measured_data.end())
    return InvalidType;
  else
    return it->second;
}

const std::set<RealDataType> &ReceiverData::get_measured_types() const
{
  return measured_data;
}

double ReceiverData::get_value (RealDataType type) const
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to request data from receiver that has been nullified!");

  typename ValuesMap::const_iterator it = values.find (type);

  if (it == values.end())
    throw std::runtime_error("You have requested datum that does not exist!");

  return it->second;
}

std::complex<double> ReceiverData::get_value(ComplexDataType type) const
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to request data from receiver that has been nullified!");

  auto real_types_pair = cmplx_to_real_table.find(type);
  if(real_types_pair == cmplx_to_real_table.end())
    throw std::runtime_error("Could not find real types of a complex type value");

  double re = 0., im = 0.;
  {
    auto it = values.find (real_types_pair->second.first);
    if (it != values.end())
      re = it->second;
  }

  {
    auto it = values.find (real_types_pair->second.second);
    if (it != values.end())
      im = it->second;
  }

  return std::complex<double>(re, im);
}

double ReceiverData::get_datum_error (RealDataType type) const
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to request data from receiver that has been nullified!");

  const auto it = error_estimates.find (type);
  if (it == error_estimates.end())
    throw std::runtime_error("You have requested datum that does not exist!");

  return it->second;
}

unsigned ReceiverData::get_datum_index (RealDataType type) const
{
  if(!is_data_present)
    throw std::runtime_error("You have tried to request data from receiver that has been nullified!");

  auto it = indices.find(type);

  if (it != indices.end())
    return it->second;
  else
    return std::numeric_limits<unsigned>::max();
}


unsigned ReceiverData::get_minimum_index() const
{
  unsigned minidx = std::numeric_limits<unsigned>::max();
  for (RealDataType v: measured_data)
  {
    const unsigned index = indices.find(v)->second;
    if(index < minidx)
      minidx = index;
  }

  return minidx;
}

void ReceiverData::copy_from(const ReceiverData &other)
{
  for(auto type: other.measured_data)
  {
    if(measured_data.find(type) != measured_data.end())
      throw std::runtime_error("When merging data for receiver duplicates were found.");

    measured_data.insert(type);

    {
      auto it = other.values.find(type);
      values.insert(std::make_pair(type, it->second));
    }

    {
      auto it = other.error_estimates.find(type);
      error_estimates.insert(std::make_pair(type, it->second));
    }

    {
      auto it = other.indices.find(type);
      indices.insert(std::make_pair(type, it->second));
    }
  }
}

#ifndef NO_MPI

void ReceiverData::broadcast_data(MPI_Comm communicator)
{
  int rank, size, root, ierr = 0;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);

  if(size == 1)
    return;

  /*
   * Upon calling this routine only one process in communicator
   * is allowed to contain data.
   */
  unsigned n_data_present;
  MPI_Allreduce (&is_data_present, &n_data_present, 1, MPI_UNSIGNED, MPI_SUM, communicator);

  if(n_data_present != 1)
  {
    throw std::runtime_error("Not one process possesses data for receiver. This is not permitted");
  }

  // Figure out which process owns data
  if(is_data_present)
  {
    root = rank;
    for (int i = 0; i < size; i++)
    {
      if (i != rank)
      {
        ierr = MPI_Send(&rank, 1, MPI_INT, i, 0, communicator);
        check_mpi_error (ierr);
      }
    }
  }
  else
  {
    ierr = MPI_Recv (&root, 1, MPI_INT, MPI_ANY_SOURCE, 0, communicator, MPI_STATUS_IGNORE);
    check_mpi_error (ierr);
  }

  // Broadcast data size
  unsigned n_data = values.size();
  ierr = MPI_Bcast(&n_data, 1, MPI_UNSIGNED, root, communicator);
  check_mpi_error (ierr);
  MPI_Barrier(communicator);

  // Broadcast data
  std::vector<double> data_values(n_data);
  std::vector<double> error_values(n_data);
  std::vector<RealDataType> data_keys(n_data);

  if(rank == root && is_data_present)
  {
    size_t idx = 0;
    for(typename ValuesMap::const_iterator it = values.begin(); it != values.end(); ++it)
    {
      data_keys[idx] = it->first;
      error_values[idx] = error_estimates[it->first];
      data_values[idx++] = it->second;
    }
  }

  ierr = MPI_Bcast(&data_values[0], n_data * sizeof(double), MPI_BYTE, root, communicator);
  ierr = MPI_Bcast(&error_values[0], n_data * sizeof(double), MPI_BYTE, root, communicator);
  ierr = MPI_Bcast(&data_keys[0], n_data, MPI_UNSIGNED, root, communicator);
  check_mpi_error (ierr);
  MPI_Barrier(communicator);

  // Store data
  if(rank != root)
  {
    for(unsigned i = 0; i < n_data; ++i)
    {
      values[data_keys[i]] = data_values[i];
      error_estimates[data_keys[i]] = error_values[i];
    }

    is_data_present = true;
  }
}
#endif
