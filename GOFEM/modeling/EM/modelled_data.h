#ifndef MODELLED_DATA_H
#define MODELLED_DATA_H

#include <string>
#include <complex>
#include <map>
#include <vector>

/*
 * This structure stores source-receiver map data in an associative
 * array s.t. it can be addressed in a convenient way from outside
 */
template<class T>
struct ModelledData
{
  void get(const std::string &source_name, const std::string &receiver_name, std::vector<T> &data) const
  {
    const auto sit = _data.find(source_name);

    if(sit == _data.end())
      throw std::runtime_error("You requested data for source " + source_name + " which does not exist.");

    const auto &sdata = sit->second;
    const auto rit = sdata.find(receiver_name);

    if(rit == sdata.end())
      throw std::runtime_error("You requested data for source " + receiver_name + " which does not exist.");

    data = rit->second;
  }

  void set(const std::string &source_name, const std::string &receiver_name, const std::vector<T> &data)
  {
    _data[source_name][receiver_name] = data;
  }

  void to_vector(std::vector<T> &data) const
  {
    data.clear();

    for(const auto &p: _data)
    {
      const std::map<std::string, std::vector<T>> &sdata = p.second;

      for(const auto &r: sdata)
      {
        const std::vector<T> &rdata = r.second;

        for(const auto &v: rdata)
          data.push_back(v);
      }
    }
  }

  void from_vector(const std::vector<T> &data)
  {
    unsigned index = 0;
    for(auto &p: _data)
    {
      std::map<std::string, std::vector<T>> &sdata = p.second;

      for(auto &r: sdata)
      {
        std::vector<T> &rdata = r.second;

        for(auto &v: rdata)
          v = data[index++];
      }
    }
  }

private:
  std::map<std::string, std::map<std::string, std::vector<T>>> _data;
};

#endif // MODELLED_DATA_H
