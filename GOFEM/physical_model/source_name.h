#ifndef SOURCE_NAME_H
#define SOURCE_NAME_H

#include <string>
#include <limits>
#include <istream>

class SourceName
{
public:
  friend std::istream& operator>> (std::istream &in, SourceName &s)
  {
    std::string type;
    in >> type >> s.name;
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return in;
  }

  bool operator<(const SourceName& other) const
  {
    return name < other.name;
  }

  std::string name;
};

#endif // SOURCE_NAME_H
