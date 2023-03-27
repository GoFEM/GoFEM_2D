#ifndef PHYSICAL_SOURCE_H
#define PHYSICAL_SOURCE_H

#include <memory>
#include <vector>
#include <string>

/*
 * Abstract source definition class.
 */
class PhysicalSource
{
public:
  PhysicalSource(const std::string source_name);
  virtual ~PhysicalSource() {}

  std::string get_name() const;
  void set_name(const std::string name);

protected:
  std::string name;
};

typedef std::shared_ptr<PhysicalSource> PhysicalSourcePtr;

#endif // PHYSICAL_SOURCE_H
