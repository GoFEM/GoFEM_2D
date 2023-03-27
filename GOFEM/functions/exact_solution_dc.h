#ifndef DC_ANALYTICAL_H
#define DC_ANALYTICAL_H

#include <boost/math/special_functions/bessel.hpp>
#include <deal.II/base/function.h>

#include "survey/dc_source.h"

using namespace dealii;

namespace
{
template<int dim>
class HalfspaceSolution : public Function<dim>
{
public:
    HalfspaceSolution(const MultiDipole &source, double conductivity, double k = 1.):
        Function<dim>(1), source_(source), sigma(conductivity), k_(k)
    {}
    virtual double value (const Point<dim> &p, const unsigned int component = 0) const;

private:
    MultiDipole source_;
    double sigma, k_;
};

template<>
double HalfspaceSolution<3>::value(const Point<3> &p, const unsigned int) const
{
    double potential = 0.;

    for(size_t i = 0; i < source_.n_electrodes(); ++i)
        potential += source_.polarization(i) / (2 * M_PI * sigma * p.distance(source_.electrode_location<Point<3>>(i)));

    return potential;
}

template<>
double HalfspaceSolution<2>::value(const Point<2> &p, const unsigned int /*component*/) const
{
    double potential = 0.;

    for(size_t i = 0; i < source_.n_electrodes(); ++i)
    {
        double r = k_ * p.distance(source_.electrode_location<Point<2>>(i));
        potential += source_.polarization(i) * boost::math::cyl_bessel_k(0, r);
    }

    return potential / (2 * M_PI * sigma);
}

template <int dim>
class WeightFunction: public Function<dim>
{
public:
    WeightFunction (const std::vector<Point<dim>> &electrode_positions, double radius)  :
        Function<dim> (1), positions(electrode_positions), radius_(radius)
    {}

    virtual double value (const Point<dim> &p, const unsigned int component = 0) const;

private:
    std::vector<Point<dim>> positions;
    double radius_;
};

template <int dim>
inline double WeightFunction<dim>::value (const Point<dim> &p, const unsigned int) const
{
    for(size_t i = 0; i < positions.size(); ++i)
        if(p.distance(positions[i]) < radius_)
            return 0.;

    return 1.;
}
}

#endif // DC_ANALYTICAL_H
