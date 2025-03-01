/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/initial_temperature/S40RTS_perturbation.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/utilities.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/std_cxx11/array.h>

#include <boost/lexical_cast.hpp>

namespace aspect
{
  namespace InitialTemperature
  {
    namespace internal
    {
      namespace S40RTS
      {
        // Read in the spherical harmonics that are located in data/initial-temperature/S40RTS
        // and were downloaded from http://www.earth.lsa.umich.edu/~jritsema/research.html
        // Ritsema et al. choose real sine and cosine coefficients that follow the normalization
        // by Dahlen & Tromp, Theoretical Global Seismology (equations B.58 and B.99).

        class SphericalHarmonicsLookup
        {
          public:
            SphericalHarmonicsLookup(const std::string &filename,
                                     const MPI_Comm &comm)
            {
              std::string temp;
              // Read data from disk and distribute among processes
              std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));

              in >> order;
              getline(in,temp);  // throw away the rest of the line

              const unsigned int num_splines = 21;
              const unsigned int maxnumber = num_splines * (order+1)*(order+1);

              // read in all coefficients as a single data vector
              std::vector<double> coeffs(maxnumber,0.0);

              for (unsigned int i=0; i<maxnumber; ++i)
                {
                  in >> coeffs[i];
                }

              // reorder the coefficients into sin and cos coefficients. a_lm will be the cos coefficients
              // and b_lm the sin coefficients.
              unsigned int ind = 0;

              a_lm.reserve(maxnumber);
              b_lm.reserve(maxnumber);
              for (unsigned int j=0; j<num_splines; ++j)
                for (unsigned int i=0; i<order+1; ++i)
                  {
                    a_lm.push_back(coeffs[ind]);
                    b_lm.push_back(0.0);
                    ++ind;

                    unsigned int ind_degree = 0;
                    while (ind_degree < i)
                      {
                        a_lm.push_back(coeffs[ind]);
                        ++ind;
                        b_lm.push_back(coeffs[ind]);
                        ++ind;
                        ++ind_degree;
                      }
                  }
            }

            // Declare a function that returns the cosine coefficients
            const std::vector<double> &cos_coeffs() const
            {
              return a_lm;
            }

            // Declare a function that returns the sine coefficients
            const std::vector<double> &sin_coeffs() const
            {
              return b_lm;
            }

            unsigned int maxdegree()
            {
              return order;
            }

          private:
            unsigned int order;
            std::vector<double> a_lm;
            std::vector<double> b_lm;

        };

        // Read in the knot points for the spline interpolation. They are located in data/
        // initial-temperature/S40RTS and were taken from the plotting script
        // lib/libS20/splhsetup.f which is part of the plotting package downloadable at
        // http://www.earth.lsa.umich.edu/~jritsema/research.html
        class SplineDepthsLookup
        {
          public:
            SplineDepthsLookup(const std::string &filename,
                               const MPI_Comm &comm)
            {
              std::string temp;
              // Read data from disk and distribute among processes
              std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));

              getline(in,temp);  // throw away the rest of the line
              getline(in,temp);  // throw away the rest of the line

              // This is fixed for this tomography model
              const unsigned int num_splines = 21;

              depths.resize(num_splines);
              for (unsigned int i=0; i<num_splines; ++i)
                {
                  in >> depths[i];
                }
            }

            const std::vector<double> &spline_depths() const
            {
              return depths;
            }

          private:
            std::vector<double> depths;
        };
      }
    }


    template <int dim>
    void
    S40RTSPerturbation<dim>::initialize()
    {
      spherical_harmonics_lookup.reset(new internal::S40RTS::SphericalHarmonicsLookup(data_directory+harmonics_coeffs_file_name,this->get_mpi_communicator()));
      spline_depths_lookup.reset(new internal::S40RTS::SplineDepthsLookup(data_directory+spline_depth_file_name,this->get_mpi_communicator()));
    }

    template <>
    double
    S40RTSPerturbation<2>::
    initial_temperature (const Point<2> &) const
    {
      // we shouldn't get here but instead should already have been
      // kicked out by the assertion in the parse_parameters()
      // function
      Assert (false, ExcNotImplemented());
      return 0;
    }


    template <>
    double
    S40RTSPerturbation<3>::
    initial_temperature (const Point<3> &position) const
    {
      const unsigned int dim = 3;

      // use either the user-input reference temperature as background temperature
      // (incompressible model) or the adiabatic temperature profile (compressible model)
      const double background_temperature = this->get_material_model().is_compressible() ?
                                            this->get_adiabatic_conditions().temperature(position) :
                                            reference_temperature;

      // get the degree from the input file (20 or 40)
      unsigned int max_degree = spherical_harmonics_lookup->maxdegree();

      // lower the maximum order if needed
      if (lower_max_order)
        {
          AssertThrow(max_order <= max_degree, ExcMessage("Specifying a maximum order higher than the order of spherical harmonic data is not allowed"));
          max_degree = max_order;
        }

      // This tomography model is parameterized by 21 layers
      const unsigned int num_spline_knots = 21;

      // get the spherical harmonics coefficients
      const std::vector<double> a_lm = spherical_harmonics_lookup->cos_coeffs();
      const std::vector<double> b_lm = spherical_harmonics_lookup->sin_coeffs();

      // get spline knots and rescale them from [-1 1] to [CMB Moho]
      const std::vector<double> r = spline_depths_lookup->spline_depths();
      const double rmoho = 6346e3;
      const double rcmb = 3480e3;
      std::vector<double> depth_values(num_spline_knots,0);

      for (unsigned int i = 0; i<num_spline_knots; ++i)
        depth_values[i] = rcmb+(rmoho-rcmb)*0.5*(r[i]+1);

      // convert coordinates from [x,y,z] to [r, phi, theta]
      std_cxx11::array<double,dim> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(position);

      // Evaluate the spherical harmonics at this position. Since they are the
      // same for all depth splines, do it once to avoid multiple evaluations.
      // NOTE: there is apparently a factor of sqrt(2) difference
      // between the standard orthonormalized spherical harmonics
      // and those used for S40RTS (see PR # 966)
      std::vector<std::vector<double> > cosine_components(max_degree+1,std::vector<double>(max_degree+1,0.0));
      std::vector<std::vector<double> > sine_components(max_degree+1,std::vector<double>(max_degree+1,0.0));

      for (unsigned int degree_l = 0; degree_l < max_degree+1; ++degree_l)
        {
          for (unsigned int order_m = 0; order_m < degree_l+1; ++order_m)
            {
              const std::pair<double,double> sph_harm_vals = Utilities::real_spherical_harmonic(degree_l, order_m, scoord[2], scoord[1]);
              cosine_components[degree_l][order_m] = sph_harm_vals.first;
              sine_components[degree_l][order_m] = sph_harm_vals.second;
            }
        }

      // iterate over all degrees and orders at each depth and sum them all up.
      std::vector<double> spline_values(num_spline_knots,0);
      double prefact;
      unsigned int ind = 0;

      for (unsigned int depth_interp = 0; depth_interp < num_spline_knots; ++depth_interp)
        {
          for (unsigned int degree_l = 0; degree_l < max_degree+1; ++degree_l)
            {
              for (unsigned int order_m = 0; order_m < degree_l+1; ++order_m)
                {
                  if (degree_l == 0)
                    prefact = (zero_out_degree_0
                               ?
                               0.
                               :
                               1.);
                  else if (order_m != 0)
                    prefact = 1./sqrt(2.);
                  else prefact = 1.0;

                  spline_values[depth_interp] += prefact * (a_lm[ind] * cosine_components[degree_l][order_m]
                                                            + b_lm[ind] * sine_components[degree_l][order_m]);

                  ++ind;
                }
            }
        }

      // We need to reorder the spline_values because the coefficients are given from
      // the surface down to the CMB and the interpolation knots range from the CMB up to
      // the surface.
      std::vector<double> spline_values_inv(num_spline_knots,0);
      for (unsigned int i=0; i<num_spline_knots; ++i)
        spline_values_inv[i] = spline_values[num_spline_knots-1 - i];

      // The boundary condition for the cubic spline interpolation is that the function is linear
      // at the boundary (i.e. Moho and CMB). Values outside the range are linearly
      // extrapolated.
      aspect::Utilities::tk::spline s;
      s.set_points(depth_values,spline_values_inv);

      // Get value at specific depth
      const double perturbation = s(scoord[0]);

      // scale the perturbation in seismic velocity into a density perturbation
      // vs_to_density is an input parameter
      const double density_perturbation = vs_to_density * perturbation;

      const double depth = this->get_geometry_model().depth(position);
      double temperature_perturbation;
      if (depth > no_perturbation_depth)
        // scale the density perturbation into a temperature perturbation
        temperature_perturbation =  -1./thermal_alpha * density_perturbation;
      else
        // set heterogeneity to zero down to a specified depth
        temperature_perturbation = 0.0;

      // add the temperature perturbation to the background temperature
      return background_temperature + temperature_perturbation;
    }


    template <int dim>
    void
    S40RTSPerturbation<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Initial temperature model");
      {
        prm.enter_subsection("S40RTS perturbation");
        {
          prm.declare_entry("Data directory", "$ASPECT_SOURCE_DIR/data/initial-temperature/S40RTS/",
                            Patterns::DirectoryName (),
                            "The path to the model data. ");
          prm.declare_entry ("Initial condition file name", "S40RTS.sph",
                             Patterns::Anything(),
                             "The file name of the spherical harmonics coefficients "
                             "from Ritsema et al.");
          prm.declare_entry ("Spline knots depth file name", "Spline_knots.txt",
                             Patterns::Anything(),
                             "The file name of the spline knot locations from "
                             "Ritsema et al.");
          prm.declare_entry ("Vs to density scaling", "0.25",
                             Patterns::Double (0),
                             "This parameter specifies how the perturbation in shear wave velocity "
                             "as prescribed by S20RTS or S40RTS is scaled into a density perturbation. "
                             "See the general description of this model for more detailed information.");
          prm.declare_entry ("Thermal expansion coefficient in initial temperature scaling", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");
          prm.declare_entry ("Remove degree 0 from perturbation","true",
                             Patterns::Bool (),
                             "Option to remove the degree zero component from the perturbation, "
                             "which will ensure that the laterally averaged temperature for a fixed "
                             "depth is equal to the background temperature.");
          prm.declare_entry ("Reference temperature", "1600.0",
                             Patterns::Double (0),
                             "The reference temperature that is perturbed by the spherical "
                             "harmonic functions. Only used in incompressible models.");
          prm.declare_entry ("Remove temperature heterogeneity down to specified depth", boost::lexical_cast<std::string>(-std::numeric_limits<double>::max()),
                             Patterns::Double (),
                             "This will set the heterogeneity prescribed by S20RTS or S40RTS to zero "
                             "down to the specified depth (in meters). Note that your resolution has "
                             "to be adequate to capture this cutoff. For example if you specify a depth "
                             "of 660km, but your closest spherical depth layers are only at 500km and "
                             "750km (due to a coarse resolution) it will only zero out heterogeneities "
                             "down to 500km. Similar caution has to be taken when using adaptive meshing.");
          prm.declare_entry ("Specify a lower maximum order","false",
                             Patterns::Bool (),
                             "Option to use a lower maximum order when reading the data file of spherical "
                             "harmonic coefficients. This is probably used for the faster tests or when the "
                             "users only want to see the spherical harmonic pattern up to a certain order.");
          prm.declare_entry ("Maximum order","20",
                             Patterns::Integer (0),
                             "The maximum order the users specify when reading the data file of spherical harmonic "
                             "coefficients, which must be smaller than the maximum order the data file stored. "
                             "This parameter will be used only if 'Specify a lower maximum order' is set to true.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    S40RTSPerturbation<dim>::parse_parameters (ParameterHandler &prm)
    {
      AssertThrow (dim == 3,
                   ExcMessage ("The 'S40RTS perturbation' model for the initial "
                               "temperature is only available for 3d computations."));

      prm.enter_subsection ("Initial temperature model");
      {
        prm.enter_subsection("S40RTS perturbation");
        {
          data_directory = Utilities::expand_ASPECT_SOURCE_DIR(prm.get ("Data directory"));

          if ((data_directory.size() > 0) && (data_directory[data_directory.size()-1] != '/'))
            data_directory += "/";
          harmonics_coeffs_file_name = prm.get ("Initial condition file name");
          spline_depth_file_name  = prm.get ("Spline knots depth file name");
          vs_to_density           = prm.get_double ("Vs to density scaling");
          thermal_alpha           = prm.get_double ("Thermal expansion coefficient in initial temperature scaling");
          zero_out_degree_0       = prm.get_bool ("Remove degree 0 from perturbation");
          reference_temperature   = prm.get_double ("Reference temperature");
          no_perturbation_depth   = prm.get_double ("Remove temperature heterogeneity down to specified depth");
          lower_max_order         = prm.get_bool ("Specify a lower maximum order");
          max_order               = prm.get_integer ("Maximum order");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

      initialize ();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(S40RTSPerturbation,
                                              "S40RTS perturbation",
                                              "An initial temperature field in which the temperature "
                                              "is perturbed following the S20RTS or S40RTS shear wave "
                                              "velocity model by Ritsema and others, which can be downloaded "
                                              "here \\url{http://www.earth.lsa.umich.edu/~jritsema/research.html}. "
                                              "Information on the vs model can be found in Ritsema, J., Deuss, "
                                              "A., van Heijst, H.J. \\& Woodhouse, J.H., 2011. S40RTS: a "
                                              "degree-40 shear-velocity model for the mantle from new Rayleigh "
                                              "wave dispersion, teleseismic traveltime and normal-mode "
                                              "splitting function measurements, Geophys. J. Int. 184, 1223-1236. "
                                              "The scaling between the shear wave perturbation and the "
                                              "temperature perturbation can be set by the user with the "
                                              "'Vs to density scaling' parameter and the 'Thermal "
                                              "expansion coefficient in initial temperature scaling' "
                                              "parameter. The scaling is as follows: $\\delta ln \\rho "
                                              "(r,\\theta,\\phi) = \\xi \\cdot \\delta ln v_s(r,\\theta, "
                                              "\\phi)$ and $\\delta T(r,\\theta,\\phi) = - \\frac{1}{\\alpha} "
                                              "\\delta ln \\rho(r,\\theta,\\phi)$. $\\xi$ is the `vs to "
                                              "density scaling' parameter and $\\alpha$ is the 'Thermal "
                                              "expansion coefficient in initial temperature scaling' "
                                              "parameter. The temperature perturbation is added to an "
                                              "otherwise constant temperature (incompressible model) or "
                                              "adiabatic reference profile (compressible model). If a depth "
                                              "is specified in 'Remove temperature heterogeneity down to "
                                              "specified depth', there is no temperature perturbation "
                                              "prescribed down to that depth.")
  }
}
