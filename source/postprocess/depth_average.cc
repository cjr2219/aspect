/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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


#include <aspect/postprocess/depth_average.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/lateral_averaging.h>
#include <aspect/global.h>
#include <aspect/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out_stack.h>


#include <math.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace
    {
      // This function takes a vector of variables names and returns a vector
      // of the same variable names, except it removes all variables that are
      // not computed by the LateralAveraging class. In other words the
      // returned vector is a proper input for LateralAveraging<dim>::get_averages().
      std::vector<std::string>
      filter_non_averaging_variables(const std::vector<std::string> &variables)
      {
        std::vector<std::string> averaging_variables;
        averaging_variables.reserve(variables.size());

        for (unsigned int i=0; i<variables.size(); ++i)
          if (!((variables[i] == "adiabatic_temperature")
                || (variables[i] == "adiabatic_pressure")
                || (variables[i] == "adiabatic_density")
                || (variables[i] == "adiabatic_density_derivative")))
            averaging_variables.push_back(variables[i]);
        return averaging_variables;
      }
    }

    template <int dim>
    template <class Archive>
    void DepthAverage<dim>::DataPoint::serialize (Archive &ar,
                                                  const unsigned int)
    {
      ar &time &values;
    }


    template <int dim>
    DepthAverage<dim>::DepthAverage ()
      :
      // the following value is later read from the input file
      output_interval (0),
      // initialize this to a nonsensical value; set it to the actual time
      // the first time around we get to check it
      last_output_time (std::numeric_limits<double>::quiet_NaN()),
      n_depth_zones (numbers::invalid_unsigned_int),
      ascii_output(false)
    {}



    template <int dim>
    std::pair<std::string,std::string>
    DepthAverage<dim>::execute (TableHandler &)
    {
      // if this is the first time we get here, set the next output time
      // to the current time. this makes sure we always produce data during
      // the first time step
      if (std::isnan(last_output_time))
        last_output_time = this->get_time() - output_interval;

      // see if output is requested at this time
      if (this->get_time() < last_output_time + output_interval)
        return std::pair<std::string,std::string>();

      DataPoint data_point;
      data_point.time       = this->get_time();

      // Add all the requested fields
      {
        const std::vector<std::string> averaging_variables = filter_non_averaging_variables(variables);

        // Compute averaged variables
        data_point.values = this->get_lateral_averaging().get_averages(n_depth_zones,averaging_variables);

        // Grow data_point.values to include adiabatic properties, and reorder
        // starting from end (to avoid unnecessary copies), and fill in the adiabatic variables.
        data_point.values.resize(variables.size(), std::vector<double> (n_depth_zones));
        for (unsigned int i = variables.size(), j = averaging_variables.size(); i>0; --i)
          {
            // Swap averaged values to correct field, and move to next one
            if (variables[i-1] == averaging_variables[j-1])
              {
                data_point.values[i-1].swap(data_point.values[j-1]);
                --j;
              }
            // We are in an adiabatic property field, compute it and move on
            // without decrementing j
            else
              {
                if (variables[i-1] == "adiabatic_temperature")
                  this->get_adiabatic_conditions().get_adiabatic_temperature_profile(data_point.values[i-1]);
                else if (variables[i-1] == "adiabatic_pressure")
                  this->get_adiabatic_conditions().get_adiabatic_pressure_profile(data_point.values[i-1]);
                else if (variables[i-1] == "adiabatic_density")
                  this->get_adiabatic_conditions().get_adiabatic_density_profile(data_point.values[i-1]);
                else if (variables[i-1] == "adiabatic_density_derivative")
                  this->get_adiabatic_conditions().get_adiabatic_density_derivative_profile(data_point.values[i-1]);
                else
                  Assert(false,ExcInternalError());
              }
          }
      }
      entries.push_back (data_point);

      const double max_depth = this->get_geometry_model().maximal_depth();

      // On the root process, write out the file. do this using the DataOutStack
      // class on a piece-wise constant finite element space on
      // a 1d mesh with the correct subdivisions
      std::string filename;
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          Triangulation<1> mesh;
          GridGenerator::subdivided_hyper_cube(mesh, n_depth_zones, 0, max_depth);

          FE_DGQ<1> fe(0);
          DoFHandler<1> dof_handler (mesh);
          dof_handler.distribute_dofs(fe);
          Assert (dof_handler.n_dofs() == n_depth_zones, ExcInternalError());

          DataOutStack<1> data_out_stack;

          if (!ascii_output)
            {
              for (unsigned int j=0; j<variables.size(); ++j)
                data_out_stack.declare_data_vector (variables[j],
                                                    DataOutStack<1>::cell_vector);

              for (unsigned int i=0; i<entries.size(); ++i)
                {
                  data_out_stack.new_parameter_value ((this->convert_output_to_years()
                                                       ?
                                                       entries[i].time / year_in_seconds
                                                       :
                                                       entries[i].time),
                                                      // declare the time step, which here is the difference
                                                      // between successive output times. we don't have anything
                                                      // for the first time step, however. we could do a zero
                                                      // delta, but that leads to invisible output. rather, we
                                                      // use an artificial value of one tenth of the first interval,
                                                      // if available
                                                      (i == 0 ?
                                                       (entries.size() > 1 ? (entries[1].time - entries[0].time)/10 : 0) :
                                                       entries[i].time - entries[i-1].time) /
                                                      (this->convert_output_to_years()
                                                       ?
                                                       year_in_seconds
                                                       :
                                                       1));

                  data_out_stack.attach_dof_handler (dof_handler);

                  Vector<double> tmp(n_depth_zones);
                  for (unsigned int j=0; j<variables.size(); ++j)
                    {
                      std::copy (entries[i].values[j].begin(),
                                 entries[i].values[j].end(),
                                 tmp.begin());
                      data_out_stack.add_data_vector (tmp,
                                                      variables[j]);
                    }
                  data_out_stack.build_patches ();
                  data_out_stack.finish_parameter_value ();
                }


              filename = (this->get_output_directory() +
                          "depth_average" +
                          DataOutBase::default_suffix(output_format));
              std::ofstream f (filename.c_str());
              data_out_stack.write (f, output_format);

              AssertThrow (f, ExcMessage("Writing data to <" + filename +
                                         "> did not succeed in the `point values' "
                                         "postprocessor."));
            }
          else
            {
              filename = (this->get_output_directory() + "depth_average.txt");
              std::ofstream f(filename.c_str(), std::ofstream::out);

              // Write the header
              f << "#       time" << "        depth";
              for ( unsigned int i = 0; i < variables.size(); ++i)
                f << " " << variables[i];
              f << std::endl;

              // Output each data point in the entries object
              for (typename std::vector<DataPoint>::const_iterator point = entries.begin();
                   point != entries.end(); ++point)
                {
                  double depth = max_depth/static_cast<double>(point->values[0].size())/2.0;
                  for (unsigned int d = 0; d < point->values[0].size(); ++d)
                    {
                      f << std::setw(12)
                        << (this->convert_output_to_years() ? point->time/year_in_seconds : point->time)
                        << ' ' << std::setw(12) << depth;
                      for ( unsigned int i = 0; i < variables.size(); ++i )
                        f << ' ' << std::setw(12) << point->values[i][d];
                      f << std::endl;
                      depth+= max_depth/static_cast<double>(point->values[0].size() );
                    }
                }

              AssertThrow (f, ExcMessage("Writing data to <" + filename +
                                         "> did not succeed in the `point values' "
                                         "postprocessor."));
            }
        }

      set_last_output_time (this->get_time());

      // return what should be printed to the screen. note that we had
      // just incremented the number, so use the previous value
      return std::make_pair (std::string ("Writing depth average:"),
                             filename);
    }


    template <int dim>
    void
    DepthAverage<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Depth average");
        {
          prm.declare_entry ("Time between graphical output", "1e8",
                             Patterns::Double (0),
                             "The time interval between each generation of "
                             "graphical output files. A value of zero indicates "
                             "that output should be generated in each time step. "
                             "Units: years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "seconds otherwise.");
          prm.declare_entry ("Number of zones", "10",
                             Patterns::Integer (1),
                             "The number of zones in depth direction within which we "
                             "are to compute averages. By default, we subdivide the entire "
                             "domain into 10 depth zones and compute temperature and other "
                             "averages within each of these zones. However, if you have a "
                             "very coarse mesh, it may not make much sense to subdivide "
                             "the domain into so many zones and you may wish to choose "
                             "less than this default. It may also make computations slightly "
                             "faster. On the other hand, if you have an extremely highly "
                             "resolved mesh, choosing more zones might also make sense.");
          prm.declare_entry ("Output format", "gnuplot",
                             Patterns::Selection(DataOutBase::get_output_format_names().append("|txt")),
                             "The format in which the output shall be produced. The "
                             "format in which the output is generated also determines "
                             "the extension of the file into which data is written.");
          const std::string variables =
            "all|temperature|composition|"
            "adiabatic temperature|adiabatic pressure|adiabatic density|adiabatic density derivative|"
            "velocity magnitude|sinking velocity|Vs|Vp|"
            "viscosity|vertical heat flux";
          prm.declare_entry("List of output variables", "all",
                            Patterns::MultipleSelection(variables.c_str()),
                            "A comma separated list which specifies which quantities to "
                            "average in each depth slice. It defaults to averaging all "
                            "available quantities, but this can be an expensive operation, "
                            "so you may want to select only a few.\n\n"
                            "List of options:\n"
                            +variables);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    DepthAverage<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Depth average");
        {
          output_interval = prm.get_double ("Time between graphical output");
          if (this->convert_output_to_years())
            output_interval *= year_in_seconds;
          n_depth_zones = prm.get_integer ("Number of zones");

          if (output_interval > 0.0)
            {
              // since we increase the time indicating when to write the next graphical output
              // every time we execute the depth average postprocessor, there is no good way to
              // figure out when to write graphical output for the nonlinear iterations if we do
              // not want to output every time step
              AssertThrow(this->get_parameters().run_postprocessors_on_nonlinear_iterations == false,
                          ExcMessage("Postprocessing nonlinear iterations is only supported if every time "
                                     "step is visualized, or in other words, if the 'Time between graphical "
                                     "output' in the Depth average postprocessor is set to zero."));
            }

          std::vector<std::string> output_variables = Utilities::split_string_list(prm.get("List of output variables"));
          AssertThrow(Utilities::has_unique_entries(output_variables),
                      ExcMessage("The list of strings for the parameter "
                                 "'Postprocess/Depth average/List of output variables' contains entries more than once. "
                                 "This is not allowed. Please check your parameter file."));

          bool output_all_variables = false;
          if ( std::find( output_variables.begin(), output_variables.end(), "all") != output_variables.end())
            output_all_variables = true;

          // we have to parse the list in this order to match the output columns
          {
            if (output_all_variables || std::find( output_variables.begin(), output_variables.end(), "temperature") != output_variables.end() )
              variables.push_back("temperature");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "composition") != output_variables.end() )
              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                variables.push_back(std::string("C_") + Utilities::int_to_string(c));

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "adiabatic temperature") != output_variables.end() )
              variables.push_back("adiabatic_temperature");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "adiabatic pressure") != output_variables.end() )
              variables.push_back("adiabatic_pressure");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "adiabatic density") != output_variables.end() )
              variables.push_back("adiabatic_density");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "adiabatic density derivative") != output_variables.end() )
              variables.push_back("adiabatic_density_derivative");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "velocity magnitude") != output_variables.end() )
              variables.push_back("velocity_magnitude");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "sinking velocity") != output_variables.end() )
              variables.push_back("sinking_velocity");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "Vs") != output_variables.end() )
              variables.push_back("Vs");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "Vp") != output_variables.end() )
              variables.push_back("Vp");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "viscosity") != output_variables.end() )
              variables.push_back("viscosity");

            if ( output_all_variables || std::find( output_variables.begin(), output_variables.end(), "vertical heat flux") != output_variables.end() )
              variables.push_back("vertical_heat_flux");
          }

          std::string x_output_format = prm.get("Output format");
          if (x_output_format == "txt")
            ascii_output = true;
          else
            output_format = DataOutBase::parse_output_format(prm.get("Output format"));
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    template <class Archive>
    void DepthAverage<dim>::serialize (Archive &ar, const unsigned int)
    {
      ar &last_output_time
      & entries;
    }


    template <int dim>
    void
    DepthAverage<dim>::save (std::map<std::string, std::string> &status_strings) const
    {
      std::ostringstream os;
      aspect::oarchive oa (os);
      oa << (*this);

      status_strings["DepthAverage"] = os.str();
    }


    template <int dim>
    void
    DepthAverage<dim>::load (const std::map<std::string, std::string> &status_strings)
    {
      // see if something was saved
      if (status_strings.find("DepthAverage") != status_strings.end())
        {
          std::istringstream is (status_strings.find("DepthAverage")->second);
          aspect::iarchive ia (is);
          ia >> (*this);
        }
    }


    template <int dim>
    void
    DepthAverage<dim>::set_last_output_time (const double current_time)
    {
      // if output_interval is positive, then set the next output interval to
      // a positive multiple.
      if (output_interval > 0)
        {
          // We need to find the last time output was supposed to be written.
          // this is the last_output_time plus the largest positive multiple
          // of output_intervals that passed since then. We need to handle the
          // edge case where last_output_time+output_interval==current_time,
          // we did an output and std::floor sadly rounds to zero. This is done
          // by forcing std::floor to round 1.0-eps to 1.0.
          const double magic = 1.0+2.0*std::numeric_limits<double>::epsilon();
          last_output_time = last_output_time + std::floor((current_time-last_output_time)/output_interval*magic) * output_interval/magic;
        }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(DepthAverage,
                                  "depth average",
                                  "A postprocessor that computes depth averaged "
                                  "quantities and writes them into a file <depth_average.ext> "
                                  "in the output directory, where the extension of the file "
                                  "is determined by the output format you select. In addition "
                                  "to the output format, a number of other parameters also influence "
                                  "this postprocessor, and they can be set in the section "
                                  "\\texttt{Postprocess/Depth average} in the input file."
                                  "\n\n"
                                  "In the output files, the $x$-value of each data point corresponds "
                                  "to the depth, whereas the $y$-value corresponds to the "
                                  "simulation time. The time is provided in seconds or, if the "
                                  "global ``Use years in output instead of seconds'' parameter is "
                                  "set, in years.")
  }
}
