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


#include <aspect/simulator.h>
#include <aspect/melt.h>
#include <aspect/newton.h>
#include <aspect/global.h>

#include <aspect/geometry_model/interface.h>
#include <aspect/heating_model/interface.h>
#include <aspect/heating_model/adiabatic_heating.h>
#include <aspect/material_model/interface.h>
#include <aspect/particle/generator/interface.h>
#include <aspect/particle/integrator/interface.h>
#include <aspect/particle/interpolator/interface.h>
#include <aspect/particle/output/interface.h>
#include <aspect/postprocess/visualization.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <locale>
#include <string>


namespace aspect
{

  template <int dim>
  Simulator<dim>::AdvectionField::
  AdvectionField (const FieldType field_type,
                  const unsigned int compositional_variable)
    :
    field_type (field_type),
    compositional_variable (compositional_variable)
  {
    if (field_type == temperature_field)
      Assert (compositional_variable == numbers::invalid_unsigned_int,
              ExcMessage ("You can't specify a compositional variable if you "
                          "have in fact selected the temperature."));
  }



  template <int dim>
  typename Simulator<dim>::AdvectionField
  Simulator<dim>::AdvectionField::temperature ()
  {
    return AdvectionField(temperature_field);
  }



  template <int dim>
  typename Simulator<dim>::AdvectionField
  Simulator<dim>::AdvectionField::composition (const unsigned int compositional_variable)
  {
    return AdvectionField(compositional_field,
                          compositional_variable);
  }


  template <int dim>
  bool
  Simulator<dim>::AdvectionField::is_temperature() const
  {
    return (field_type == temperature_field);
  }

  template <int dim>
  bool
  Simulator<dim>::AdvectionField::is_discontinuous(const Introspection<dim> &introspection) const
  {
    if (field_type == temperature_field)
      return introspection.use_discontinuous_temperature_discretization;
    else if (field_type == compositional_field)
      return introspection.use_discontinuous_composition_discretization;

    Assert (false, ExcInternalError());
    return false;
  }

  template <int dim>
  typename Parameters<dim>::AdvectionFieldMethod::Kind
  Simulator<dim>::AdvectionField::advection_method(const Introspection<dim> &introspection) const
  {
    return introspection.compositional_field_methods[compositional_variable];
  }

  template <int dim>
  unsigned int
  Simulator<dim>::AdvectionField::block_index(const Introspection<dim> &introspection) const
  {
    if (this->is_temperature())
      return introspection.block_indices.temperature;
    else
      return introspection.block_indices.compositional_fields[compositional_variable];
  }

  template <int dim>
  unsigned int
  Simulator<dim>::AdvectionField::component_index(const Introspection<dim> &introspection) const
  {
    if (this->is_temperature())
      return introspection.component_indices.temperature;
    else
      return introspection.component_indices.compositional_fields[compositional_variable];
  }

  template <int dim>
  unsigned int
  Simulator<dim>::AdvectionField::field_index() const
  {
    if (this->is_temperature())
      return 0;
    else
      return compositional_variable + 1;
  }

  template <int dim>
  unsigned int
  Simulator<dim>::AdvectionField::base_element(const Introspection<dim> &introspection) const
  {
    if (this->is_temperature())
      return introspection.base_elements.temperature;
    else
      return introspection.base_elements.compositional_fields;
  }

  template <int dim>
  FEValuesExtractors::Scalar
  Simulator<dim>::AdvectionField::scalar_extractor(const Introspection<dim> &introspection) const
  {
    if (this->is_temperature())
      return introspection.extractors.temperature;
    else
      return introspection.extractors.compositional_fields[compositional_variable];
  }

  template <int dim>
  unsigned int
  Simulator<dim>::AdvectionField::polynomial_degree(const Introspection<dim> &introspection) const
  {
    if (this->is_temperature())
      return introspection.polynomial_degree.temperature;
    else
      return introspection.polynomial_degree.compositional_fields;
  }


  namespace
  {
    /**
     * A function that writes the statistics object into a file.
     *
     * @param stat_file_name The name of the file into which the result
     * should go
     * @param copy_of_table A copy of the table that we're to write. Since
     * this function is called in the background on a separate thread,
     * the actual table might be modified while we are about to write
     * it, so we need to work on a copy. This copy is deleted at the end
     * of this function.
     */
    void do_output_statistics (const std::string stat_file_name,
                               const TableHandler *copy_of_table)
    {
      // write into a temporary file for now so that we don't
      // interrupt anyone who might want to look at the real
      // statistics file while the program is still running
      const std::string tmp_file_name = stat_file_name + " tmp";

      std::ofstream stat_file (tmp_file_name.c_str());
      copy_of_table->write_text (stat_file,
                                 TableHandler::table_with_separate_column_description);
      stat_file.close();

      // now move the temporary file into place
      std::rename(tmp_file_name.c_str(), stat_file_name.c_str());

      // delete the copy now:
      delete copy_of_table;
    }
  }



  template <int dim>
  void Simulator<dim>::write_plugin_graph (std::ostream &out) const
  {
    // write the preamble
    out << "digraph Plugins\n"
        "{\n"
        "  splines=line;\n"
        "  splines=true;\n"
        "  overlap=false;\n"
        "  edge [fontname=\"FreeSans\",\n"
        "        fontsize=\"10\",\n"
        "        labelfontname=\"FreeSans\",\n"
        "        labelfontsize=\"10\",\n"
        "        color=\"black\",\n"
        "        style=\"solid\"];\n"
        "  node [fontname=\"FreeSans\",\n"
        "        fontsize=\"10\",\n"
        "        shape=\"rectangle\",\n"
        "        height=0.2,\n"
        "        width=0.4,\n"
        "        color=\"black\",\n"
        "        fillcolor=\"white\",\n"
        "        style=\"filled\"];\n"
        "  layout=neato;\n"
        "\n";

    // then also write nodes for the Simulator and SimulatorAccess classes,
    // and an arrow from the former to the latter to indicate flow of
    // information
    out << "  Simulator [height=1.5,width=2,shape=\"octagon\",fillcolor=\"yellow\"];\n";
    out << "  SimulatorAccess [height=1.2,width=1.2,shape=\"rect\",fillcolor=\"yellow\"];\n";
    out << "  Simulator -> SimulatorAccess [len=1, weight=100];\n";

    // then go through all plugin systems and output everything we have
    AdiabaticConditions::write_plugin_graph<dim>(out);
    BoundaryComposition::Manager<dim>::write_plugin_graph(out);
    BoundaryFluidPressure::write_plugin_graph<dim>(out);
    BoundaryTemperature::Manager<dim>::write_plugin_graph(out);
    BoundaryTraction::write_plugin_graph<dim>(out);
    BoundaryVelocity::Manager<dim>::write_plugin_graph(out);
    InitialTopographyModel::write_plugin_graph<dim>(out);
    GeometryModel::write_plugin_graph<dim>(out);
    GravityModel::write_plugin_graph<dim>(out);
    HeatingModel::Manager<dim>::write_plugin_graph(out);
    InitialComposition::Manager<dim>::write_plugin_graph(out);
    InitialTemperature::Manager<dim>::write_plugin_graph(out);
    MaterialModel::write_plugin_graph<dim>(out);
    MeshRefinement::Manager<dim>::write_plugin_graph(out);
    Particle::Generator::write_plugin_graph<dim>(out);
    Particle::Integrator::write_plugin_graph<dim>(out);
    Particle::Interpolator::write_plugin_graph<dim>(out);
    Particle::Output::write_plugin_graph<dim>(out);
    Particle::Property::Manager<dim>::write_plugin_graph(out);
    Postprocess::Manager<dim>::write_plugin_graph(out);
    Postprocess::Visualization<dim>::write_plugin_graph(out);
    PrescribedStokesSolution::write_plugin_graph<dim>(out);
    TerminationCriteria::Manager<dim>::write_plugin_graph(out);

    // end the graph
    out << "}"
        << std::endl;
  }



  template <int dim>
  void Simulator<dim>::output_statistics()
  {
    // only write the statistics file from processor zero
    if (Utilities::MPI::this_mpi_process(mpi_communicator)!=0)
      return;

    // formatting the table we're about to output and writing the
    // actual file may take some time, so do it on a separate
    // thread. we pass a pointer to a copy of the statistics
    // object which the called function then has to destroy
    //
    // before we can start working on a new thread, we need to
    // make sure that the previous thread is done or they'll
    // stomp on each other's feet
    output_statistics_thread.join();
    output_statistics_thread = Threads::new_thread (&do_output_statistics,
                                                    parameters.output_directory+"statistics",
                                                    new TableHandler(statistics));
  }



  template <int dim>
  double
  Simulator<dim>::
  get_maximal_velocity (const LinearAlgebra::BlockVector &solution) const
  {
    // use a quadrature formula that has one point at
    // the location of each degree of freedom in the
    // velocity element
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             parameters.stokes_velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size();


    FEValues<dim> fe_values (*mapping, finite_element, quadrature_formula, update_values);
    std::vector<Tensor<1,dim> > velocity_values(n_q_points);

    double max_local_velocity = 0;

    // loop over all locally owned cells and evaluate the velocities at each
    // quadrature point (i.e. each node). keep a running tally of the largest
    // such velocity
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values[introspection.extractors.velocities].get_function_values (solution,
                                                                              velocity_values);

          for (unsigned int q=0; q<n_q_points; ++q)
            max_local_velocity = std::max (max_local_velocity,
                                           velocity_values[q].norm());
        }

    // return the largest value over all processors
    return Utilities::MPI::max (max_local_velocity, mpi_communicator);
  }



  template <int dim>
  bool Simulator<dim>::maybe_do_initial_refinement (const unsigned int max_refinement_level)
  {
    if (pre_refinement_step < parameters.initial_adaptive_refinement)
      {
        if (parameters.timing_output_frequency ==0)
          computing_timer.print_summary ();

        output_statistics();

        // we only want to do the postprocessing here if it is not already done in
        // the nonlinear iteration scheme, which is the case if we run postprocessors
        // on all nonlinear iterations
        if (parameters.run_postprocessors_on_initial_refinement && (!parameters.run_postprocessors_on_nonlinear_iterations))
          postprocess ();

        refine_mesh (max_refinement_level);
        ++pre_refinement_step;
        return true;
      }
    else
      {
        // invalidate the value of pre_refinement_step since it will no longer be used from here on
        pre_refinement_step = std::numeric_limits<unsigned int>::max();
        return false;
      }
  }



  template <int dim>
  void Simulator<dim>::maybe_refine_mesh (const double new_time_step,
                                          unsigned int &max_refinement_level)
  {
    /*
     * see if this is an additional refinement cycle. An additional refinement
     * cycle differs from a regular, because the maximal refinement level allowed
     * is increased by one from this time on.
     */
    if ((parameters.additional_refinement_times.size() > 0)
        &&
        (parameters.additional_refinement_times.front () < time+new_time_step))
      {
        // loop over as many times as this is necessary
        while ((parameters.additional_refinement_times.size() > 0)
               &&
               (parameters.additional_refinement_times.front () < time+new_time_step))
          {
            ++max_refinement_level;
            refine_mesh (max_refinement_level);

            parameters.additional_refinement_times
            .erase (parameters.additional_refinement_times.begin());
          }
      }
    // see if this is a time step where regular refinement is requested
    else if ((timestep_number > 0
              &&
              parameters.adaptive_refinement_interval > 0
              &&
              timestep_number % parameters.adaptive_refinement_interval == 0)
             ||
             (timestep_number == 0 && parameters.adaptive_refinement_interval == 1)
            )
      {
        refine_mesh (max_refinement_level);
      }
  }



  template <int dim>
  void Simulator<dim>::maybe_write_timing_output () const
  {
    bool write_timing_output = false;
    if (parameters.timing_output_frequency <= 1)
      write_timing_output = true;
    else if ((timestep_number > 0) &&
             (timestep_number % parameters.timing_output_frequency == 0))
      write_timing_output = true;

    // if requested output a summary of the current timing information
    if (write_timing_output)
      computing_timer.print_summary ();
  }



  template <int dim>
  bool Simulator<dim>::maybe_write_checkpoint (const time_t last_checkpoint_time,
                                               const std::pair<bool,bool> termination_output)
  {
    bool write_checkpoint = false;
    // If we base checkpoint frequency on timing, measure the time at process 0
    // This prevents race conditions where some processes will checkpoint and others won't
    if (parameters.checkpoint_time_secs > 0)
      {
        int global_do_checkpoint = ((std::time(NULL)-last_checkpoint_time) >=
                                    parameters.checkpoint_time_secs);
        MPI_Bcast(&global_do_checkpoint, 1, MPI_INT, 0, mpi_communicator);

        if (global_do_checkpoint == 1)
          write_checkpoint = true;
      }

    // If we base checkpoint frequency on steps, see if it's time for another checkpoint
    if ((parameters.checkpoint_time_secs == 0) &&
        (parameters.checkpoint_steps > 0) &&
        (timestep_number % parameters.checkpoint_steps == 0))
      write_checkpoint = true;

    // Do a checkpoint if this is the end of simulation,
    // and the termination criteria say to checkpoint at the end.
    if (termination_output.first && termination_output.second)
      write_checkpoint = true;


    // Do a checkpoint if indicated by checkpoint parameters
    if (write_checkpoint)
      {
        create_snapshot();
        // matrices will be regenerated after a resume, so do that here too
        // to be consistent. otherwise we would get different results
        // for a restarted computation than for one that ran straight
        // through
        rebuild_stokes_matrix =
          rebuild_stokes_preconditioner = true;
      }
    return write_checkpoint;
  }



  template <int dim>
  double Simulator<dim>::compute_time_step () const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             parameters.stokes_velocity_degree);

    FEValues<dim> fe_values (*mapping,
                             finite_element,
                             quadrature_formula,
                             update_values |
                             update_gradients |
                             ((parameters.use_conduction_timestep || parameters.include_melt_transport)
                              ?
                              update_quadrature_points
                              :
                              update_default));

    const unsigned int n_q_points = quadrature_formula.size();


    std::vector<Tensor<1,dim> > velocity_values(n_q_points);
    std::vector<Tensor<1,dim> > fluid_velocity_values(n_q_points);
    std::vector<std::vector<double> > composition_values (introspection.n_compositional_fields,std::vector<double> (n_q_points));

    double max_local_speed_over_meshsize = 0;
    double min_local_conduction_timestep = std::numeric_limits<double>::max();

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();


    MaterialModel::MaterialModelInputs<dim> in(n_q_points,
                                               introspection.n_compositional_fields);
    MaterialModel::MaterialModelOutputs<dim> out(n_q_points,
                                                 introspection.n_compositional_fields);

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values[introspection.extractors.velocities].get_function_values (solution,
                                                                              velocity_values);

          double max_local_velocity = 0;
          for (unsigned int q=0; q<n_q_points; ++q)
            max_local_velocity = std::max (max_local_velocity,
                                           velocity_values[q].norm());

          if (parameters.include_melt_transport)
            {
              const FEValuesExtractors::Vector ex_u_f = introspection.variable("fluid velocity").extractor_vector();
              fe_values[ex_u_f].get_function_values (solution,fluid_velocity_values);

              for (unsigned int q=0; q<n_q_points; ++q)
                max_local_velocity = std::max (max_local_velocity,
                                               fluid_velocity_values[q].norm());
            }

          max_local_speed_over_meshsize = std::max(max_local_speed_over_meshsize,
                                                   max_local_velocity
                                                   /
                                                   cell->minimum_vertex_distance());

          if (parameters.use_conduction_timestep)
            {
              in.reinit(fe_values,
                        cell,
                        introspection,
                        solution);

              material_model->evaluate(in, out);


              // Evaluate thermal diffusivity at each quadrature point and
              // calculate the corresponding conduction timestep, if applicable
              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const double k = out.thermal_conductivities[q];
                  const double rho = out.densities[q];
                  const double c_p = out.specific_heat[q];

                  Assert(rho * c_p > 0,
                         ExcMessage ("The product of density and c_P needs to be a "
                                     "non-negative quantity."));

                  const double thermal_diffusivity = k/(rho*c_p);

                  if (thermal_diffusivity > 0)
                    {
                      min_local_conduction_timestep = std::min(min_local_conduction_timestep,
                                                               parameters.CFL_number*pow(cell->minimum_vertex_distance(),2)
                                                               / thermal_diffusivity);
                    }
                }
            }
        }

    const double max_global_speed_over_meshsize
      = Utilities::MPI::max (max_local_speed_over_meshsize, mpi_communicator);

    double min_convection_timestep = std::numeric_limits<double>::max();
    double min_conduction_timestep = std::numeric_limits<double>::max();

    if (max_global_speed_over_meshsize != 0.0)
      min_convection_timestep = parameters.CFL_number / (parameters.temperature_degree * max_global_speed_over_meshsize);

    if (parameters.use_conduction_timestep)
      min_conduction_timestep = - Utilities::MPI::max (-min_local_conduction_timestep, mpi_communicator);

    double new_time_step = std::min(min_convection_timestep,
                                    min_conduction_timestep);

    if (new_time_step == std::numeric_limits<double>::max())
      {
        // In some models the velocity is zero, either because that is the prescribed
        // Stokes solution, or just because there is no buoyancy and nothing is moving.
        // If this is the case, and if we either do not compute the conduction time
        // step or do not have any conduction, it is somewhat arbitrary what time step
        // we should choose. In that case, set the time step to the 'Maximum time step'.
        new_time_step = parameters.maximum_time_step;
      }

    // make sure that the timestep doesn't increase too fast
    if (time_step != 0)
      new_time_step = std::min(new_time_step, time_step + time_step * parameters.maximum_relative_increase_time_step);

    new_time_step = termination_manager.check_for_last_time_step(std::min(new_time_step,
                                                                          parameters.maximum_time_step));

    return new_time_step;
  }



  template <int dim>
  std::pair<double,double>
  Simulator<dim>::
  get_extrapolated_advection_field_range (const AdvectionField &advection_field) const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             advection_field.polynomial_degree(introspection));

    const unsigned int n_q_points = quadrature_formula.size();

    const FEValuesExtractors::Scalar field = advection_field.scalar_extractor(introspection);

    FEValues<dim> fe_values (*mapping, finite_element, quadrature_formula,
                             update_values);
    std::vector<double> old_field_values(n_q_points);
    std::vector<double> old_old_field_values(n_q_points);

    // This presets the minimum with a bigger
    // and the maximum with a smaller number
    // than one that is going to appear. Will
    // be overwritten in the cell loop or in
    // the communication step at the
    // latest.
    double min_local_field = std::numeric_limits<double>::max(),
           max_local_field = -std::numeric_limits<double>::max();

    if (timestep_number > 1)
      {
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values[field].get_function_values (old_solution,
                                                    old_field_values);
              fe_values[field].get_function_values (old_old_solution,
                                                    old_old_field_values);

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const double extrapolated_field =
                    (1. + time_step/old_time_step) * old_field_values[q]-
                    time_step/old_time_step * old_old_field_values[q];

                  min_local_field = std::min (min_local_field,
                                              extrapolated_field);
                  max_local_field = std::max (max_local_field,
                                              extrapolated_field);
                }
            }
      }
    else
      {
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values[field].get_function_values (old_solution,
                                                    old_field_values);

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const double extrapolated_field = old_field_values[q];

                  min_local_field = std::min (min_local_field,
                                              extrapolated_field);
                  max_local_field = std::max (max_local_field,
                                              extrapolated_field);
                }
            }
      }

    return std::make_pair(Utilities::MPI::min (min_local_field,
                                               mpi_communicator),
                          Utilities::MPI::max (max_local_field,
                                               mpi_communicator));
  }


  template <int dim>
  void Simulator<dim>::interpolate_onto_velocity_system(const TensorFunction<1,dim> &func,
                                                        LinearAlgebra::Vector &vec)
  {
    ConstraintMatrix hanging_constraints(introspection.index_sets.system_relevant_set);
    DoFTools::make_hanging_node_constraints(dof_handler, hanging_constraints);
    hanging_constraints.close();

    Assert(introspection.block_indices.velocities == 0, ExcNotImplemented());
    const std::vector<Point<dim> > mesh_support_points = finite_element.base_element(introspection.base_elements.velocities).get_unit_support_points();
    FEValues<dim> mesh_points (*mapping, finite_element, mesh_support_points, update_quadrature_points);
    std::vector<types::global_dof_index> cell_dof_indices (finite_element.dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          mesh_points.reinit(cell);
          cell->get_dof_indices (cell_dof_indices);
          for (unsigned int j=0; j<finite_element.base_element(introspection.base_elements.velocities).dofs_per_cell; ++j)
            for (unsigned int dir=0; dir<dim; ++dir)
              {
                unsigned int support_point_index
                  = finite_element.component_to_system_index(/*velocity component=*/ introspection.component_indices.velocities[dir],
                                                                                     /*dof index within component=*/ j);
                Assert(introspection.block_indices.velocities == 0, ExcNotImplemented());
                vec[cell_dof_indices[support_point_index]] = func.value(mesh_points.quadrature_point(j))[dir];
              }
        }

    vec.compress(VectorOperation::insert);
    hanging_constraints.distribute(vec);
  }



  template <int dim>
  double Simulator<dim>::normalize_pressure (LinearAlgebra::BlockVector &vector) const
  {
    if (parameters.pressure_normalization == "no")
      return 0;

    const FEValuesExtractors::Scalar &extractor_pressure =
      (parameters.include_melt_transport ?
       introspection.variable("fluid pressure").extractor_scalar()
       : introspection.extractors.pressure);

    double my_pressure = 0.0;
    double my_area = 0.0;
    if (parameters.pressure_normalization == "surface")
      {
        QGauss < dim - 1 > quadrature (parameters.stokes_velocity_degree + 1);

        const unsigned int n_q_points = quadrature.size();
        FEFaceValues<dim> fe_face_values (*mapping, finite_element,  quadrature,
                                          update_JxW_values | update_values);

        std::vector<double> pressure_values(n_q_points);

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
                {
                  const typename DoFHandler<dim>::face_iterator face = cell->face (face_no);
                  if (face->at_boundary()
                      &&
                      (geometry_model->depth (face->center()) <
                       (face->diameter() / std::sqrt(1.*dim-1) / 3)))
                    {
                      fe_face_values.reinit (cell, face_no);
                      fe_face_values[extractor_pressure].get_function_values(vector,
                                                                             pressure_values);

                      for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                          my_pressure += pressure_values[q]
                                         * fe_face_values.JxW (q);
                          my_area += fe_face_values.JxW (q);
                        }
                    }
                }
            }
      }
    else if (parameters.pressure_normalization == "volume")
      {
        const QGauss<dim> quadrature (parameters.stokes_velocity_degree + 1);

        const unsigned int n_q_points = quadrature.size();
        FEValues<dim> fe_values (*mapping, finite_element,  quadrature,
                                 update_JxW_values | update_values);

        std::vector<double> pressure_values(n_q_points);

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values[extractor_pressure].get_function_values(vector,
                                                                pressure_values);

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  my_pressure += pressure_values[q]
                                 * fe_values.JxW (q);
                  my_area += fe_values.JxW (q);
                }
            }
      }
    else
      AssertThrow (false, ExcMessage("Invalid pressure normalization method: " +
                                     parameters.pressure_normalization));

    // sum up the integrals from each processor and compute the result we care about
    double pressure_adjustment = numbers::signaling_nan<double>();
    {
      const double my_temp[2] = {my_pressure, my_area};
      double temp[2];
      Utilities::MPI::sum (my_temp, mpi_communicator, temp);

      if (parameters.pressure_normalization == "surface")
        pressure_adjustment = -temp[0]/temp[1] + parameters.surface_pressure;
      else if (parameters.pressure_normalization == "volume")
        pressure_adjustment = -temp[0]/temp[1];
      else
        AssertThrow(false, ExcNotImplemented());
    }

    // A complication is that we can't modify individual
    // elements of the solution vector since that one has ghost element.
    // rather, we first need to localize it and then distribute back
    LinearAlgebra::BlockVector distributed_vector (introspection.index_sets.system_partitioning,
                                                   mpi_communicator);
    distributed_vector = vector;

    if (parameters.use_locally_conservative_discretization == false)
      {
        if (introspection.block_indices.velocities != introspection.block_indices.pressure
            && !parameters.include_melt_transport)
          distributed_vector.block(introspection.block_indices.pressure).add(pressure_adjustment);
        else
          {
            // pressure is not in a separate block, so we have to modify the values manually
            const unsigned int pressure_component = (parameters.include_melt_transport ?
                                                     introspection.variable("fluid pressure").first_component_index
                                                     : introspection.component_indices.pressure);
            const unsigned int n_local_pressure_dofs = (parameters.include_melt_transport ?
                                                        finite_element.base_element(introspection.variable("fluid pressure").base_index).dofs_per_cell
                                                        : finite_element.base_element(introspection.base_elements.pressure).dofs_per_cell);
            std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);
            typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
            for (; cell != endc; ++cell)
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices (local_dof_indices);
                  for (unsigned int j=0; j<n_local_pressure_dofs; ++j)
                    {
                      unsigned int support_point_index
                        = finite_element.component_to_system_index(pressure_component,
                                                                   /*dof index within component=*/ j);

                      // then adjust its value. Note that because we end up touching
                      // entries more than once, we are not simply incrementing
                      // distributed_vector but copy from the unchanged vector.
                      distributed_vector(local_dof_indices[support_point_index]) = vector(local_dof_indices[support_point_index]) + pressure_adjustment;
                    }
                }
            distributed_vector.compress(VectorOperation::insert);
          }
      }
    else
      {
        // this case is a bit more complicated: if the condition above is false
        // then we use the FE_DGP element for which the shape functions do not
        // add up to one; consequently, adding a constant to all degrees of
        // freedom does not alter the overall function by that constant, but
        // by something different
        //
        // we can work around this by using the documented property of the
        // FE_DGP element that the first shape function is constant.
        // consequently, adding the adjustment to the global function is
        // achieved by adding the adjustment to the first pressure degree
        // of freedom on each cell.
        Assert (dynamic_cast<const FE_DGP<dim>*>(&finite_element.base_element(introspection.base_elements.pressure)) != 0,
                ExcInternalError());
        const unsigned int pressure_component = (parameters.include_melt_transport ?
                                                 introspection.variable("fluid pressure").first_component_index
                                                 : introspection.component_indices.pressure);
        std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              // identify the first pressure dof
              cell->get_dof_indices (local_dof_indices);
              const unsigned int first_pressure_dof
                = finite_element.component_to_system_index (pressure_component, 0);

              // make sure that this DoF is really owned by the current processor
              // and that it is in fact a pressure dof
              Assert (dof_handler.locally_owned_dofs().is_element(local_dof_indices[first_pressure_dof]),
                      ExcInternalError());

              // then adjust its value
              distributed_vector(local_dof_indices[first_pressure_dof]) = vector(local_dof_indices[first_pressure_dof])
                                                                          + pressure_adjustment;
            }
        distributed_vector.compress(VectorOperation::insert);
      }

    // now get back to the original vector and return the adjustment used
    // in the computations above
    vector = distributed_vector;

    return pressure_adjustment;
  }



  template <int dim>
  void
  Simulator<dim>::
  denormalize_pressure (const double                      pressure_adjustment,
                        LinearAlgebra::BlockVector       &vector,
                        const LinearAlgebra::BlockVector &relevant_vector) const
  {
    if (parameters.pressure_normalization == "no")
      return;

    if (parameters.use_locally_conservative_discretization == false)
      {
        if ((introspection.block_indices.velocities != introspection.block_indices.pressure)
            && !parameters.include_melt_transport)
          vector.block(introspection.block_indices.pressure).add(-1.0 * pressure_adjustment);
        else
          {
            // pressure is not in a separate block so we have to modify the values manually
            const unsigned int pressure_component = (parameters.include_melt_transport ?
                                                     introspection.variable("fluid pressure").first_component_index
                                                     : introspection.component_indices.pressure);
            const unsigned int n_local_pressure_dofs = (parameters.include_melt_transport ?
                                                        finite_element.base_element(introspection.variable("fluid pressure").base_index).dofs_per_cell
                                                        : finite_element.base_element(introspection.base_elements.pressure).dofs_per_cell);

            std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);
            typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
            for (; cell != endc; ++cell)
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices (local_dof_indices);
                  for (unsigned int j=0; j<n_local_pressure_dofs; ++j)
                    {
                      const unsigned int local_dof_index
                        = finite_element.component_to_system_index(pressure_component,
                                                                   /*dof index within component=*/ j);

                      // then adjust its value. Note that because we end up touching
                      // entries more than once, we are not simply incrementing
                      // distributed_vector but copy from the unchanged vector.
                      vector(local_dof_indices[local_dof_index])
                        = relevant_vector(local_dof_indices[local_dof_index]) - pressure_adjustment;
                    }
                }
            vector.compress(VectorOperation::insert);
          }
      }
    else
      {
        // this case is a bit more complicated: if the condition above is false
        // then we use the FE_DGP element for which the shape functions do not
        // add up to one; consequently, adding a constant to all degrees of
        // freedom does not alter the overall function by that constant, but
        // by something different
        //
        // we can work around this by using the documented property of the
        // FE_DGP element that the first shape function is constant.
        // consequently, adding the adjustment to the global function is
        // achieved by adding the adjustment to the first pressure degree
        // of freedom on each cell.
        Assert (dynamic_cast<const FE_DGP<dim>*>(&finite_element.base_element(introspection.base_elements.pressure)) != 0,
                ExcInternalError());
        Assert(!parameters.include_melt_transport, ExcNotImplemented());
        const unsigned int pressure_component = introspection.component_indices.pressure;
        std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              // identify the first pressure dof
              cell->get_dof_indices (local_dof_indices);
              const unsigned int first_pressure_dof
                = finite_element.component_to_system_index (pressure_component, 0);

              // make sure that this DoF is really owned by the current processor
              // and that it is in fact a pressure dof
              Assert (dof_handler.locally_owned_dofs().is_element(local_dof_indices[first_pressure_dof]),
                      ExcInternalError());

              // then adjust its value
              vector (local_dof_indices[first_pressure_dof]) = relevant_vector(local_dof_indices[first_pressure_dof])
                                                               - pressure_adjustment;
            }

        vector.compress(VectorOperation::insert);
      }
  }



  template <int dim>
  void
  Simulator<dim>::make_pressure_rhs_compatible(LinearAlgebra::BlockVector &vector)
  {
    // If the mass conservation is written as
    //   div u = f
    // make sure this is solvable by modifying f to ensure that
    // int_\Omega f = int_\Omega div u = 0
    //
    // We have to deal with several complications:
    // - we can have an FE_Q or an FE_DGP for the pressure
    // - we might use a direct solver, so pressure and velocity is in the same block
    // - we might have melt transport, where we need to operate only on p_f
    //
    // We ensure int_\Omega f = 0 by computing a correction factor
    //   c = \int f
    // and adjust pressure RHS to be
    //  fnew = f - c/|\Omega|
    // such that
    //   \int fnew = \int f - c/|\Omega| = -c + \int f = 0.
    //
    // We can compute
    //   c = \int f = (f, 1) = (f, \sum_i \phi_i) = \sum_i (f, \phi_i) = \sum_i F_i
    // which is just the sum over the RHS vector for FE_Q. For FE_DGP we need
    // to restrict to 0th shape functions on each cell because this is how we
    // represent the function 1.
    //
    // To make the adjustment fnew = f - c/|\Omega|
    // note that
    // fnew_i = f_i - c/|\Omega| * (1, \phi_i)
    // and the same logic for FE_DGP applies


    if ((!parameters.use_locally_conservative_discretization)
        &&
        (!parameters.include_melt_transport)
        &&
        (introspection.block_indices.velocities != introspection.block_indices.pressure))
      {
        // Easy Case. We have an FE_Q in a separate block, so we can use
        // mean_value() and vector.block(p) += correction:
        const double mean = vector.block(introspection.block_indices.pressure).mean_value();
        const double int_rhs = mean * vector.block(introspection.block_indices.pressure).size();
        const double correction = -int_rhs / global_volume;

        vector.block(introspection.block_indices.pressure).add(correction, pressure_shape_function_integrals.block(introspection.block_indices.pressure));
      }
    else if (!parameters.use_locally_conservative_discretization)
      {
        // FE_Q but we can not access the pressure block separately (either
        // a direct solver or we have melt with p_f and p_c in the same block).
        // Luckily we don't need to go over DoFs on each cell, because we
        // have IndexSets to help us:

        // we need to operate only on p_f not on p_c
        const IndexSet &idxset = parameters.include_melt_transport ?
                                 introspection.index_sets.locally_owned_fluid_pressure_dofs
                                 :
                                 introspection.index_sets.locally_owned_pressure_dofs;
        double int_rhs = 0.0;

        for (unsigned int i=0; i < idxset.n_elements(); ++i)
          {
            types::global_dof_index idx = idxset.nth_index_in_set(i);
            int_rhs += vector(idx);
          }

        // We do not have to integrate over the normal velocity at the
        // boundaries with a prescribed velocity because the constraints
        // are already distributed to the right hand side in
        // current_constraints.distribute.
        const double global_int_rhs = Utilities::MPI::sum(int_rhs, mpi_communicator);
        const double correction = - global_int_rhs / global_volume;

        for (unsigned int i=0; i < idxset.n_elements(); ++i)
          {
            types::global_dof_index idx = idxset.nth_index_in_set(i);
            vector(idx) += correction * pressure_shape_function_integrals(idx);
          }

        vector.compress(VectorOperation::add);
      }
    else
      {
        // Locally conservative with or without direct solver and with or
        // without melt: grab a pickaxe and do everything by hand!
        AssertThrow(parameters.use_locally_conservative_discretization,
                    ExcInternalError());

        double int_rhs = 0.0;
        const unsigned int pressure_component = (parameters.include_melt_transport ?
                                                 introspection.variable("fluid pressure").first_component_index
                                                 : introspection.component_indices.pressure);
        std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              // identify the first pressure dof
              cell->get_dof_indices (local_dof_indices);
              const unsigned int first_pressure_dof
                = finite_element.component_to_system_index (pressure_component, 0);

              // make sure that this DoF is really owned by the current processor
              // and that it is in fact a pressure dof
              Assert (dof_handler.locally_owned_dofs().is_element(local_dof_indices[first_pressure_dof]),
                      ExcInternalError());

              // compute integral:
              int_rhs += vector(local_dof_indices[first_pressure_dof]);
            }

        const double global_int_rhs = Utilities::MPI::sum(int_rhs, mpi_communicator);
        const double correction = - global_int_rhs / global_volume;

        // Now modify our RHS with the correction factor:
        for (cell = dof_handler.begin_active(); cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              // identify the first pressure dof
              cell->get_dof_indices (local_dof_indices);
              const unsigned int first_pressure_dof
                = finite_element.component_to_system_index (pressure_component, 0);

              // make sure that this DoF is really owned by the current processor
              // and that it is in fact a pressure dof
              Assert (dof_handler.locally_owned_dofs().is_element(local_dof_indices[first_pressure_dof]),
                      ExcInternalError());

              // correct:
              types::global_dof_index idx = local_dof_indices[first_pressure_dof];
              vector(idx) += correction * pressure_shape_function_integrals(idx);
            }

        vector.compress(VectorOperation::add);
      }


  }


  template <int dim>
  double
  Simulator<dim>::compute_initial_stokes_residual()
  {
    LinearAlgebra::BlockVector linearized_stokes_variables (introspection.index_sets.stokes_partitioning, mpi_communicator);
    LinearAlgebra::BlockVector residual (introspection.index_sets.stokes_partitioning, mpi_communicator);
    const unsigned int block_p =
      parameters.include_melt_transport ?
      introspection.variable("fluid pressure").block_index
      :
      introspection.block_indices.pressure;

    // if velocity and pressure are in the same block, we have to copy the
    // pressure to the solution and RHS vector with a zero velocity
    if (block_p == introspection.block_indices.velocities)
      {
        const IndexSet &idxset = (parameters.include_melt_transport) ?
                                 introspection.index_sets.locally_owned_fluid_pressure_dofs
                                 :
                                 introspection.index_sets.locally_owned_pressure_dofs;

        for (unsigned int i=0; i < idxset.n_elements(); ++i)
          {
            types::global_dof_index idx = idxset.nth_index_in_set(i);
            linearized_stokes_variables(idx)        = current_linearization_point(idx);
          }
        linearized_stokes_variables.block(block_p).compress(VectorOperation::insert);
      }
    else
      linearized_stokes_variables.block (block_p) = current_linearization_point.block (block_p);

    // TODO: we don't have .stokes_relevant_partitioning so I am creating a much
    // bigger vector here, oh well.
    LinearAlgebra::BlockVector ghosted (introspection.index_sets.system_partitioning,
                                        introspection.index_sets.system_relevant_partitioning,
                                        mpi_communicator);
    // TODO for Timo: can we create the ghost vector inside of denormalize_pressure
    // (only in cases where we need it)
    ghosted.block(block_p) = linearized_stokes_variables.block(block_p);
    denormalize_pressure (this->last_pressure_normalization_adjustment, linearized_stokes_variables, ghosted);
    current_constraints.set_zero (linearized_stokes_variables);

    linearized_stokes_variables.block (block_p) /= pressure_scaling;

    // we calculate the velocity residual with a zero velocity,
    // computing only the part of the RHS not balanced by the static pressure
    if (block_p == introspection.block_indices.velocities)
      {
        // we can use the whole block here because we set the velocity to zero above
        return system_matrix.block(0,0).residual (residual.block(0),
                                                  linearized_stokes_variables.block(0),
                                                  system_rhs.block(0));
      }
    else
      {
        const double residual_u = system_matrix.block(0,1).residual (residual.block(0),
                                                                     linearized_stokes_variables.block(1),
                                                                     system_rhs.block(0));
        const double residual_p = system_rhs.block(block_p).l2_norm();
        return std::sqrt(residual_u*residual_u+residual_p*residual_p);
      }
  }



  template <int dim>
  bool
  Simulator<dim>::stokes_matrix_depends_on_solution() const
  {
    // Currently, the only coefficient that really appears on the
    // left hand side of the Stokes equation is the viscosity and possibly
    // the density in the case of the implicit reference density profile
    // approximation.
    // If melt transport is included in the simulation, we have an
    // additional equation with more coefficients on the left hand
    // side.

    return (material_model->get_model_dependence().viscosity != MaterialModel::NonlinearDependence::none)
           || (parameters.formulation_mass_conservation ==
               Parameters<dim>::Formulation::MassConservation::implicit_reference_density_profile)
           || parameters.include_melt_transport;
  }



  template <int dim>
  void Simulator<dim>::apply_limiter_to_dg_solutions (const AdvectionField &advection_field)
  {
    /*
     * First setup the quadrature points which are used to find the maximum and minimum solution values at those points.
     * A quadrature formula that combines all quadrature points constructed as all tensor products of
     * 1) one dimensional Gauss points; 2) one dimensional Gauss-Lobatto points.
     * We require that the Gauss-Lobatto points (2) appear in only one direction.
     * Therefore, possible combination
     * in 2D: the combinations are 21, 12
     * in 3D: the combinations are 211, 121, 112
     */
    const QGauss<1> quadrature_formula_1 (advection_field.polynomial_degree(introspection)+1);
    const QGaussLobatto<1> quadrature_formula_2 (advection_field.polynomial_degree(introspection)+1);

    const unsigned int n_q_points_1 = quadrature_formula_1.size();
    const unsigned int n_q_points_2 = quadrature_formula_2.size();
    const unsigned int n_q_points   = dim * n_q_points_2 *std::pow(n_q_points_1, dim-1) ;

    std::vector< Point <dim> > quadrature_points (n_q_points);

    switch (dim)
      {
        case 2:
        {
          // append quadrature points combination 12
          for ( unsigned int i=0; i < n_q_points_1 ; i++)
            {
              const double  x = quadrature_formula_1.point(i)(0);
              for ( unsigned int j=0; j < n_q_points_2 ; j++)
                {
                  const double  y = quadrature_formula_2.point(j)(0);
                  quadrature_points[i * n_q_points_2+j] = Point<dim>(x,y);
                }
            }
          const unsigned int n_q_points_12 = n_q_points_1 * n_q_points_2;
          // append quadrature points combination 21
          for ( unsigned int i=0; i < n_q_points_2 ; i++)
            {
              const double  x = quadrature_formula_2.point(i)(0);
              for ( unsigned int j=0; j < n_q_points_1 ; j++)
                {
                  const double  y = quadrature_formula_1.point(j)(0);
                  quadrature_points[n_q_points_12 + i * n_q_points_1+j ] = Point<dim>(x,y);
                }
            }
          break;
        }

        case 3:
        {
          // append quadrature points combination 121
          for ( unsigned int i=0; i < n_q_points_1 ; i++)
            {
              const double  x = quadrature_formula_1.point(i)(0);
              for ( unsigned int j=0; j < n_q_points_2 ; j++)
                {
                  const double  y = quadrature_formula_2.point(j)(0);
                  for ( unsigned int k=0; k < n_q_points_1 ; k++)
                    {
                      const unsigned int k_index = i * n_q_points_2 * n_q_points_1 + j * n_q_points_2 + k;
                      const double  z = quadrature_formula_1.point(k)(0);
                      quadrature_points[k_index] = Point<dim>(x,y,z);
                    }
                }
            }
          const unsigned int n_q_points_121 = n_q_points_1 * n_q_points_2 * n_q_points_1;
          // append quadrature points combination 112
          for ( unsigned int i=0; i < n_q_points_1 ; i++)
            {
              const double  x = quadrature_formula_1.point(i)(0);
              for ( unsigned int j=0; j < n_q_points_1 ; j++)
                {
                  const double y = quadrature_formula_1.point(j)(0);
                  for ( unsigned int k=0; k < n_q_points_2 ; k++)
                    {
                      const unsigned int k_index =
                        n_q_points_121 + i * n_q_points_1 * n_q_points_2 + j * n_q_points_2 + k;
                      const double  z = quadrature_formula_2.point(k)(0);
                      quadrature_points[k_index] = Point<dim>(x,y,z);
                    }
                }
            }
          // append quadrature points combination 211
          for ( unsigned int i=0; i < n_q_points_2 ; i++)
            {
              const double  x = quadrature_formula_2.point(i)(0);
              for ( unsigned int j=0; j < n_q_points_1 ; j++)
                {
                  const double  y = quadrature_formula_1.point(j)(0);
                  for ( unsigned int k=0; k < n_q_points_1 ; k++)
                    {
                      const unsigned int k_index =
                        2 * n_q_points_121 + i * n_q_points_2 * n_q_points_1 + j * n_q_points_1 + k;
                      const double  z = quadrature_formula_1.point(k)(0);
                      quadrature_points[k_index] = Point<dim>(x,y,z);
                    }
                }
            }
          break;
        }

        default:
          Assert (false, ExcNotImplemented());
      }
    Quadrature<dim> quadrature_formula(quadrature_points);

    // Quadrature rules only used for the numerical integration for better accuracy
    const QGauss<dim> quadrature_formula_0 (advection_field.polynomial_degree(introspection)+1);

    const unsigned int n_q_points_0 = quadrature_formula_0.size();

    // fe values for points evaluation
    FEValues<dim> fe_values (*mapping,
                             finite_element,
                             quadrature_formula,
                             update_values   |
                             update_quadrature_points);
    std::vector<double> values (n_q_points);
    // fe values for numerical integration, with a number of quadrature points
    // that is equal to 1/dim times the number of total points above
    FEValues<dim> fe_values_0 (*mapping,
                               finite_element,
                               quadrature_formula_0,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);
    std::vector<double> values_0 (n_q_points_0);

    const FEValuesExtractors::Scalar field
      = (advection_field.is_temperature()
         ?
         introspection.extractors.temperature
         :
         introspection.extractors.compositional_fields[advection_field.compositional_variable]
        );

    const double max_solution_exact_global = (advection_field.is_temperature()
                                              ?
                                              parameters.global_temperature_max_preset
                                              :
                                              parameters.global_composition_max_preset[advection_field.compositional_variable]
                                             );
    const double min_solution_exact_global = (advection_field.is_temperature()
                                              ?
                                              parameters.global_temperature_min_preset
                                              :
                                              parameters.global_composition_min_preset[advection_field.compositional_variable]
                                             );

    LinearAlgebra::BlockVector distributed_solution (introspection.index_sets.system_partitioning,
                                                     mpi_communicator);
    const unsigned int block_idx = advection_field.block_index(introspection);
    distributed_solution.block(block_idx) = solution.block(block_idx);

    std::vector<types::global_dof_index> local_dof_indices (finite_element.dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices (local_dof_indices);
            // used to find the maximum, minimum
            fe_values.reinit (cell);
            fe_values[field].get_function_values(solution, values);
            // used for the numerical integration
            fe_values_0.reinit (cell);
            fe_values_0[field].get_function_values(solution, values_0);

            // Find the local max and local min
            const double min_solution_local = *std::min_element (values.begin(), values.end());
            const double max_solution_local = *std::max_element (values.begin(), values.end());
            // Find the trouble cell
            if (min_solution_local < min_solution_exact_global
                || max_solution_local > max_solution_exact_global)
              {
                // Compute the cell area and cell solution average
                double local_area = 0;
                double local_solution_average = 0;
                for (unsigned int q = 0; q < n_q_points_0; ++q)
                  {
                    local_area += fe_values_0.JxW(q);
                    local_solution_average += values_0[q]*fe_values_0.JxW(q);
                  }
                local_solution_average /= local_area;
                /*
                 * Define theta: a scaling constant used to correct the old solution by the formula
                 *   new_value = theta * (old_value-old_solution_cell_average)+old_solution_cell_average
                 * where theta \in [0,1] defined as below.
                 * After the correction, the new solution does not exceed the user-given
                 * exact global maximum/minimum values. Meanwhile, the new solution's cell average
                 * equals to the old solution's cell average.
                 */
                double theta = std::min<double>
                               (1, abs((max_solution_exact_global-local_solution_average)
                                       /(max_solution_local-local_solution_average)));
                theta = std::min<double>
                        (theta, abs((min_solution_exact_global-local_solution_average)
                                    /(min_solution_local-local_solution_average)));
                /* Modify the advection degrees of freedom of the numerical solution.
                 * note that we are using DG elements, so every DoF on a locally owned cell is locally owned;
                 * this means that we do not need to check whether the 'distributed_solution' vector actually
                 *  stores the element we read from/write to here.
                 */
                for (unsigned int j = 0;
                     j < finite_element.base_element(advection_field.base_element(introspection)).dofs_per_cell;
                     ++j)
                  {
                    const unsigned int support_point_index = finite_element.component_to_system_index(
                                                               (advection_field.is_temperature()
                                                                ?
                                                                introspection.component_indices.temperature
                                                                :
                                                                introspection.component_indices.compositional_fields[advection_field.compositional_variable]
                                                               ),
                                                               /*dof index within component=*/ j);
                    const double solution_value = solution(local_dof_indices[support_point_index]);
                    const double limited_solution_value = theta * (solution_value-local_solution_average) + local_solution_average;
                    distributed_solution(local_dof_indices[support_point_index]) = limited_solution_value;
                  }
              }
          }
      }
    distributed_solution.compress(VectorOperation::insert);
    // now get back to the original vector
    solution.block(block_idx) = distributed_solution.block(block_idx);
  }



  template <int dim>
  void Simulator<dim>::compute_reactions ()
  {
    // if the time step has a length of zero, there are no reactions
    if (time_step == 0)
      return;

    // we need some temporary vectors to store our updates to composition and temperature in
    // while we do the time stepping, before we copy them over to the solution vector in the end
    LinearAlgebra::BlockVector distributed_vector (introspection.index_sets.system_partitioning,
                                                   mpi_communicator);

    LinearAlgebra::BlockVector distributed_reaction_vector (introspection.index_sets.system_partitioning,
                                                            mpi_communicator);

    // we use a different (potentially smaller) time step than in the advection scheme,
    // and we want all of our reaction time steps (within one advection step) to have the same size
    const unsigned int number_of_reaction_steps = std::max(static_cast<unsigned int>(time_step / parameters.reaction_time_step),
                                                           std::max(parameters.reaction_steps_per_advection_step,1U));

    const double reaction_time_step_size = time_step / static_cast<double>(number_of_reaction_steps);

    Assert (reaction_time_step_size > 0,
            ExcMessage("Reaction time step must be greater than 0."));

    pcout << "   Solving composition reactions in "
          << number_of_reaction_steps
          << " substep(s)."
          << std::endl;

    // make one fevalues for the composition, and one for the temperature (they might use different finite elements)
    const Quadrature<dim> quadrature_C(dof_handler.get_fe().base_element(introspection.base_elements.compositional_fields).get_unit_support_points());

    FEValues<dim> fe_values_C (*mapping,
                               dof_handler.get_fe(),
                               quadrature_C,
                               update_quadrature_points | update_values | update_gradients);

    std::vector<types::global_dof_index> local_dof_indices (dof_handler.get_fe().dofs_per_cell);
    MaterialModel::MaterialModelInputs<dim> in_C(quadrature_C.size(), introspection.n_compositional_fields);
    MaterialModel::MaterialModelOutputs<dim> out_C(quadrature_C.size(), introspection.n_compositional_fields);
    HeatingModel::HeatingModelOutputs heating_model_outputs_C(quadrature_C.size(), introspection.n_compositional_fields);

    // temperature element
    const Quadrature<dim> quadrature_T(dof_handler.get_fe().base_element(introspection.base_elements.temperature).get_unit_support_points());

    FEValues<dim> fe_values_T (*mapping,
                               dof_handler.get_fe(),
                               quadrature_T,
                               update_quadrature_points | update_values | update_gradients);

    MaterialModel::MaterialModelInputs<dim> in_T(quadrature_T.size(), introspection.n_compositional_fields);
    MaterialModel::MaterialModelOutputs<dim> out_T(quadrature_T.size(), introspection.n_compositional_fields);
    HeatingModel::HeatingModelOutputs heating_model_outputs_T(quadrature_T.size(), introspection.n_compositional_fields);

    // add reaction rate outputs
    material_model->create_additional_named_outputs(out_C);
    material_model->create_additional_named_outputs(out_T);

    MaterialModel::ReactionRateOutputs<dim> *reaction_rate_outputs_C
      = out_C.template get_additional_output<MaterialModel::ReactionRateOutputs<dim> >();

    MaterialModel::ReactionRateOutputs<dim> *reaction_rate_outputs_T
      = out_T.template get_additional_output<MaterialModel::ReactionRateOutputs<dim> >();

    AssertThrow(reaction_rate_outputs_C != NULL && reaction_rate_outputs_T != NULL,
                ExcMessage("You are trying to use the operator splitting solver scheme, "
                           "but the material model you use does not support operator splitting "
                           "(it does not create ReactionRateOutputs, which are required for this "
                           "solver scheme)."));

    // some heating models require the additional outputs
    heating_model_manager.create_additional_material_model_outputs(out_C);
    heating_model_manager.create_additional_material_model_outputs(out_T);

    // Make a loop first over all cells, than over all reaction time steps, and then over
    // all degrees of freedom in each element to compute the reactions. This is possible
    // because the reactions only depend on the temperature and composition values at a given
    // degree of freedom (and are independent of the solution in other points).

    // Note that the values for some degrees of freedom are set more than once in the loop
    // below where we assign the new values to distributed_vector (if they are located on the
    // interface between cells), as we loop over all cells, and then over all degrees of freedom
    // on each cell. Although this means we do some additional work, the results are still
    // correct, as we never read from distributed_vector inside the loop over all cells.
    // We initialize the material model inputs objects in_T and in_C using the solution vector
    // on every cell, compute the update, and then on every cell put the result into the
    // distributed_vector vector. Only after the loop over all cells do we copy distributed_vector
    // back onto the solution vector.
    // So even though we touch some DoF twice, we always start from the same value, compute the
    // same value, and then overwrite the same value in distributed_vector.
    // TODO: make this more effective
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values_C.reinit (cell);
          cell->get_dof_indices (local_dof_indices);
          in_C.reinit(fe_values_C, cell, introspection, solution);

          fe_values_T.reinit (cell);
          in_T.reinit(fe_values_T, cell, introspection, solution);

          std::vector<std::vector<double> > accumulated_reactions_C (quadrature_C.size(),std::vector<double> (introspection.n_compositional_fields));
          std::vector<double> accumulated_reactions_T (quadrature_T.size());

          // Make the reaction time steps: We have to update the values of compositional fields and the temperature.
          // Because temperature and composition might use different finite elements, we loop through their elements
          // separately, and update the temperature and the compositions for both.
          // We can reuse the same material model inputs and outputs structure for each reaction time step.
          // We store the computed updates to temperature and composition in a separate (accumulated_reactions) vector,
          // so that we can later copy it over to the solution vector.
          for (unsigned int i=0; i<number_of_reaction_steps; ++i)
            {
              // Loop over composition element
              material_model->evaluate(in_C, out_C);

              heating_model_manager.evaluate(in_C, out_C, heating_model_outputs_C);

              for (unsigned int j=0; j<dof_handler.get_fe().base_element(introspection.base_elements.compositional_fields).dofs_per_cell; ++j)
                {
                  for (unsigned int c=0; c<introspection.n_compositional_fields; ++c)
                    {
                      // simple forward euler
                      in_C.composition[j][c] = in_C.composition[j][c]
                                               + reaction_time_step_size * reaction_rate_outputs_C->reaction_rates[j][c];
                      accumulated_reactions_C[j][c] += reaction_time_step_size * reaction_rate_outputs_C->reaction_rates[j][c];
                    }
                  in_C.temperature[j] = in_C.temperature[j]
                                        + reaction_time_step_size * heating_model_outputs_C.rates_of_temperature_change[j];
                }

              // loop over temperature element
              material_model->evaluate(in_T, out_T);

              heating_model_manager.evaluate(in_T, out_T, heating_model_outputs_T);

              for (unsigned int j=0; j<dof_handler.get_fe().base_element(introspection.base_elements.temperature).dofs_per_cell; ++j)
                {
                  // simple forward euler
                  in_T.temperature[j] = in_T.temperature[j]
                                        + reaction_time_step_size * heating_model_outputs_T.rates_of_temperature_change[j];
                  accumulated_reactions_T[j] += reaction_time_step_size * heating_model_outputs_T.rates_of_temperature_change[j];

                  for (unsigned int c=0; c<introspection.n_compositional_fields; ++c)
                    in_T.composition[j][c] = in_T.composition[j][c]
                                             + reaction_time_step_size * reaction_rate_outputs_T->reaction_rates[j][c];
                }
            }

          // copy reaction rates and new values for the compositional fields
          for (unsigned int j=0; j<dof_handler.get_fe().base_element(introspection.base_elements.compositional_fields).dofs_per_cell; ++j)
            for (unsigned int c=0; c<introspection.n_compositional_fields; ++c)
              {
                const unsigned int composition_idx
                  = dof_handler.get_fe().component_to_system_index(introspection.component_indices.compositional_fields[c],
                                                                   /*dof index within component=*/ j);

                // skip entries that are not locally owned:
                if (dof_handler.locally_owned_dofs().is_element(local_dof_indices[composition_idx]))
                  {
                    distributed_vector(local_dof_indices[composition_idx]) = in_C.composition[j][c];
                    distributed_reaction_vector(local_dof_indices[composition_idx]) = accumulated_reactions_C[j][c];
                  }
              }

          // copy reaction rates and new values for the temperature field
          for (unsigned int j=0; j<dof_handler.get_fe().base_element(introspection.base_elements.temperature).dofs_per_cell; ++j)
            for (unsigned int c=0; c<introspection.n_compositional_fields; ++c)
              {
                const unsigned int temperature_idx
                  = dof_handler.get_fe().component_to_system_index(introspection.component_indices.temperature,
                                                                   /*dof index within component=*/ j);

                // skip entries that are not locally owned:
                if (dof_handler.locally_owned_dofs().is_element(local_dof_indices[temperature_idx]))
                  {
                    distributed_vector(local_dof_indices[temperature_idx]) = in_T.temperature[j];
                    distributed_reaction_vector(local_dof_indices[temperature_idx]) = accumulated_reactions_T[j];
                  }
              }
        }

    // put the final values into the solution vector
    for (unsigned int c=0; c<introspection.n_compositional_fields; ++c)
      {
        const unsigned int block_c = introspection.block_indices.compositional_fields[c];
        distributed_vector.block(block_c).compress(VectorOperation::insert);
        solution.block(block_c) = distributed_vector.block(block_c);

        // we have to update the old solution with our reaction update too
        // so that the advection scheme will have the correct time stepping in the next step
        distributed_reaction_vector.block(block_c).compress(VectorOperation::insert);

        // we do not need distributed_vector any more, use it to temporarily store the update
        distributed_vector.block(block_c) = old_solution.block(block_c);
        distributed_vector.block(block_c) +=  distributed_reaction_vector.block(block_c);
        old_solution.block(block_c) = distributed_vector.block(block_c);

        distributed_vector.block(block_c) = old_old_solution.block(block_c);
        distributed_vector.block(block_c) +=  distributed_reaction_vector.block(block_c);
        old_old_solution.block(block_c) = distributed_vector.block(block_c);
      }

    const unsigned int block_T = introspection.block_indices.temperature;
    distributed_vector.block(block_T).compress(VectorOperation::insert);
    solution.block(block_T) = distributed_vector.block(block_T);

    // we have to update the old solution with our reaction update too
    // so that the advection scheme will have the correct time stepping in the next step
    distributed_reaction_vector.block(block_T).compress(VectorOperation::insert);

    // we do not need distributed_vector any more, use it to temporarily store the update
    distributed_vector.block(block_T) = old_solution.block(block_T);
    distributed_vector.block(block_T) +=  distributed_reaction_vector.block(block_T);
    old_solution.block(block_T) = distributed_vector.block(block_T);

    distributed_vector.block(block_T) = old_old_solution.block(block_T);
    distributed_vector.block(block_T) +=  distributed_reaction_vector.block(block_T);
    old_old_solution.block(block_T) = distributed_vector.block(block_T);

    current_linearization_point = old_solution;
  }


  template <int dim>
  void
  Simulator<dim>::check_consistency_of_formulation()
  {
    // Replace Formulation::MassConservation::ask_material_model by the respective terms to avoid
    // complicated checks later on
    if (parameters.formulation_mass_conservation == Parameters<dim>::Formulation::MassConservation::ask_material_model)
      {
        if (material_model->is_compressible() == true)
          parameters.formulation_mass_conservation = Parameters<dim>::Formulation::MassConservation::isothermal_compression;
        else
          parameters.formulation_mass_conservation = Parameters<dim>::Formulation::MassConservation::incompressible;
      }

    // Ensure the material model supports the selected formulation of the mass conservation equation
    if (parameters.formulation_mass_conservation == Parameters<dim>::Formulation::MassConservation::incompressible)
      {
        AssertThrow(material_model->is_compressible() == false,
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The mass conservation equation was selected to be incompressible, "
                               "but the provided material model reports that it is compressible. "
                               "Please check the consistency of your material model and selected formulation."));
      }
    else if (parameters.formulation_mass_conservation == Parameters<dim>::Formulation::MassConservation::isothermal_compression
             || parameters.formulation_mass_conservation == Parameters<dim>::Formulation::MassConservation::reference_density_profile
             || parameters.formulation_mass_conservation == Parameters<dim>::Formulation::MassConservation::implicit_reference_density_profile)
      {
        AssertThrow(material_model->is_compressible() == true,
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The mass conservation equation was selected to be compressible, "
                               "but the provided material model reports that it is incompressible. "
                               "Please check the consistency of your material model and selected formulation."));
      }

    // Ensure that the correct heating terms have been selected for the chosen combined formulation
    // Note that if the combined formulation is 'custom' there is no check
    // (useful e.g. for smaller scale lithospheric models with shear heating but without adiabatic heating)
    if (parameters.formulation == Parameters<dim>::Formulation::isothermal_compression)
      {
        AssertThrow(heating_model_manager.adiabatic_heating_enabled(),
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The `isothermal compression' formulation expects adiabatic heating to be enabled, "
                               "but the `adiabatic heating' plugin has not been selected in the input file. "
                               "Please check the consistency of your input file."));

        AssertThrow(heating_model_manager.shear_heating_enabled(),
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The `isothermal compression' formulation expects shear heating to be enabled, "
                               "but the `shear heating' plugin has not been selected in the input file. "
                               "Please check the consistency of your input file."));
      }
    else if (parameters.formulation == Parameters<dim>::Formulation::boussinesq_approximation)
      {
        AssertThrow(!heating_model_manager.adiabatic_heating_enabled(),
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The 'Boussinesq approximation' formulation expects adiabatic heating to be disabled, "
                               "but the `adiabatic heating' plugin has been selected in the input file. "
                               "Please check the consistency of your input file."));

        AssertThrow(!heating_model_manager.shear_heating_enabled(),
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The 'Boussinesq approximation' formulation expects shear heating to be disabled, "
                               "but the `shear heating' plugin has been selected in the input file. "
                               "Please check the consistency of your input file."));
      }
    else if (parameters.formulation == Parameters<dim>::Formulation::anelastic_liquid_approximation)
      {
        AssertThrow(heating_model_manager.adiabatic_heating_enabled(),
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The `anelastic liquid approximation' formulation expects adiabatic heating to be enabled, "
                               "but the `adiabatic heating' plugin has not been selected in the input file. "
                               "Please check the consistency of your input file."));

        AssertThrow(heating_model_manager.shear_heating_enabled(),
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The `anelastic liquid approximation' formulation expects shear heating to be enabled, "
                               "but the `shear heating' plugin has not been selected in the input file. "
                               "Please check the consistency of your input file."));

        const bool use_simplified_adiabatic_heating =
          heating_model_manager.template find_heating_model<HeatingModel::AdiabaticHeating<dim> >()
          ->use_simplified_adiabatic_heating();

        AssertThrow(use_simplified_adiabatic_heating == true,
                    ExcMessage("ASPECT detected an inconsistency in the provided input file. "
                               "The `anelastic liquid approximation' formulation expects adiabatic heating to use "
                               "a simplified heating term that neglects dynamic pressure influences, "
                               "but the adiabatic heating plugin does not report to simplify this term. "
                               "Please check the consistency of your input file."));
      }
  }



  namespace
  {
    /**
     * Return whether t is an element of the given container object.
     */
    template <typename Container>
    bool is_element (const typename Container::value_type &t,
                     const Container                      &container)
    {
      for (typename Container::const_iterator p = container.begin();
           p != container.end();
           ++p)
        if (*p == t)
          return true;

      return false;
    }
  }



  template <int dim>
  void
  Simulator<dim>::check_consistency_of_boundary_conditions() const
  {
    // make sure velocity and traction boundary indicators don't appear in multiple lists
    std::set<types::boundary_id> boundary_indicator_lists[6]
      = { boundary_velocity_manager.get_zero_boundary_velocity_indicators(),
          boundary_velocity_manager.get_tangential_boundary_velocity_indicators(),
          parameters.free_surface_boundary_indicators,
          std::set<types::boundary_id>()   // to be prescribed velocity and traction boundary indicators
        };

    // sets of the boundary indicators only (no selectors and values)
    std::set<types::boundary_id> velocity_bi;
    std::set<types::boundary_id> traction_bi;

    for (std::map<types::boundary_id, std::pair<std::string,std::vector<std::string> > >::const_iterator
         p = boundary_velocity_manager.get_active_boundary_velocity_names().begin();
         p != boundary_velocity_manager.get_active_boundary_velocity_names().end();
         ++p)
      velocity_bi.insert(p->first);

    for (std::map<types::boundary_id,std::pair<std::string, std::string> >::const_iterator
         r = parameters.prescribed_traction_boundary_indicators.begin();
         r != parameters.prescribed_traction_boundary_indicators.end();
         ++r)
      traction_bi.insert(r->first);

    // are there any indicators that occur in both the prescribed velocity and traction list?
    std::set<types::boundary_id> intersection;
    std::set_intersection (velocity_bi.begin(),
                           velocity_bi.end(),
                           traction_bi.begin(),
                           traction_bi.end(),
                           std::inserter(intersection, intersection.end()));

    // if so, do they have different selectors?
    if (!intersection.empty())
      {
        for (std::set<types::boundary_id>::const_iterator
             it = intersection.begin();
             it != intersection.end();
             ++it)
          {
            const std::map<types::boundary_id, std::pair<std::string,std::vector<std::string> > >::const_iterator
            boundary_velocity_names = boundary_velocity_manager.get_active_boundary_velocity_names().find(*it);
            Assert(boundary_velocity_names != boundary_velocity_manager.get_active_boundary_velocity_names().end(),
                   ExcInternalError());

            std::set<char> velocity_selector;
            std::set<char> traction_selector;

            for (std::string::const_iterator
                 it_selector  = boundary_velocity_names->second.first.begin();
                 it_selector != boundary_velocity_names->second.first.end();
                 ++it_selector)
              velocity_selector.insert(*it_selector);

            for (std::string::const_iterator
                 it_selector  = parameters.prescribed_traction_boundary_indicators.find(*it)->second.first.begin();
                 it_selector != parameters.prescribed_traction_boundary_indicators.find(*it)->second.first.end();
                 ++it_selector)
              traction_selector.insert(*it_selector);

            // if there are no selectors specified, throw exception
            AssertThrow(!velocity_selector.empty() || !traction_selector.empty(),
                        ExcMessage ("Boundary indicator <"
                                    +
                                    Utilities::int_to_string(*it)
                                    +
                                    "> with symbolic name <"
                                    +
                                    geometry_model->translate_id_to_symbol_name (*it)
                                    +
                                    "> is listed as having both "
                                    "velocity and traction boundary conditions in the input file."));

            std::set<char> intersection_selector;
            std::set_intersection (velocity_selector.begin(),
                                   velocity_selector.end(),
                                   traction_selector.begin(),
                                   traction_selector.end(),
                                   std::inserter(intersection_selector, intersection_selector.end()));

            // if the same selectors are specified, throw exception
            AssertThrow(intersection_selector.empty(),
                        ExcMessage ("Selectors of boundary indicator <"
                                    +
                                    Utilities::int_to_string(*it)
                                    +
                                    "> with symbolic name <"
                                    +
                                    geometry_model->translate_id_to_symbol_name (*it)
                                    +
                                    "> are listed as having both "
                                    "velocity and traction boundary conditions in the input file."));
          }
      }


    // remove correct boundary indicators that occur in both the velocity and the traction set
    // but have different selectors
    std::set<types::boundary_id> union_set;
    std::set_union (velocity_bi.begin(),
                    velocity_bi.end(),
                    traction_bi.begin(),
                    traction_bi.end(),
                    std::inserter(union_set, union_set.end()));

    // assign the prescribed boundary indicator list to the boundary_indicator_lists
    boundary_indicator_lists[3] = union_set;

    // for each combination of boundary indicator lists, make sure that the
    // intersection is empty
    for (unsigned int i=0; i<sizeof(boundary_indicator_lists)/sizeof(boundary_indicator_lists[0]); ++i)
      for (unsigned int j=i+1; j<sizeof(boundary_indicator_lists)/sizeof(boundary_indicator_lists[0]); ++j)
        {
          std::set<types::boundary_id> intersection;
          std::set_intersection (boundary_indicator_lists[i].begin(),
                                 boundary_indicator_lists[i].end(),
                                 boundary_indicator_lists[j].begin(),
                                 boundary_indicator_lists[j].end(),
                                 std::inserter(intersection, intersection.end()));

          // if the same indicators are specified for different boundary conditions, throw exception
          AssertThrow (intersection.empty(),
                       ExcMessage ("Boundary indicator <"
                                   +
                                   Utilities::int_to_string(*intersection.begin())
                                   +
                                   "> with symbolic name <"
                                   +
                                   geometry_model->translate_id_to_symbol_name (*intersection.begin())
                                   +
                                   "> is listed as having more "
                                   "than one type of velocity or traction boundary condition in the input file."));
        }

    // Check that the periodic boundaries do not have other boundary conditions set
    typedef std::set< std::pair< std::pair< types::boundary_id, types::boundary_id>, unsigned int> >
    periodic_boundary_set;
    periodic_boundary_set pbs = geometry_model->get_periodic_boundary_pairs();

    for (periodic_boundary_set::iterator p = pbs.begin(); p != pbs.end(); ++p)
      {
        // Throw error if we are trying to use the same boundary for more than one boundary condition
        AssertThrow( is_element( (*p).first.first, boundary_temperature_manager.get_fixed_temperature_boundary_indicators() ) == false &&
                     is_element( (*p).first.second, boundary_temperature_manager.get_fixed_temperature_boundary_indicators() ) == false &&
                     is_element( (*p).first.first, boundary_composition_manager.get_fixed_composition_boundary_indicators() ) == false &&
                     is_element( (*p).first.second, boundary_composition_manager.get_fixed_composition_boundary_indicators() ) == false &&
                     is_element( (*p).first.first, boundary_indicator_lists[0] ) == false && // zero velocity
                     is_element( (*p).first.second, boundary_indicator_lists[0] ) == false && // zero velocity
                     is_element( (*p).first.first, boundary_indicator_lists[1] ) == false && // tangential velocity
                     is_element( (*p).first.second, boundary_indicator_lists[1] ) == false && // tangential velocity
                     is_element( (*p).first.first, boundary_indicator_lists[2] ) == false && // free surface
                     is_element( (*p).first.second, boundary_indicator_lists[2] ) == false && // free surface
                     is_element( (*p).first.first, boundary_indicator_lists[3] ) == false && // prescribed traction or velocity
                     is_element( (*p).first.second, boundary_indicator_lists[3] ) == false,  // prescribed traction or velocity
                     ExcMessage("Periodic boundaries must not have boundary conditions set."));
      }

    const std::set<types::boundary_id> all_boundary_indicators
      = geometry_model->get_used_boundary_indicators();
    if (parameters.nonlinear_solver != NonlinearSolver::single_Advection_no_Stokes)
      {
        // next make sure that all listed indicators are actually used by
        // this geometry
        for (unsigned int i=0; i<sizeof(boundary_indicator_lists)/sizeof(boundary_indicator_lists[0]); ++i)
          for (typename std::set<types::boundary_id>::const_iterator
               p = boundary_indicator_lists[i].begin();
               p != boundary_indicator_lists[i].end(); ++p)
            AssertThrow (all_boundary_indicators.find (*p)
                         != all_boundary_indicators.end(),
                         ExcMessage ("One of the boundary indicators listed in the input file "
                                     "is not used by the geometry model."));
      }
    else
      {
        // next make sure that there are no listed indicators
        for (unsigned  int i = 0; i<sizeof(boundary_indicator_lists)/sizeof(boundary_indicator_lists[0]); ++i)
          AssertThrow (boundary_indicator_lists[i].empty(),
                       ExcMessage ("With the solver scheme `single Advection, no Stokes', "
                                   "one cannot set boundary conditions for velocity."));
      }


    // now do the same for the fixed temperature indicators and the
    // compositional indicators
    for (typename std::set<types::boundary_id>::const_iterator
         p = boundary_temperature_manager.get_fixed_temperature_boundary_indicators().begin();
         p != boundary_temperature_manager.get_fixed_temperature_boundary_indicators().end(); ++p)
      AssertThrow (all_boundary_indicators.find (*p)
                   != all_boundary_indicators.end(),
                   ExcMessage ("One of the fixed boundary temperature indicators listed in the input file "
                               "is not used by the geometry model."));
    for (typename std::set<types::boundary_id>::const_iterator
         p = boundary_composition_manager.get_fixed_composition_boundary_indicators().begin();
         p != boundary_composition_manager.get_fixed_composition_boundary_indicators().end(); ++p)
      AssertThrow (all_boundary_indicators.find (*p)
                   != all_boundary_indicators.end(),
                   ExcMessage ("One of the fixed boundary composition indicators listed in the input file "
                               "is not used by the geometry model."));
  }



  template <int dim>
  double
  Simulator<dim>::compute_initial_newton_residual(const LinearAlgebra::BlockVector &linearized_stokes_initial_guess)
  {
    // Store the values of the current_linearization_point and linearized_stokes_initial_guess so we can reset them again.
    LinearAlgebra::BlockVector temp_linearization_point = current_linearization_point;
    LinearAlgebra::BlockVector temp_linearized_stokes_initial_guess = linearized_stokes_initial_guess;
    const unsigned int block_vel = introspection.block_indices.velocities;

    // Set the velocity initial guess to zero, but we use the initial guess for the pressure.
    current_linearization_point.block(introspection.block_indices.velocities) = 0;
    temp_linearized_stokes_initial_guess.block (block_vel) = 0;

    denormalize_pressure (last_pressure_normalization_adjustment,
                          temp_linearized_stokes_initial_guess,
                          current_linearization_point);

    // rebuild the whole system to compute the rhs.
    rebuild_stokes_matrix = assemble_newton_stokes_system = assemble_newton_stokes_matrix = true;
    rebuild_stokes_preconditioner = false;

    compute_current_constraints ();

    assemble_stokes_system();

    last_pressure_normalization_adjustment = normalize_pressure(current_linearization_point);

    const double initial_newton_residual_vel = system_rhs.block(introspection.block_indices.velocities).l2_norm();
    const double initial_newton_residual_p = system_rhs.block(introspection.block_indices.pressure).l2_norm();
    const double initial_newton_residual = std::sqrt(initial_newton_residual_vel * initial_newton_residual_vel + initial_newton_residual_p * initial_newton_residual_p);

    current_linearization_point = temp_linearization_point;

    pcout << "   Initial Newton Stokes residual = " << initial_newton_residual << ", v = " << initial_newton_residual_vel << ", p = " << initial_newton_residual_p << std::endl << std::endl;
    return initial_newton_residual;
  }



  template <int dim>
  double
  Simulator<dim>::compute_Eisenstat_Walker_linear_tolerance(const bool EisenstatWalkerChoiceOne,
                                                            const double maximum_linear_stokes_solver_tolerance,
                                                            const double linear_stokes_solver_tolerance,
                                                            const double stokes_residual,
                                                            const double newton_residual,
                                                            const double newton_residual_old)
  {
    /**
       * The Eisenstat and Walker (1996) method is used for determining the linear tolerance of
       * the iteration after the first iteration. The paper gives two preferred choices of computing
       * this tolerance. Both choices are implemented here with the suggested parameter values and
       * safeguards.
     */
    double new_linear_stokes_solver_tolerance = linear_stokes_solver_tolerance;
    if (EisenstatWalkerChoiceOne)
      {
        // This is the preferred value for this parameter in the paper.
        // A value of 2 for the power-term might also work fine.
        const double powerterm = (1+std::sqrt(5))*0.5;
        if (std::pow(linear_stokes_solver_tolerance,powerterm) <= 0.1)
          {
            new_linear_stokes_solver_tolerance = std::min(maximum_linear_stokes_solver_tolerance,
                                                          std::fabs(newton_residual-stokes_residual)/(newton_residual_old));
          }
        else
          {
            new_linear_stokes_solver_tolerance = std::min(maximum_linear_stokes_solver_tolerance,
                                                          std::max(std::fabs(newton_residual-stokes_residual)/newton_residual_old,
                                                                   std::pow(linear_stokes_solver_tolerance,powerterm)));
          }
      }
    else
      {
        if (0.9*linear_stokes_solver_tolerance * linear_stokes_solver_tolerance <= 0.1)
          {
            new_linear_stokes_solver_tolerance =  std::min(maximum_linear_stokes_solver_tolerance,
                                                           0.9 * std::fabs(newton_residual * newton_residual) /
                                                           (newton_residual_old * newton_residual_old));
          }
        else
          {
            new_linear_stokes_solver_tolerance = std::min(newton_handler->parameters.maximum_linear_stokes_solver_tolerance,
                                                          std::max(0.9 * std::fabs(newton_residual*newton_residual)
                                                                   /
                                                                   (newton_residual_old*newton_residual_old),
                                                                   0.9*linear_stokes_solver_tolerance*linear_stokes_solver_tolerance));
          }
      }
    return new_linear_stokes_solver_tolerance;
  }
}
// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template struct Simulator<dim>::AdvectionField; \
  template double Simulator<dim>::normalize_pressure(LinearAlgebra::BlockVector &vector) const; \
  template void Simulator<dim>::denormalize_pressure(const double pressure_adjustment, \
                                                     LinearAlgebra::BlockVector &vector, \
                                                     const LinearAlgebra::BlockVector &relevant_vector) const; \
  template double Simulator<dim>::get_maximal_velocity (const LinearAlgebra::BlockVector &solution) const; \
  template std::pair<double,double> Simulator<dim>::get_extrapolated_advection_field_range (const AdvectionField &advection_field) const; \
  template void Simulator<dim>::maybe_write_timing_output () const; \
  template bool Simulator<dim>::maybe_write_checkpoint (const time_t last_checkpoint_time, const std::pair<bool,bool> termination_output); \
  template bool Simulator<dim>::maybe_do_initial_refinement (const unsigned int max_refinement_level); \
  template void Simulator<dim>::maybe_refine_mesh (const double new_time_step, unsigned int &max_refinement_level); \
  template double Simulator<dim>::compute_time_step () const; \
  template void Simulator<dim>::make_pressure_rhs_compatible(LinearAlgebra::BlockVector &vector); \
  template void Simulator<dim>::output_statistics(); \
  template void Simulator<dim>::write_plugin_graph(std::ostream &) const; \
  template double Simulator<dim>::compute_initial_stokes_residual(); \
  template bool Simulator<dim>::stokes_matrix_depends_on_solution() const; \
  template void Simulator<dim>::interpolate_onto_velocity_system(const TensorFunction<1,dim> &func, LinearAlgebra::Vector &vec);\
  template void Simulator<dim>::apply_limiter_to_dg_solutions(const AdvectionField &advection_field); \
  template void Simulator<dim>::compute_reactions(); \
  template void Simulator<dim>::check_consistency_of_formulation(); \
  template void Simulator<dim>::check_consistency_of_boundary_conditions() const; \
  template double Simulator<dim>::compute_initial_newton_residual(const LinearAlgebra::BlockVector &linearized_stokes_initial_guess); \
  template double Simulator<dim>::compute_Eisenstat_Walker_linear_tolerance(const bool EisenstatWalkerChoiceOne, \
                                                                            const double maximum_linear_stokes_solver_tolerance, \
                                                                            const double linear_stokes_solver_tolerance, \
                                                                            const double stokes_residual, \
                                                                            const double newton_residual, \
                                                                            const double newton_residual_old);

  ASPECT_INSTANTIATE(INSTANTIATE)
}
