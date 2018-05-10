/*opyright (C) 2011 - 2015 by the authors of the ASPECT code.
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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <aspect/postprocess/gravity.h>
#include <aspect/simulator.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    Gravity<dim>::execute (TableHandler &)
    {
      const unsigned int quadrature_degree = this->get_fe().base_element(this->introspection().base_elements.velocities).degree;
      const QGauss<dim> quadrature_formula(quadrature_degree);

      FEValues<dim> fe_volume_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values |
                               update_gradients |
                               update_q_points | update_JxW_values);

      // have a stream into which we write the data. the text stream is then
      // later sent to processor 0
      std::ostringstream output;

      // Gravitational constant
      double G = 6.67408E-11;

      std::vector<std::pair<Point<dim>,double> > stored_values;

      // loop over all of the surface cells and if one less than h/3 away from
      // the top surface, evaluate the stress at its center
      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();

      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          if (cell->at_boundary())
            {
              // see if the cell is at the *top* or *bottom* boundary, not just any boundary
              unsigned int face_idx = numbers::invalid_unsigned_int;
              for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                {
                  const double depth_face_center = this->get_geometry_model().depth (cell->face(f)->center());
                  const double upper_depth_cutoff = cell->face(f)->minimum_vertex_distance()/3.0;

                  // Check if the face is at the top or bottom boundary
                  if (depth_face_center < upper_depth_cutoff) 
                    {
                      face_idx = f;
                      break;
                    }
                }

              if (face_idx == numbers::invalid_unsigned_int)
                continue;

      // if cell is at surface loop over all cells and integrate density
      const Point<dim> midpoint_at_surface = cell->face(face_idx)->center();      
      const Tensor<1,dim> gravity_down = this->get_gravity_model().gravity_vector(midpoint_at_surface);

      // set local gravity to zero for integration
      Tensor<1,dim> integrated_gravity;

      typename DoFHandler<dim>::active_cell_iterator
      cell_in = this->get_dof_handler().begin_active(),
      endc_in = this->get_dof_handler().end();

      for (; cell_in!=endc_in; ++cell_in)
        {

          fe_volume_values.reinit (cell_in);

          // Evaluate the material model in the cell volume.
 //        MaterialModel::MaterialModelInputs<dim> in_volume(fe_volume_values.n_quadrature_points, this->n_compositional_fields());

          MaterialModel::MaterialModelInputs<dim> in_volume(fe_volume_values,cell_in, this->introspection(), this->get_solution());

          MaterialModel::MaterialModelOutputs<dim> out_volume(fe_volume_values.n_quadrature_points, this->n_compositional_fields());
          this->get_material_model().evaluate(in_volume, out_volume);

          const Point<dim> point_in_cell = cell_in->center();

          // get distance between surface cell (surface face) and midpoint of cell_in
        const Tensor<1,dim> direction =  midpoint_at_surface - point_in_cell; 
	const double distance = direction.norm();

          for (unsigned int q=0; q<quadrature_formula.size(); ++q)
           {
              // get gravtiy 
              const double density = out_volume.densities[q];        
	      
	      // test what's causing nans
              
//	      std::cout << "density " << density << std::endl;		
//	      std::cout << "distance " << distance << std::endl;
//	      std::cout << "direction " << direction << std::endl;

              // multiply by volume and integrate / add to integrated_gravity
              integrated_gravity +=  G * density * fe_volume_values.JxW(q) / (distance * distance) * direction/distance;
           }
        }

//	      std::cout << "gravity " << integrated_gravity << std::endl;

        const double gravity_downward_projection = integrated_gravity * gravity_down;
        stored_values.push_back (std::make_pair(midpoint_at_surface, gravity_downward_projection));
        }

      // Write the solution to an output file
      for (unsigned int i=0; i<stored_values.size(); ++i)
        {
          output << stored_values[i].first
                 << ' '
                 << stored_values[i].second
                 << std::endl;
        }


      const std::string filename = this->get_output_directory() +
                                   "gravity." +
                                   Utilities::int_to_string(this->get_timestep_number(), 5);

      const unsigned int max_data_length = Utilities::MPI::max (output.str().size()+1,
                                                                this->get_mpi_communicator());
      const unsigned int mpi_tag = 123;

      // on processor 0, collect all of the data the individual processors send
      // and concatenate them into one file
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          std::ofstream file (filename.c_str());

          // first write out the data we have created locally
          file << output.str();

          std::string tmp;
          tmp.resize (max_data_length, '\0');

          // then loop through all of the other processors and collect
          // data, then write it to the file
          for (unsigned int p=1; p<Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()); ++p)
            {
              MPI_Status status;
              // get the data. note that MPI says that an MPI_Recv may receive
              // less data than the length specified here. since we have already
              // determined the maximal message length, we use this feature here
              // rather than trying to find out the exact message length with
              // a call to MPI_Probe.
              MPI_Recv (&tmp[0], max_data_length, MPI_CHAR, p, mpi_tag,
                        this->get_mpi_communicator(), &status);

              // output the string. note that 'tmp' has length max_data_length,
              // but we only wrote a certain piece of it in the MPI_Recv, ended
              // by a \0 character. write only this part by outputting it as a
              // C string object, rather than as a std::string
              file << tmp.c_str();
            }
        }
      else
        // on other processors, send the data to processor zero. include the \0
        // character at the end of the string
        {
          MPI_Send (&output.str()[0], output.str().size()+1, MPI_CHAR, 0, mpi_tag,
                    this->get_mpi_communicator());
        }

      return std::pair<std::string,std::string>("Writing gravity:",
                                                filename);
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(Gravity,
                                  "gravity",
                                  "Documentation")
  }
}
