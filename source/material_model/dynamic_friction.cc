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


#include <aspect/material_model/dynamic_friction.h>
#include <aspect/simulator.h>
#include <aspect/utilities.h>

#include <numeric>


namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    const std::vector<double>
    DynamicFriction<dim>::
    compute_volume_fractions( const std::vector<double> &compositional_fields) const
    {
      std::vector<double> volume_fractions( compositional_fields.size()+1);

      //clip the compositional fields so they are between zero and one
      std::vector<double> x_comp = compositional_fields;
      for ( unsigned int i=0; i < x_comp.size(); ++i)
        x_comp[i] = std::min(std::max(x_comp[i], 0.0), 1.0);

      //sum the compositional fields for normalization purposes
      double sum_composition = 0.0;
      for ( unsigned int i=0; i < x_comp.size(); ++i)
        sum_composition += x_comp[i];

      if (sum_composition >= 1.0)
        {
          volume_fractions[0] = 0.0;  //background mantle
          for ( unsigned int i=1; i <= x_comp.size(); ++i)
            volume_fractions[i] = x_comp[i-1]/sum_composition;
        }
      else
        {
          volume_fractions[0] = 1.0 - sum_composition; //background mantle
          for ( unsigned int i=1; i <= x_comp.size(); ++i)
            volume_fractions[i] = x_comp[i-1];
        }
      return volume_fractions;
    }

    template <int dim>
    const std::vector<double>
    DynamicFriction<dim>::
    compute_viscosities(
      const double pressure,
      const SymmetricTensor<2,dim> &strain_rate) const
    {

      std::vector<double> viscosities( mu_s.size());

      // second invariant for strain tensor
      const double edot_ii = ( (this->get_timestep_number() == 0 && strain_rate.norm() <= std::numeric_limits<double>::min())
                               ?
                               reference_strain_rate
                               :
                               std::max(std::sqrt(std::fabs(second_invariant(deviator(strain_rate)))),
                                        minimum_strain_rate) );

      const double strain_rate_dev_inv2 = ( (this->get_timestep_number() == 0 && strain_rate.norm() <= std::numeric_limits<double>::min())
                                            ?
                                            reference_strain_rate * reference_strain_rate
                                            :
                                            std::fabs(second_invariant(deviator(strain_rate))));
      // In later timesteps, we still need to care about cases of very small
      // strain rates. We expect the viscosity to approach the maximum_viscosity
      // in these cases. This check prevents a division-by-zero.
      for (unsigned int i = 0; i < mu_s.size(); i++)
        {
          std::vector<double> mu( mu_s.size());
          std::vector<double> phi( mu_s.size());
          std::vector<double> strength( mu_s.size());
          std::vector<double> viscous_stress( mu_s.size());

          // Calculate viscous stress
          viscous_stress[i] = 2. * background_viscosities[i] * edot_ii;

          // Calculate effective steady-state friction coefficient. The formula below is equivalent to the
          // equation 13 in van Dinther et al., (2013, JGR) . Although here the dynamic friction coefficient
          // is directly specified. In addition, we also use a reference strain rate in place of a characteristic
          // velocity divided by local element size.
          mu[i]  = mu_d[i] + ( mu_s[i] - mu_d[i] ) / ( ( 1 + strain_rate_dev_inv2/reference_strain_rate ) );

          // Convert effective steady-state friction coefficient to internal angle of friction.
          phi[i] = std::atan (mu[i]);

          if (std::sqrt(strain_rate_dev_inv2) <= std::numeric_limits<double>::min())
            viscosities[i] = maximum_viscosity;

          // Drucker Prager yield criterion.
          strength[i] = ( (dim==3)
                          ?
                          ( 6.0 * cohesions[i] * std::cos(phi[i]) + 6.0 * std::max(pressure,0.0) * std::sin(phi[i]) )
                          / ( std::sqrt(3.0) * ( 3.0 + std::sin(phi[i]) ) )
                          :
                          cohesions[i] * std::cos(phi[i]) + std::max(pressure,0.0) * std::sin(phi[i]) );

          // Rescale the viscosity back onto the yield surface
          viscosities[i] = strength[i] / ( 2.0 * std::sqrt(strain_rate_dev_inv2) );

          // Cut off the viscosity between a minimum and maximum value to avoid
          // a numerically unfavourable large viscosity range.
          viscosities[i] = 1.0 / ( ( 1.0 / ( viscosities[i] + minimum_viscosity ) ) + ( 1.0 / maximum_viscosity ) );

        }
      return viscosities;
    }

    template <int dim>
    double
    DynamicFriction<dim>::
    average_value ( const std::vector<double> &volume_fractions,
                    const std::vector<double> &parameter_values,
                    const enum AveragingScheme &average_type) const
    {
      double averaged_parameter = 0.0;

      switch (average_type)
        {
          case arithmetic:
          {
            for (unsigned int i=0; i< volume_fractions.size(); ++i)
              averaged_parameter += volume_fractions[i]*parameter_values[i];
            break;
          }
          case harmonic:
          {
            for (unsigned int i=0; i< volume_fractions.size(); ++i)
              averaged_parameter += volume_fractions[i]/(parameter_values[i]);
            averaged_parameter = 1.0/averaged_parameter;
            break;
          }
          case geometric:
          {
            for (unsigned int i=0; i < volume_fractions.size(); ++i)
              averaged_parameter += volume_fractions[i]*std::log(parameter_values[i]);
            averaged_parameter = std::exp(averaged_parameter);
            break;
          }
          case maximum_composition:
          {
            const unsigned int i = (unsigned int)(std::max_element( volume_fractions.begin(),
                                                                    volume_fractions.end() )
                                                  - volume_fractions.begin());
            averaged_parameter = parameter_values[i];
            break;
          }
          default:
          {
            AssertThrow( false, ExcNotImplemented() );
            break;
          }
        }
      return averaged_parameter;
    }


    template <int dim>
    void
    DynamicFriction<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      for (unsigned int i=0; i < in.position.size(); ++i)
        {

          const std::vector<double> composition = in.composition[i];
          const std::vector<double> volume_fractions = compute_volume_fractions(composition);

          if (in.strain_rate.size() > 0)
            {
              const std::vector<double> viscosities = compute_viscosities(in.pressure[i], in.strain_rate[i]);
              out.viscosities[i] = average_value ( volume_fractions, viscosities, viscosity_averaging);
            }
          out.specific_heat[i] = average_value ( volume_fractions, specific_heats, arithmetic);


          // Arithmetic averaging of thermal conductivities
          // This may not be strictly the most reasonable thing, but for most Earth materials we hope
          // that they do not vary so much that it is a big problem.
          out.thermal_conductivities[i] = average_value ( volume_fractions, thermal_conductivities, arithmetic);

          double density = 0.0;
          for (unsigned int j=0; j < volume_fractions.size(); ++j)
            {
              // not strictly correct if thermal expansivities are different, since we are interpreting
              // these compositions as volume fractions, but the error introduced should not be too bad.
              const double temperature_factor = (1.0 - thermal_expansivities[j] * (in.temperature[i] - reference_T));
              density += volume_fractions[j] * densities[j] * temperature_factor;
            }
          out.densities[i] = density;


          out.thermal_expansion_coefficients[i] = average_value ( volume_fractions, thermal_expansivities, arithmetic);


          // Compressibility at the given positions.
          // The compressibility is given as
          // $\frac 1\rho \frac{\partial\rho}{\partial p}$.
          // (here we use an incompressible medium)
          out.compressibilities[i] = 0.0;
          // Pressure derivative of entropy at the given positions.
          out.entropy_derivative_pressure[i] = 0.0;
          // Temperature derivative of entropy at the given positions.
          out.entropy_derivative_temperature[i] = 0.0;
          // Change in composition due to chemical reactions at the
          // given positions. The term reaction_terms[i][c] is the
          // change in compositional field c at point i.
          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;

        }
    }

    template <int dim>
    double
    DynamicFriction<dim>::
    reference_viscosity () const
    {
      return background_viscosities[0]; //background
    }

    template <int dim>
    bool
    DynamicFriction<dim>::
    is_compressible () const
    {
      return false;
    }

    template <int dim>
    void
    DynamicFriction<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Dynamic Friction");
        {
          prm.declare_entry ("Reference temperature", "293",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. Units: $K$.");
          prm.declare_entry ("Densities", "3300.",
                             Patterns::List(Patterns::Double(0)),
                             "List of densities for background mantle and compositional fields,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value.  Units: $kg / m^3$");
          prm.declare_entry ("Thermal expansivities", "4.e-5",
                             Patterns::List(Patterns::Double(0)),
                             "List of thermal expansivities for background mantle and compositional fields,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value. Units: $1/K$");
          prm.declare_entry ("Specific heats", "1250.",
                             Patterns::List(Patterns::Double(0)),
                             "List of specific heats $C_p$ for background mantle and compositional fields,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value. Units: $J /kg /K$");
          prm.declare_entry ("Thermal conductivities", "4.7",
                             Patterns::List(Patterns::Double(0)),
                             "List of thermal conductivities for background mantle and compositional fields,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value. Units: $W/m/K$ ");
          prm.declare_entry("Viscosity averaging scheme", "harmonic",
                            Patterns::Selection("arithmetic|harmonic|geometric|maximum composition"),
                            "When more than one compositional field is present at a point "
                            "with different viscosities, we need to come up with an average "
                            "viscosity at that point.  Select a weighted harmonic, arithmetic, "
                            "geometric, or maximum composition.");
          prm.enter_subsection("Viscosities");
          {
            prm.declare_entry ("Minimum viscosity", "1e19",
                               Patterns::Double (0),
                               "The value of the minimum viscosity cutoff $\\eta_min$. Units: $Pa\\;s$.");
            prm.declare_entry ("Maximum viscosity", "1e24",
                               Patterns::Double (0),
                               "The value of the maximum viscosity cutoff $\\eta_max$. Units: $Pa\\;s$.");
            prm.declare_entry ("Reference strain rate", "1e-15",
                               Patterns::Double (0),
                               "The value of the initial strain rate prescribed during the "
                               "first nonlinear iteration $\\dot{\\epsilon}_ref$. Units: $1/s$.");
            prm.declare_entry ("Coefficients of static friction", "0.5",
                               Patterns::List(Patterns::Double(0)),
                               "List of coefficients of static friction for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $dimensionless$");
            prm.declare_entry ("Coefficients of dynamic friction", "0.4",
                               Patterns::List(Patterns::Double(0)),
                               "List of coefficients of dynamic friction for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $dimensionless$");
            prm.declare_entry ("Cohesions", "4.e6",
                               Patterns::List(Patterns::Double(0)),
                               "List of cohesions for background mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $Pa$");
            prm.declare_entry ("Background Viscosities", "1.e20",
                               Patterns::List(Patterns::Double(0)),
                               "List of background viscosities for mantle and compositional fields,"
                               "for a total of N+1 values, where N is the number of compositional fields."
                               "If only one value is given, then all use the same value. Units: $Pa s $");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    DynamicFriction<dim>::parse_parameters (ParameterHandler &prm)
    {
      //not pretty, but we need to get the number of compositional fields before
      //simulatoraccess has been initialized here...

      prm.enter_subsection ("Compositional fields");
      const unsigned int n_fields = this->n_compositional_fields() + 1;
      prm.leave_subsection();

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Dynamic Friction");
        {
          reference_T = prm.get_double ("Reference temperature");

          if (prm.get ("Viscosity averaging scheme") == "harmonic")
            viscosity_averaging = harmonic;
          else if (prm.get ("Viscosity averaging scheme") == "arithmetic")
            viscosity_averaging = arithmetic;
          else if (prm.get ("Viscosity averaging scheme") == "geometric")
            viscosity_averaging = geometric;
          else if (prm.get ("Viscosity averaging scheme") == "maximum composition")
            viscosity_averaging = maximum_composition;
          else
            AssertThrow(false, ExcMessage("Not a valid viscosity averaging scheme"));

          // Parse DynamicFriction properties
          densities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Densities"))),
                                                              n_fields,
                                                              "Densities");
          thermal_conductivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal conductivities"))),
                                                                           n_fields,
                                                                           "Thermal conductivities");
          thermal_expansivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal expansivities"))),
                                                                          n_fields,
                                                                          "Thermal expansivities");
          specific_heats = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Specific heats"))),
                                                                   n_fields,
                                                                   "Specific heats");
          prm.enter_subsection("Viscosities");
          {
            minimum_viscosity  = prm.get_double ("Minimum viscosity");
            maximum_viscosity  = prm.get_double ("Maximum viscosity");
            reference_strain_rate    = prm.get_double ("Reference strain rate");

            mu_s = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Coefficients of static friction"))),
                                                           n_fields,
                                                           "Coefficients of static friction");
            mu_d = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Coefficients of dynamic friction"))),
                                                           n_fields,
                                                           "Coefficients of dynamic friction");
            cohesions = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Cohesions"))),
                                                                n_fields,
                                                                "Cohesions");
            background_viscosities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Background Viscosities"))),
                                                                             n_fields,
                                                                             "Background Viscosities");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Declare dependencies on solution variables
      this->model_dependence.viscosity = NonlinearDependence::compositional_fields | NonlinearDependence::strain_rate;
      this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::compositional_fields;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::compositional_fields;
      this->model_dependence.thermal_conductivity = NonlinearDependence::compositional_fields;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(DynamicFriction,
                                   "dynamic friction",
                                   "This model is for use with an arbitrary number of compositional fields, where each field "
                                   "represents a rock type which can have completely different properties from the others."
                                   "Each rock type itself has constant material properties, with the exception of viscosity "
                                   "which is modified according to a Drucker-Prager yield criterion. Unlike the drucker prager "
                                   "or visco plastic material models, the angle of internal friction is a function of velocity. "
                                   "This relationship is similar to rate-and-state friction constitutive relationships, which "
                                   "are applicable to the strength of rocks during earthquakes. The formulation used here is "
                                   "derived from van Dinther et al. 2013, JGR. Each compositional field is interpreed as a volume fraction. "
                                   "If the sum of the fields is greater than one, they are renormalized. If it is less than one, material properties "
                                   "for ``background material'' make up the rest. When more than one field is present, the "
                                   "material properties are averaged arithmetically. An exception is the viscosity, "
                                   "where the averaging should make more of a difference. For this, the user selects"
                                   "between arithmetic, harmonic, geometric, or maximum composition averaging. ")
  }
}
