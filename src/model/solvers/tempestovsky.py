import matplotlib.pyplot as plt

import numpy as np
from numba import jit
from .base_solver import TemperatureSolver
from src.utilities.utils import conditional_print, compute_tumbling_dynamics
from src.model.insolation import calculate_insolation, calculate_insolation_whole_orbit, calculate_insolation_orbit_section
from src.model.simulation import ThermalData_propagation
import astropy.constants as const
import time
import os

import sys

#from src.utilities.tumbling_final_2 import compute_tumbling_dynamics

# Standalone numba functions
@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    if len(visible_facets) == 0 or len(view_factors) == 0:
        return 0.0
    return self_heating_const * np.sum(temperatures[visible_facets]**4 * view_factors)

@jit(nopython=True)
def calculate_temperatures(temperatures, layer_temperatures, insolation, visible_facets_list, 
                        view_factors_list, const1, const2, const3, self_heating_const,
                        timesteps_per_day, n_layers, include_self_heating):

    n_facets = temperatures.shape[0]
    current_column = 0  # Use column 0 for current timestep
    prev_column = 1    # Use column 1 for previous timestep

    for time_step in range(timesteps_per_day):
        # Swap columns for next iteration
        current_column, prev_column = prev_column, current_column
        
        for i in range(n_facets):
            # Surface temperature calculation
            prev_temp = layer_temperatures[i, prev_column, 0]
            prev_temp_layer1 = layer_temperatures[i, prev_column, 1]
            
            insolation_term = insolation[i, time_step] * const1
            re_emitted_radiation_term = -const2 * (prev_temp**4)
            
            secondary_radiation_term = 0.0
            if include_self_heating:
                secondary_radiation_term = calculate_secondary_radiation(
                    layer_temperatures[:, prev_column, 0], 
                    visible_facets_list[i], 
                    view_factors_list[i], 
                    self_heating_const
                )
            
            conducted_heat_term = const3 * (prev_temp_layer1 - prev_temp)
            
            new_temp = (prev_temp + 
                    insolation_term + 
                    re_emitted_radiation_term + 
                    conducted_heat_term + 
                    secondary_radiation_term)

            temperatures[i, time_step] = new_temp
            layer_temperatures[i, current_column, 0] = new_temp
            
            # Update subsurface temperatures
            for layer in range(1, n_layers - 1):
                prev_layer = layer_temperatures[i, prev_column, layer]
                prev_layer_plus = layer_temperatures[i, prev_column, layer + 1]
                prev_layer_minus = layer_temperatures[i, prev_column, layer - 1]

                layer_temperatures[i, current_column, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 
                            2 * prev_layer + 
                            prev_layer_minus)
                )
                    
    return temperatures



def update_thermal_state(thermal_data, current_insolation, simulation):
    '''
    thermal_data.layer_temperatures: (n_facets, n_layers) - trenutne temperature
    current_insolation: (n_facets) - fluks za ovaj specifični trenutak na orbiti
    '''
    T = thermal_data.layer_temperatures
    T_new = np.copy(T)

    coeff = simulation.thermal_diffusivity * simulation.delta_t / (simulation.layer_thickness**2)
    
    if coeff > 0.5:
        print("Upozorenje: Model je numerički nestabilan.")

    T_new[:, 1:-1] = T[:, 1:-1] + coeff * (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2])
    T_new[:, -1] = T_new[:, -2]
    
    # OVDE JE BILA GREŠKA: 
    # Dodajemo [:, 0] i [:, 1] da bi rezultat bio (1266,) a ne (1266, 45)
    conduction_to_subsurface = simulation.thermal_conductivity * (T[:, 0] - T[:, 1]) / simulation.layer_thickness

    # Sada će ova linija raditi jer su svi nizovi (1266,)
    dT_surf = (simulation.delta_t / (simulation.density * simulation.specific_heat_capacity * simulation.layer_thickness)) * (current_insolation - simulation.emissivity * const.sigma_sb.value * T[:, 0]**4 - conduction_to_subsurface)
    
    T_new[:, 0] = T[:, 0] + dT_surf
    
    return T_new


def update_thermal_state_thickness(thermal_data, current_insolation, simulation):
    '''
    thermal_data.layer_temperatures: (n_facets, n_layers) - trenutne temperature
    current_insolation: (n_facets) - fluks za ovaj specifični trenutak na orbiti
    '''
    T = thermal_data.layer_temperatures
    T_new = np.copy(T)

    dz = simulation.layer_thickness

    coeff = simulation.thermal_diffusivity * simulation.delta_t / (np.min(dz)**2)
    
    if coeff > 0.5:
        print("Upozorenje: Model je numerički nestabilan.")

    # rastojanja između centara slojeva
    dz_center_up = (dz[1:-1] + dz[2:]) / 2
    dz_center_down = (dz[:-2] + dz[1:-1]) / 2

    # temperaturni gradijenti
    grad_up = (T[:, 2:] - T[:, 1:-1]) / dz_center_up
    grad_down = (T[:, 1:-1] - T[:, :-2]) / dz_center_down

    # divergencija fluksa
    laplacian = (grad_up - grad_down) / dz[1:-1]

    T_new[:, 1:-1] = T[:, 1:-1] + simulation.thermal_diffusivity * simulation.delta_t * laplacian

    # donja granica (adiabatska)
    T_new[:, -1] = T_new[:, -2]

    # provodjenje sa površine u podzemlje (centar-centar)
    dz_surface = (dz[0] + dz[1]) / 2
    conduction_to_subsurface = simulation.thermal_conductivity * (T[:, 0] - T[:, 1]) / dz_surface

    # energetski bilans površinskog sloja
    dT_surf = (simulation.delta_t / (simulation.density * simulation.specific_heat_capacity * dz[0])) * (
        current_insolation
        - simulation.emissivity * const.sigma_sb.value * T[:, 0]**4
        - conduction_to_subsurface
    )

    T_new[:, 0] = T[:, 0] + dT_surf

    thermal_data.layer_temperatures = T_new




    
'''
ADDED 

'''

# def geometry_for_yarko(shape_model):
#     positions = np.array([facet.position for facet in shape_model])
#     normals   = np.array([facet.normal   for facet in shape_model])
#     areas     = np.array([facet.area     for facet in shape_model])

#     volume = (1.0 / 3.0) * np.sum(
#         areas * np.einsum('ij,ij->i', positions, normals)
#     )

#     return abs(volume), normals, areas



import numpy as np

def geometry_for_yarko(shape_model):
    # Izvlačenje podataka preko atributa koji garantovano postoje
    positions = np.array([facet.position for facet in shape_model])
    raw_normals = np.array([facet.normal for facet in shape_model])
    areas = np.array([facet.area for facet in shape_model])

    # 1. Normalizacija normala (zaštita od loše zapisanih STL fajlova)
    norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    # Sprečavanje deljenja sa nulom za degenerisane trouglove
    norms[norms == 0] = 1.0 
    normals = raw_normals / norms

    # 2. Računanje zapremine preko Teoreme o divergenciji
    # V = (1/3) * sum( Area * (Position . Normal) )
    dot_products = np.einsum('ij,ij->i', positions, normals)
    volume = (1.0 / 3.0) * np.sum(areas * dot_products)

    return abs(volume), normals, areas


def calculate_yarkovsky(simulation, normals, areas, asteroid_mass, r_rad, r_trans, true_anomaly, sun_distance, temperatures):
    
    F=2/3*simulation.emissivity*const.sigma_sb.value / const.c.value * np.sum(temperatures[:, None]**4 * normals * areas[:, None], axis=0)
     
    B = np.dot(F, r_trans)
        
    # Radial thermal force
    R = np.dot(F, r_rad)

    
    dadt = 2 * simulation.mean_motion * (simulation.a_au * const.au.value)**2 / const.GM_sun.value * (
                   R * simulation.a_au * const.au.value * simulation.ecc * np.sin(true_anomaly) / np.sqrt(1-simulation.ecc**2) + 
                   B * simulation.a_au**2 * const.au.value * np.sqrt(1-simulation.ecc**2) / sun_distance)/asteroid_mass
            
    # dadt = 2 * simulation.a_au / simulation.mean_motion / sun_distance * np.sqrt(1 - simulation.ecc**2) * B / asteroid_mass
            
  
    return dadt


class YarkovskySolver(TemperatureSolver):
    def __init__(self):
        super().__init__("tempest_standard_yarko")
        self.required_parameters = [
            "emissivity",
            "density",
            "specific_heat_capacity",
            "thermal_conductivity",
            "n_layers",
            "convergence_target",
            "beaming_factor"
        ]

    def diurnal_day(self, thermal_data, simulation, config, const1, const2, const3, self_heating_const):
        """Advance the temperatures over one rotation (diurnal initialisation). Overridden by the implicit solver."""
        return calculate_temperatures(
            thermal_data.temperatures,
            thermal_data.layer_temperatures,
            thermal_data.insolation,
            thermal_data.visible_facets,
            thermal_data.thermal_view_factors,
            const1, const2, const3, self_heating_const,
            simulation.timesteps_per_day, simulation.n_layers,
            config.include_self_heating
        )

    def orbit_step(self, thermal_data, current_insolation, simulation):
        """Advance the temperatures by one timestep along the orbit. Overridden by the implicit solver."""
        return update_thermal_state(thermal_data, current_insolation, simulation)

    def solve(self, thermal_data, shape_model, simulation, config):
        ''' 
        This is the main calculation function for the thermophysical body model. It calls the necessary functions to read in the shape model, set material and model properties, calculate 
        insolation and temperature arrays, and iterate until the model converges.
        '''

        # Geometry for Yarkovsky
        
        volume, normals, areas = geometry_for_yarko(shape_model)
        asteroid_mass = volume * simulation.density
        

#        # indeks najblizi ekvatoru
#        idx_equator = np.argmin(np.abs(normals[:,2]))
#
#        # indeks najblizi polu
#        idx_pole = np.argmax(np.abs(normals[:,2]))

        poluprecnik = (3 * volume /4 / np.pi)**(1/3)

        # Initialize constants
        const1 = simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
        const2 = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
        const3 = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2
        self_heating_const = 5.670374419e-8 * simulation.delta_t * simulation.emissivity**2 / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)

        convergence_error = simulation.convergence_target + 1
        day = 0
        error_history = []
        comparison_temps = thermal_data.temperatures[:, 0].copy()
        
        # Initialization


        # unikuta dnevna inicijacija
        while day <= simulation.max_days and (day < simulation.min_days or convergence_error > simulation.convergence_target):
            current_day_temperature = self.diurnal_day(
                thermal_data, simulation, config,
                const1, const2, const3, self_heating_const
            )

            # Check for invalid temperatures
            for i in range(len(shape_model)):
                for time_step in range(simulation.timesteps_per_day):
                    if np.isnan(current_day_temperature[i, time_step]) or np.isinf(current_day_temperature[i, time_step]) or current_day_temperature[i, time_step] < 0:
                        conditional_print(config.silent_mode, f"Invalid temperature {current_day_temperature[i, time_step]} K detected for facet {i} at timestep {time_step}")
                        return {
                            "final_day_temperatures": None,
                            "final_day_temperatures_all_layers": None,
                            "final_timestep_temperatures": None,
                            "days_to_convergence": day,
                            "mean_temperature_error": None,
                            "max_temperature_error": None
                        }

            if config.calculate_energy_terms:
                energy_terms = self.calculate_energy_terms(
                    current_day_temperature, 
                    thermal_data.insolation,
                    simulation.delta_t,
                    simulation.emissivity,
                    simulation.beaming_factor,
                    simulation.density,
                    simulation.specific_heat_capacity,
                    simulation.layer_thickness,
                    simulation.thermal_conductivity,
                    simulation.timesteps_per_day,
                    simulation.n_layers
                )
                
                thermal_data.insolation_energy = energy_terms[:, :, 0]
                thermal_data.re_emitted_energy = energy_terms[:, :, 1]
                thermal_data.surface_energy_change = energy_terms[:, :, 2]
                thermal_data.conducted_energy = energy_terms[:, :, 3]
                thermal_data.unphysical_energy_loss = energy_terms[:, :, 4]

            # Calculate convergence
            temperature_errors = np.abs(current_day_temperature[:, 0] - comparison_temps)
            
            if config.convergence_method == 'mean':
                convergence_error = np.mean(temperature_errors)
            else:
                convergence_error = np.max(temperature_errors)

            max_temperature_error = np.max(temperature_errors)
            mean_temperature_error = np.mean(temperature_errors)

            conditional_print(config.silent_mode, f"Day: {day} | Mean Temperature error: {mean_temperature_error:.6f} K | Max Temp Error: {max_temperature_error:.6f} K")
            
            comparison_temps = current_day_temperature[:, 0].copy()
            error_history.append(convergence_error)
            day += 1
            
        # Check for invalid temperatures
        for i in range(len(shape_model)):
            for time_step in range(simulation.timesteps_per_day):
                if np.isnan(current_day_temperature[i, time_step]) or np.isinf(current_day_temperature[i, time_step]) or current_day_temperature[i, time_step] < 0:
                    conditional_print(config.silent_mode, f"Invalid temperature {current_day_temperature[i, time_step]} K detected for facet {i} at timestep {time_step}")
                    return {
                        "final_day_temperatures": None,
                        "final_day_temperatures_all_layers": None,
                        "final_timestep_temperatures": None,
                        "days_to_convergence": day,
                        "mean_temperature_error": None,
                        "max_temperature_error": None
                    }

        if config.calculate_energy_terms:
            energy_terms = self.calculate_energy_terms(
                current_day_temperature, 
                thermal_data.insolation,
                simulation.delta_t,
                simulation.emissivity,
                simulation.beaming_factor,
                simulation.density,
                simulation.specific_heat_capacity,
                simulation.layer_thickness,
                simulation.thermal_conductivity,
                simulation.timesteps_per_day,
                simulation.n_layers
            )
            
            thermal_data.insolation_energy = energy_terms[:, :, 0]
            thermal_data.re_emitted_energy = energy_terms[:, :, 1]
            thermal_data.surface_energy_change = energy_terms[:, :, 2]
            thermal_data.conducted_energy = energy_terms[:, :, 3]
            thermal_data.unphysical_energy_loss = energy_terms[:, :, 4]
            
        print('**********************************************')
        print('**********************************************')
        print('**********************************************')
        print('Diurnal Initialization finished') 
        print('delta t', simulation.delta_t)
        print('**********************************************')
        print('**********************************************')
        print('**********************************************')
        
        
        
        '''
        this must be input parameter
        '''
        number_of_orbit_sections = 100
        
        
        
# 

#        
        
        counter = 0
        thermal_data.layer_temperatures = thermal_data.layer_temperatures[:, 1, :]
        last_state = 0
        
        if simulation.orbital_initialisation > 0:
            
            number_of_rotations = np.floor(simulation.orbital_period * simulation.orbital_initialisation/simulation.rotation_period_s) # OK
            
            total_number_of_initial_timesteps = (number_of_rotations * simulation.timesteps_per_day).astype(int)
            
            simulation.orbital_initialisation = number_of_rotations * simulation.rotation_period_s / simulation.orbital_period # OK
            number_of_initial_sections = int(np.ceil(number_of_orbit_sections * simulation.orbital_initialisation)) # OK
            timesteps_per_orbit_section = (np.ones(number_of_initial_sections) * np.floor(simulation.timesteps_per_orbit/number_of_initial_sections * simulation.orbital_initialisation)).astype(int)
            # timesteps_per_orbit_section[-1] = np.ceil(simulation.timesteps_per_orbit * simulation.orbital_initialisation) - sum(timesteps_per_orbit_section[:-1]) - 1
            

            timesteps_per_orbit_section[-1] = total_number_of_initial_timesteps - sum(timesteps_per_orbit_section[:-1])
            
            
            simulation.total_initialisation_time = (total_number_of_initial_timesteps) * simulation.delta_t

#            surface_history = np.zeros([len(areas), np.sum(timesteps_per_orbit_section)])
#            mean_T = np.zeros(np.sum(timesteps_per_orbit_section))
#            mean_insolation = np.zeros(np.sum(timesteps_per_orbit_section))
#            true_anomaly_orbit = np.zeros(np.sum(timesteps_per_orbit_section))
#            r_sun = np.zeros(np.sum(timesteps_per_orbit_section))


            '''
            ovde treba da se izvrti u nazad da bi se dobilo inicijalno stanje
            
            
            
            
            '''
            
            # print('broj koraka')
            # print(timesteps_per_orbit_section)


            
         
            for orbit_section in range(number_of_initial_sections):
                
                if orbit_section == 0:
                    
                    reverse_rotation = compute_tumbling_dynamics(dt = -simulation.delta_t, timesteps = total_number_of_initial_timesteps, 
                                                         y0 = simulation.y0,
                                                         I = simulation.I, 
                                                         r_inertial = np.tile(np.array([1, 0, 0]), (total_number_of_initial_timesteps, 1)), # doesn't matter
                                                         lambda_L_deg = simulation.lambda_L,
                                                         beta_L_deg = simulation.beta_L,
                                                         initialisation = 1
                                                         )
                    initial_rotation_state = reverse_rotation['last_state']
                else:
                    initial_rotation_state = last_state
                    
                
                
                precomputed_insolation, true_anomaly, current_sun_distance, r_rad, r_trans, last_state = calculate_insolation_orbit_section(thermal_data, shape_model, simulation, config, timesteps_per_orbit_section, orbit_section, initial_rotation_state = initial_rotation_state, initialisation = 1)
    
    
                # print(f'initial insolation = {np.mean(precomputed_insolation)}')
    
                # print(f'last state = {initial_rotation_state}')
                
                for t in range(timesteps_per_orbit_section[orbit_section]):
        
    
                    thermal_data.layer_temperatures = self.orbit_step(thermal_data, precomputed_insolation[:, t], simulation)
        
                    surface_temperatures = thermal_data.layer_temperatures[:, 0]

        
        print('**********************************************')
        print('**********************************************')
        print('**********************************************')
        print('       Seasonal Initialization finished       ')
        print('**********************************************')
        print('**********************************************')
        print('**********************************************')
        
        
        surface_history = np.zeros([len(areas), simulation.timesteps_per_orbit])
        
        mean_T = np.zeros(simulation.timesteps_per_orbit)
        mean_insolation = np.zeros(simulation.timesteps_per_orbit)
        drift = np.zeros(simulation.timesteps_per_orbit)
        
        true_anomaly_orbit = np.zeros(simulation.timesteps_per_orbit)
        r_sun = np.zeros(simulation.timesteps_per_orbit)

        
#        number_of_orbit_sections = 3
        timesteps_per_orbit_section = (np.ones(number_of_orbit_sections) * simulation.timesteps_per_orbit/number_of_orbit_sections).astype(int)
        timesteps_per_orbit_section[-1] = simulation.timesteps_per_orbit - sum(timesteps_per_orbit_section[:-1])
        
        # if simulation.version == 'new':
        #     timesteps_per_orbit_section[-1] = simulation.timesteps_per_orbit - sum(timesteps_per_orbit_section[:-1]) + 100
        
        
        counter = 0
        os.makedirs("output", exist_ok=True)

        last_state = simulation.y0

#        thermal_data.layer_temperatures = thermal_data.layer_temperatures[:, 1, :]
        for orbit_section in range(number_of_orbit_sections):
            
            (
                precomputed_insolation,
                true_anomaly,
                current_sun_distance,
                r_rad,
                r_trans,
                last_state
            ) = calculate_insolation_orbit_section(
                thermal_data,
                shape_model,
                simulation,
                config,
                timesteps_per_orbit_section,
                orbit_section,
                initial_rotation_state = last_state,
                initialisation=0
            )
        
            # # Dodavanje rezultata na kraj fajlova
            # with open(f"output/precomputed_insolation_{simulation.version}.npy", "ab") as f:
            #     np.save(f, precomputed_insolation)
        
            # with open(f"output/true_anomaly_{simulation.version}.npy", "ab") as f:
            #     np.save(f, true_anomaly)
        
            # with open(f"output/current_sun_distance_{simulation.version}.npy", "ab") as f:
            #     np.save(f, current_sun_distance)
        
            # with open(f"output/r_rad_{simulation.version}.npy", "ab") as f:
            #     np.save(f, r_rad)
        
            # with open(f"output/r_trans_{simulation.version}.npy", "ab") as f:
            #     np.save(f, r_trans)
        
            # with open(f"output/last_state_{simulation.version}.npy", "ab") as f:
            #     np.save(f, last_state)
                
                
            # print(timesteps_per_orbit_section[0])
                
            # if orbit_section == 5:

            #     raise SystemExit
        
            for t in range(timesteps_per_orbit_section[orbit_section]):
    
                
                
                # KLJUČNI MOMENAT: 
                # Pozivamo funkciju i REZULTAT upisujemo nazad u thermal_data.
                # Tako u sledećoj iteraciji (t+1) funkcija uzima temperaturu od (t).
                thermal_data.layer_temperatures = self.orbit_step(thermal_data, precomputed_insolation[:, t], simulation)
    
                surface_temperatures = thermal_data.layer_temperatures[:, 0]
                # Ovde možeš sačuvati površinsku temperaturu za ovaj trenutak ako ti treba za grafikon
                surface_history[:, counter] = surface_temperatures
                
                drift[counter] = calculate_yarkovsky(simulation, normals, areas, asteroid_mass, 
                                               r_rad[t], r_trans[t], true_anomaly[t], 
                                               current_sun_distance[t], 
                                               surface_temperatures)
                
                mean_T[counter] = np.mean(surface_history[:, counter])
                mean_insolation[counter] = np.mean(precomputed_insolation[:, t])
                
                true_anomaly_orbit[counter] = true_anomaly[t]
                r_sun[counter] = current_sun_distance[t]
                
                # if np.mod(counter, 1000) == 0:
                #     print(f'step {counter} out of {simulation.timesteps_per_orbit}, section: {orbit_section}')
    
                
        
                # angles_rad = np.rad2deg(np.arctan2(r_rad[:,1], r_rad[:,0]))
                # angles_trans = np.rad2deg(np.arctan2(r_trans[:,1], r_trans[:,0]))
                
                
                # vertikal_trans = np.rad2deg(np.arcsin(r_trans[:,2]))
                
                # vertikal_rad = np.rad2deg(np.arcsin(r_rad[:,2]))
                
                counter += 1
                

            # with open(f"output/precomputed_insolation_{simulation.version}.npy", "ab") as f:
            #     np.save(f, precomputed_insolation)
        
            # with open(f"output/true_anomaly_{simulation.version}.npy", "ab") as f:
            #     np.save(f, true_anomaly)
        
            # with open(f"output/current_sun_distance_{simulation.version}.npy", "ab") as f:
            #     np.save(f, current_sun_distance)
                
            # if orbit_section == 5:
            
            #     raise SystemExit

        # print('poluprecnik', poluprecnik)
        # print('zapremina', volume)
        # print('povrsina', np.sum(areas))
        # print('drift', np.mean(drift))
        
        with open(simulation.output_file, 'w') as ff:
            ff.write(f"drift (m/s): {np.mean(drift)}\n")
        ff.close()

        # plt.figure()
        # plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, mean_insolation)
        # plt.title('INSOLATION')
        # plt.grid()
        # plt.show()
        
        # plt.figure()
        # plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, mean_T)
        # plt.title('MEAN t')
        # plt.grid()
        # plt.show()
        
        # plt.figure()
        # plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, true_anomaly_orbit)
        # plt.title('true anomaly')
        # plt.grid()
        # plt.show()
        
        # plt.figure()
        # plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, r_sun)
        # plt.title('r Sun')
        # plt.grid()
        # plt.show()
        
#        plt.figure()
#        plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, angles_rad)
#        plt.title('RAD')
#        plt.show()
#        
#        plt.figure()
#        plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, angles_trans)
#        plt.title('TRANS')
#        plt.show()
#        
#        
#        plt.figure()
#        plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, vertikal_rad)
#        plt.title('vertikal RAD')
#        plt.show()
#        
#        plt.figure()
#        plt.plot(np.arange(simulation.timesteps_per_orbit)*simulation.delta_t/3600, vertikal_trans)
#        plt.title('vertikal TRANS')
#        plt.show()

        
        # plt.figure()
        # plt.plot(np.arange(simulation.timesteps_per_day)*simulation.delta_t/3600, np.rad2deg())
        # plt.title('DRIFT')
        # plt.show()
        
        
        
        
        
        return {
            "final_day_temperatures": current_day_temperature,
            "final_day_temperatures_all_layers": thermal_data.layer_temperatures,
            "final_timestep_temperatures": current_day_temperature[:, -1],
            "days_to_convergence": day,
            "mean_temperature_error": mean_temperature_error,
            "max_temperature_error": max_temperature_error,
            "Yarkovsky drift": np.mean(drift)
        } 
            
