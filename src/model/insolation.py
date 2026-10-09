# src/model/insolation.py

'''
This module calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

NOTE: Currently this requires a lot of RAM - look for ways to reduce this and check if it's the main bottleneck in the code.
'''

import time
import numpy as np
import numba
from numba import jit, prange
from joblib import Parallel, delayed
from src.utilities.utils import (
    conditional_tqdm,
    conditional_print,
    rays_triangles_intersection,
    calculate_rotation_matrix, sun_direction
)   
from src.model.scattering import BRDFLookupTable
from src.utilities.tumbling_matrices import tumbling_rotation

from src.utilities.tumbling_final_2 import compute_tumbling_dynamics

from tqdm import tqdm
import astropy.constants as const

L_sun_value = const.L_sun.value


def print_and_log(message, value, filename):
    print(message, flush=True)
    with open(filename, "w") as f:
        f.write(message + "\n")
    return value


def calculate_insolation(thermal_data, shape_model, simulation, config):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

    TODO: Parallelise this function.
    '''
        
    
    '''
    Ne treba u afelu, nego na mestu pocetka orbitalne inicijalizacije. Treba izracunati kako stoji osa tada i uraditi inicijalizaciju kao da rotira samo oko te ose
    '''
    orbital_period = 2*np.pi / np.sqrt(const.GM_sun.value / (simulation.a_au * const.au.value)**3)    
    current_sunlight_direction, current_sun_distance = sun_direction(-orbital_period * simulation.orbital_initialisation, simulation)[0], sun_direction(-orbital_period * simulation.orbital_initialisation, simulation)[1]
    
    print(f'orbital_period: {np.round(orbital_period / 86400 / 365.25)}')
    print(f'current sunlight direction: {current_sunlight_direction}')
    print(f'current sun distance: {current_sun_distance / const.au.value}')


#    current_sunlight_direction = np.array([1, 0, 0])
#    current_sun_distance = simulation.a_au * (1 + simulation.ecc) * const.au.value
    number_of_time_steps = simulation.timesteps_per_day


    # Initialize insolation array with zeros for all facets and timesteps
    insolation_array = np.zeros((len(shape_model), number_of_time_steps))
    
    # Precompute rotation matrices and rotated sunlight directions
    rotation_matrices = np.zeros((number_of_time_steps, 3, 3), dtype=np.float64)
    rotated_sunlight_directions = np.zeros((number_of_time_steps, 3), dtype=np.float64)


    for t in range(number_of_time_steps):
        rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, 
                                                 (2 * np.pi / number_of_time_steps) * t)
        rotation_matrices[t] = rotation_matrix
        rotated_sunlight_directions[t] = np.dot(rotation_matrix.T, current_sunlight_direction)
        rotated_sunlight_directions[t] /= np.linalg.norm(rotated_sunlight_directions[t])
        
        

    # Create chunks for parallel processing
    n_facets = len(shape_model)
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (config.n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
              for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]

    # Extract numpy arrays from shape model and ensure float64 dtype
    normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    shape_model_vertices = np.array([facet.vertices for facet in shape_model], dtype=np.float64)

    if getattr(config, 'numba_insolation', True):
        # Same kernel as for the orbit, with the Sun fixed during this day
        vis_ptr, vis_idx = visible_facets_csr(thermal_data.visible_facets)
        set_numba_threads(config.n_jobs)
        thermal_data.insolation[:] = insolation_orbit_numba(
            normals, positions, vis_ptr, vis_idx,
            rotation_matrices,
            rotated_sunlight_directions,
            simulation.albedo,
            np.full(number_of_time_steps, current_sun_distance, dtype=np.float64),
            np.tile(current_sunlight_direction.astype(np.float64), (number_of_time_steps, 1)),
            config.include_shadowing,
            shape_model_vertices,
            L_sun_value
        )
    else:
        # Process chunks in parallel
        parallel = Parallel(n_jobs=config.n_jobs, verbose=0)
        results = parallel(
            delayed(process_insolation_chunk)(
                normals[start_idx:end_idx],
                positions[start_idx:end_idx],
                thermal_data.visible_facets[start_idx:end_idx],
                rotation_matrices,
                rotated_sunlight_directions,
                simulation.albedo,
                current_sun_distance, # we start from aphelion
                current_sunlight_direction.astype(np.float64),  # Ensure float64
                config.include_shadowing,
                shape_model_vertices
            )
            for start_idx, end_idx in chunks
        )


        for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
            thermal_data.insolation[start_idx:end_idx] = results[chunk_idx]

    if config.n_scatters > 0:
        conditional_print(config.silent_mode, 
                        f"Applying light scattering with {config.n_scatters} iterations...")
        scattering_start = time.time()
        thermal_data = apply_scattering(thermal_data, shape_model, simulation, config,
                                      rotation_matrices, rotated_sunlight_directions)
        scattering_end = time.time()
        conditional_print(config.silent_mode, 
                        f"Time taken to apply light scattering: {scattering_end - scattering_start:.2f} seconds")

    return thermal_data



def calculate_insolation_for_initialization(thermal_data, shape_model, simulation, config):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

    TODO: Parallelise this function.
    '''
        
    
    '''
    Ne treba u afelu, nego na mestu pocetka orbitalne inicijalizacije. Treba izracunati kako stoji osa tada i uraditi inicijalizaciju kao da rotira samo oko te ose
    '''
    orbital_period = 2*np.pi / np.sqrt(const.GM_sun.value / (simulation.a_au * const.au.value)**3)    
    current_sunlight_direction, current_sun_distance = sun_direction(-orbital_period * simulation.orbital_initialisation, simulation)[0], sun_direction(-orbital_period * simulation.orbital_initialisation, simulation)[1]
    
    print(f'orbital_period: {np.round(orbital_period / 86400 / 365.25)}')
    print(f'current sunlight direction: {current_sunlight_direction}')
    print(f'current sun distance: {current_sun_distance / const.au.value}')


#    current_sunlight_direction = np.array([1, 0, 0])
#    current_sun_distance = simulation.a_au * (1 + simulation.ecc) * const.au.value
    number_of_time_steps = simulation.timesteps_per_day


    # Initialize insolation array with zeros for all facets and timesteps
    insolation_array = np.zeros((len(shape_model), number_of_time_steps))
    
    # Precompute rotation matrices and rotated sunlight directions
    rotation_matrices = np.zeros((number_of_time_steps, 3, 3), dtype=np.float64)
    rotated_sunlight_directions = np.zeros((number_of_time_steps, 3), dtype=np.float64)
    
    

    for t in range(number_of_time_steps):
        rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, 
                                                 (2 * np.pi / number_of_time_steps) * t)
        rotation_matrices[t] = rotation_matrix
        rotated_sunlight_directions[t] = np.dot(rotation_matrix.T, current_sunlight_direction)
        rotated_sunlight_directions[t] /= np.linalg.norm(rotated_sunlight_directions[t])
        
        

    # Create chunks for parallel processing
    n_facets = len(shape_model)
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (config.n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
              for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]

    # Extract numpy arrays from shape model and ensure float64 dtype
    normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    shape_model_vertices = np.array([facet.vertices for facet in shape_model], dtype=np.float64)

    # Process chunks in parallel
    parallel = Parallel(n_jobs=config.n_jobs, verbose=0)
    results = parallel(
        delayed(process_insolation_chunk)(
            normals[start_idx:end_idx],
            positions[start_idx:end_idx],
            thermal_data.visible_facets[start_idx:end_idx],
            rotation_matrices,
            rotated_sunlight_directions,
            simulation.albedo,
            current_sun_distance, # we start from aphelion
            current_sunlight_direction.astype(np.float64),  # Ensure float64
            config.include_shadowing,
            shape_model_vertices
        )
        for start_idx, end_idx in chunks
    )
       

    for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
        thermal_data.insolation[start_idx:end_idx] = results[chunk_idx]

    if config.n_scatters > 0:
        conditional_print(config.silent_mode, 
                        f"Applying light scattering with {config.n_scatters} iterations...")
        scattering_start = time.time()
        thermal_data = apply_scattering(thermal_data, shape_model, simulation, config,
                                      rotation_matrices, rotated_sunlight_directions)
        scattering_end = time.time()
        conditional_print(config.silent_mode, 
                        f"Time taken to apply light scattering: {scattering_end - scattering_start:.2f} seconds")

    return thermal_data



def calculate_insolation_whole_orbit(thermal_data, shape_model, simulation, config):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

    TODO: Parallelise this function.
    '''
    # Initialize insolation array with zeros for all facets and timesteps
    # insolation_array = np.zeros((len(shape_model), simulation.timesteps_per_orbit))
    
    # Precompute rotation matrices and rotated sunlight directions

    
    rotation_matrices = np.zeros((simulation.timesteps_per_orbit, 3, 3), dtype=np.float64)
    rotated_sunlight_directions = np.zeros((simulation.timesteps_per_orbit, 3), dtype=np.float64)
    current_sunlight_directions = np.zeros((simulation.timesteps_per_orbit, 3), dtype=np.float64)
    rotated_transfersal_directions = np.zeros((simulation.timesteps_per_orbit, 3), dtype=np.float64)

    current_sun_distance = np.zeros(simulation.timesteps_per_orbit)
    true_anomaly = np.zeros(simulation.timesteps_per_orbit)
    
    
    
    
    for t in range(simulation.timesteps_per_orbit):
        total_time = t * simulation.delta_t
           
        current_sunlight_directions[t], current_sun_distance[t], true_anomaly[t] = sun_direction(total_time, simulation)
        
        current_transfersal_direction = np.cross(current_sunlight_directions[t], np.array([0, 0, 1]))
        
        
        rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, 
                                                 (2 * np.pi / simulation.timesteps_per_day) * t)
 
        rotation_matrices[t] = rotation_matrix
        rotated_sunlight_directions[t] = np.dot(rotation_matrix.T, current_sunlight_directions[t])
        rotated_sunlight_directions[t] /= np.linalg.norm(rotated_sunlight_directions[t])
        
        rotated_transfersal_directions[t] = np.dot(rotation_matrix.T, current_transfersal_direction)
    # Create chunks for parallel processing
    n_facets = len(shape_model)
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (config.n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
              for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]

    # Extract numpy arrays from shape model and ensure float64 dtype
    normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    shape_model_vertices = np.array([facet.vertices for facet in shape_model], dtype=np.float64)

    # Process chunks in parallel
    parallel = Parallel(n_jobs=config.n_jobs, verbose=0)

    
    visible_facets_arrays = [
    np.array(facets, dtype=np.int64) for facets in thermal_data.visible_facets
    ]   
       
    n_chunks = len(chunks)

    results = parallel(
        delayed(process_insolation_chunk_orbit)(
            print(f"Processing chunk {chunk_idx+1} of {n_chunks} (indices {start_idx}:{end_idx}", flush=True) or normals[start_idx:end_idx].astype(np.float64),
            positions[start_idx:end_idx].astype(np.float64),
            np.array(visible_facets_arrays[start_idx:end_idx], dtype=object),
            rotation_matrices.astype(np.float64),
            rotated_sunlight_directions.astype(np.float64),
            simulation.albedo,
            current_sun_distance.astype(np.float64),
            current_sunlight_directions.astype(np.float64),
            config.include_shadowing,
            shape_model_vertices.astype(np.float64)
        )
        for chunk_idx, (start_idx, end_idx) in enumerate(chunks)
    )
       

    insol_array = np.empty((len(normals), simulation.timesteps_per_orbit), dtype=np.float64)

    # Popunjavamo array rezultatima po chunk-ovima
    for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
        insol_array[start_idx:end_idx] = results[chunk_idx]
        
        
    return insol_array, true_anomaly, current_sun_distance/const.au.value, -rotated_sunlight_directions, -rotated_transfersal_directions




def calculate_insolation_orbit_section(thermal_data, shape_model, simulation, config, timesteps_per_orbit_section, orbit_section, initial_rotation_state, initialisation):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

    TODO: Parallelise this function.
    '''
    # Initialize insolation array with zeros for all facets and timesteps
    # insolation_array = np.zeros((len(shape_model), simulation.timesteps_per_orbit))
    
    # Precompute rotation matrices and rotated sunlight directions
    if initialisation == 1:
        total_time = np.sum(timesteps_per_orbit_section[:orbit_section]) * simulation.delta_t - simulation.total_initialisation_time

    else:
        total_time = np.sum(timesteps_per_orbit_section[:orbit_section]) * simulation.delta_t
    
#    print('total time', total_time/86400)
#    print('time steps', timesteps_per_orbit_section[orbit_section])
    
    rotation_matrices = np.zeros((timesteps_per_orbit_section[orbit_section], 3, 3), dtype=np.float64)
    rotated_sunlight_directions = np.zeros((timesteps_per_orbit_section[orbit_section], 3), dtype=np.float64)
    current_sunlight_directions = np.zeros((timesteps_per_orbit_section[orbit_section], 3), dtype=np.float64)
    rotated_transfersal_directions = np.zeros((timesteps_per_orbit_section[orbit_section], 3), dtype=np.float64)

    current_sun_distance = np.zeros(timesteps_per_orbit_section[orbit_section])
    true_anomaly = np.zeros(timesteps_per_orbit_section[orbit_section])
    
    # print('///////////////////////////////////////////////////////////////////')
    
   
    
    # print(f'POCETAK: inicijalizacija = {initialisation}, vreme = {total_time}')
    
    for t in range(timesteps_per_orbit_section[orbit_section]):
        total_time += simulation.delta_t
           
        current_sunlight_directions[t], current_sun_distance[t], true_anomaly[t] = sun_direction(total_time, simulation) # this is OK
        
        current_transfersal_direction = np.cross(current_sunlight_directions[t], np.array([0, 0, 1])) # this is OK
        
        rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, 
                                                 (2 * np.pi / simulation.timesteps_per_day) * (np.sum(timesteps_per_orbit_section[:orbit_section]) + t)) # THIS
 
        rotation_matrices[t] = rotation_matrix
        rotated_sunlight_directions[t] = np.dot(rotation_matrix.T, current_sunlight_directions[t]) # THIS
        rotated_sunlight_directions[t] /= np.linalg.norm(rotated_sunlight_directions[t])
        
        rotated_transfersal_directions[t] = np.dot(rotation_matrix.T, current_transfersal_direction) # THIS
        
    # print(f'KRAJ: inicijalizacija = {initialisation}, vreme = {total_time}')
    last_state = 0
    
    
    
    total_tumbling_time = simulation.total_initialisation_time - np.sum(timesteps_per_orbit_section[:orbit_section]) * simulation.delta_t
    if simulation.version == 'new':
        

        
        
        
        
        
        rotation = compute_tumbling_dynamics(dt = simulation.delta_t, timesteps = timesteps_per_orbit_section[orbit_section], 
                                             y0 = initial_rotation_state,
                                             I = simulation.I, 
                                             r_inertial = current_sunlight_directions,
                                             lambda_L_deg = simulation.lambda_L,
                                             beta_L_deg = simulation.beta_L, 
                                             initialisation = initialisation
                                             )
        
        
        
        
        
        
        total_tumbling_time -= timesteps_per_orbit_section[orbit_section] * simulation.delta_t
        
        # print('---------------------------- TUMBLING TIME ---------------------------')
        # print(total_tumbling_time)
        
        last_state = rotation['last_state']
        rotated_sunlight_directions = rotation['r_body']
        rotated_transfersal_directions = rotation['t_body']
        rotation_matrices = rotation['rotations']
        
   
     
    # Create chunks for parallel processing
    n_facets = len(shape_model)
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (config.n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
              for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]

    # Extract numpy arrays from shape model and ensure float64 dtype
    normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    shape_model_vertices = np.array([facet.vertices for facet in shape_model], dtype=np.float64)

    message = (f"Processing initialization section {orbit_section + 1} out of {len(timesteps_per_orbit_section)}"
               if initialisation == 1 else
               f"Processing orbit section {orbit_section + 1} out of {len(timesteps_per_orbit_section)}")

    if getattr(config, 'numba_insolation', True):
        # Compiled, multithreaded equivalent of process_insolation_chunk_orbit (no joblib processes)
        print_and_log(message, None, simulation.progress_file)
        vis_ptr, vis_idx = visible_facets_csr(thermal_data.visible_facets)
        set_numba_threads(config.n_jobs)
        insol_array = insolation_orbit_numba(
            normals, positions, vis_ptr, vis_idx,
            np.ascontiguousarray(rotation_matrices, dtype=np.float64),
            np.ascontiguousarray(rotated_sunlight_directions, dtype=np.float64),
            simulation.albedo,
            np.ascontiguousarray(current_sun_distance, dtype=np.float64),
            np.ascontiguousarray(current_sunlight_directions, dtype=np.float64),
            config.include_shadowing,
            shape_model_vertices,
            L_sun_value
        )
    else:
        # Process chunks in parallel
        parallel = Parallel(n_jobs=config.n_jobs, verbose=0)

    
        visible_facets_arrays = [
        np.array(facets, dtype=np.int64) for facets in thermal_data.visible_facets
        ]   
       
        n_chunks = len(chunks)

    # =============================================================================
    #     results = parallel(
    #     delayed(process_insolation_chunk_orbit)(
    #         (
    #             print(
    #                 f"Processing initialization section {orbit_section + 1} out of {len(timesteps_per_orbit_section)}"
    #                 if initialisation == 1 else
    #                 f"Processing orbit section {orbit_section + 1} out of {len(timesteps_per_orbit_section)}",
    #                 flush=True
    #             ) if start_idx == 0 else None
    #         ) or normals[start_idx:end_idx].astype(np.float64),
    #         positions[start_idx:end_idx].astype(np.float64),
    #         np.array(visible_facets_arrays[start_idx:end_idx], dtype=object),
    #         rotation_matrices.astype(np.float64),
    #         rotated_sunlight_directions.astype(np.float64),
    #         simulation.albedo,
    #         current_sun_distance.astype(np.float64),
    #         current_sunlight_directions.astype(np.float64),
    #         config.include_shadowing,
    #         shape_model_vertices.astype(np.float64)
    #     )
    #     for chunk_idx, (start_idx, end_idx) in enumerate(chunks)
    #     )
    # =============================================================================

        results = parallel(
            delayed(process_insolation_chunk_orbit)(
                (
                    print_and_log(
                        message,
                        normals[start_idx:end_idx].astype(np.float64),
                        simulation.progress_file
                    )
                    if start_idx == 0 else
                    normals[start_idx:end_idx].astype(np.float64)
                ),
                positions[start_idx:end_idx].astype(np.float64),
                np.array(visible_facets_arrays[start_idx:end_idx], dtype=object),
                rotation_matrices.astype(np.float64),
                rotated_sunlight_directions.astype(np.float64),
                simulation.albedo,
                current_sun_distance.astype(np.float64),
                current_sunlight_directions.astype(np.float64),
                config.include_shadowing,
                shape_model_vertices.astype(np.float64)
            )
            for chunk_idx, (start_idx, end_idx) in enumerate(chunks)
        )
       

        insol_array = np.empty((len(normals), timesteps_per_orbit_section[orbit_section]), dtype=np.float64)

        # Popunjavamo array rezultatima po chunk-ovima
        for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
            insol_array[start_idx:end_idx] = results[chunk_idx]
        
        
    return (
        insol_array,
        true_anomaly,
        current_sun_distance / const.au.value,
        -rotated_sunlight_directions,
        -rotated_transfersal_directions,
        last_state
    )







@jit(nopython=True)
def process_insolation_chunk(normals, positions, visible_facets, rotation_matrices, 
                           rotated_sunlight_directions, albedo,
                           solar_distance_m, sunlight_direction, include_shadowing,
                           shape_model_vertices):
    """
    Process insolation calculations for a chunk of facets using only numba-compatible types.
    """
    chunk_size = len(normals)
    timesteps = len(rotation_matrices)
    insolation = np.zeros((chunk_size, timesteps))
    
    for i in range(chunk_size):
        normal = normals[i]
        position = positions[i]
        
        for t in range(timesteps):
            new_normal = np.dot(rotation_matrices[t], normal)
            new_normal_norm = np.linalg.norm(new_normal)  # Precompute new normal vector norm
            
            
            
            sun_dot_normal = np.dot(sunlight_direction, new_normal)
            
            # Calculate cosine of zenith angle
            cos_zenith_angle = sun_dot_normal / (np.linalg.norm(sunlight_direction) * new_normal_norm)
            
            if cos_zenith_angle > 0:
                illumination_factor = 1
                
                if len(visible_facets[i]) != 0 and include_shadowing:
                    illumination_factor = calculate_shadowing(
                        position, 
                        rotated_sunlight_directions[t:t+1],  # Use slice instead of creating new array
                        shape_model_vertices,
                        visible_facets[i]
                    )
                
                insolation[i, t] = (
                    L_sun_value * 
                    (1 - albedo) * 
                    illumination_factor * 
                    cos_zenith_angle / 
                    (4 * np.pi * solar_distance_m**2)
                )
    
    return insolation










#@jit(nopython=True)
def process_insolation_chunk_orbit(normals, positions, visible_facets, rotation_matrices, 
                           rotated_sunlight_directions, albedo, solar_distance_m,
                           sunlight_direction, include_shadowing,
                           shape_model_vertices):
    """
    Process insolation calculations for a chunk of facets using only numba-compatible types.
    """
    chunk_size = len(normals)
    timesteps = len(rotation_matrices)
    insolation = np.zeros((chunk_size, timesteps))
    
    for i in range(chunk_size):
        normal = normals[i]
        position = positions[i]
        

        for t in range(timesteps):
            
#            current_sunlight_direction = 
#                               
            new_normal = np.dot(rotation_matrices[t], normal)
            new_normal_norm = np.linalg.norm(new_normal)  # Precompute new normal vector norm
            
            sun_dot_normal = np.dot(sunlight_direction[t], new_normal)
            

            # Calculate cosine of zenith angle
            cos_zenith_angle = sun_dot_normal / (np.linalg.norm(sunlight_direction[t]) * new_normal_norm)
            
            if cos_zenith_angle > 0:
                illumination_factor = 1
                
                if len(visible_facets[i]) != 0 and include_shadowing:
                    illumination_factor = calculate_shadowing(
                        position, 
                        rotated_sunlight_directions[t:t+1],  # Use slice instead of creating new array
                        shape_model_vertices,
                        visible_facets[i]
                    )
                
                insolation[i, t] = (
                    L_sun_value * 
                    (1 - albedo) * 
                    illumination_factor * 
                    cos_zenith_angle / 
                    (4 * np.pi * solar_distance_m[t]**2)
                )
    
    return insolation




def visible_facets_csr(visible_facets):
    '''Pack the per-facet lists of visible facets into two flat arrays (CSR format) for numba.'''
    lengths = np.array([len(v) for v in visible_facets], dtype=np.int64)
    vis_ptr = np.zeros(len(visible_facets) + 1, dtype=np.int64)
    vis_ptr[1:] = np.cumsum(lengths)
    vis_idx = np.zeros(vis_ptr[-1], dtype=np.int64)
    for i, v in enumerate(visible_facets):
        vis_idx[vis_ptr[i]:vis_ptr[i + 1]] = v
    return vis_ptr, vis_idx


def set_numba_threads(n_jobs):
    '''Use n_jobs from the config as the number of numba threads (-1 = all available).'''
    max_threads = numba.config.NUMBA_NUM_THREADS
    n = max_threads if n_jobs is None or n_jobs < 1 else min(int(n_jobs), max_threads)
    numba.set_num_threads(n)


@jit(nopython=True, cache=True)
def ray_hits_any_triangle(origin, ray, shape_model_vertices, vis_idx, start, end):
    '''
    Same test as calculate_shadowing / rays_triangles_intersection (Moller-Trumbore, eps = 1e-6)
    for a single ray, stopping at the first triangle that is hit.
    '''
    eps = 0.000001
    for k in range(start, end):
        j = vis_idx[k]
        v1x = shape_model_vertices[j, 0, 0]; v1y = shape_model_vertices[j, 0, 1]; v1z = shape_model_vertices[j, 0, 2]
        e1x = shape_model_vertices[j, 1, 0] - v1x; e1y = shape_model_vertices[j, 1, 1] - v1y; e1z = shape_model_vertices[j, 1, 2] - v1z
        e2x = shape_model_vertices[j, 2, 0] - v1x; e2y = shape_model_vertices[j, 2, 1] - v1y; e2z = shape_model_vertices[j, 2, 2] - v1z

        # pvec = cross(ray, edge2)
        px = ray[1] * e2z - ray[2] * e2y
        py = ray[2] * e2x - ray[0] * e2z
        pz = ray[0] * e2y - ray[1] * e2x
        det = e1x * px + e1y * py + e1z * pz
        if abs(det) < eps:
            continue
        inv_det = 1.0 / det

        tx = origin[0] - v1x; ty = origin[1] - v1y; tz = origin[2] - v1z
        u = (tx * px + ty * py + tz * pz) * inv_det
        if u < 0.0 or u > 1.0:
            continue

        # qvec = cross(tvec, edge1)
        qx = ty * e1z - tz * e1y
        qy = tz * e1x - tx * e1z
        qz = tx * e1y - ty * e1x
        v = (ray[0] * qx + ray[1] * qy + ray[2] * qz) * inv_det
        if v < 0.0 or u + v > 1.0:
            continue

        t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
        if t < eps:
            continue
        return True
    return False


@jit(nopython=True, parallel=True, cache=True)
def insolation_orbit_numba(normals, positions, vis_ptr, vis_idx, rotation_matrices,
                           rotated_sunlight_directions, albedo, solar_distance_m,
                           sunlight_direction, include_shadowing, shape_model_vertices, L_sun):
    '''Compiled equivalent of process_insolation_chunk_orbit for all facets at once (parallel over facets).'''
    n_facets = normals.shape[0]
    timesteps = rotation_matrices.shape[0]
    insolation = np.zeros((n_facets, timesteps))

    for i in prange(n_facets):
        nx = normals[i, 0]; ny = normals[i, 1]; nz = normals[i, 2]
        start = vis_ptr[i]
        end = vis_ptr[i + 1]

        for t in range(timesteps):
            R = rotation_matrices[t]
            mx = R[0, 0] * nx + R[0, 1] * ny + R[0, 2] * nz
            my = R[1, 0] * nx + R[1, 1] * ny + R[1, 2] * nz
            mz = R[2, 0] * nx + R[2, 1] * ny + R[2, 2] * nz
            new_normal_norm = np.sqrt(mx * mx + my * my + mz * mz)

            sx = sunlight_direction[t, 0]; sy = sunlight_direction[t, 1]; sz = sunlight_direction[t, 2]
            sun_dot_normal = sx * mx + sy * my + sz * mz
            cos_zenith_angle = sun_dot_normal / (np.sqrt(sx * sx + sy * sy + sz * sz) * new_normal_norm)

            if cos_zenith_angle > 0:
                illumination_factor = 1

                if end > start and include_shadowing:
                    if ray_hits_any_triangle(positions[i], rotated_sunlight_directions[t],
                                             shape_model_vertices, vis_idx, start, end):
                        illumination_factor = 0

                insolation[i, t] = (
                    L_sun *
                    (1 - albedo) *
                    illumination_factor *
                    cos_zenith_angle /
                    (4 * np.pi * solar_distance_m[t]**2)
                )

    return insolation


@jit(nopython=True)
def calculate_shadowing(subject_positions, sunlight_directions, shape_model_vertices, visible_facet_indices):
    '''
    This function calculates whether a facet is in shadow at a given time step. It cycles through all visible facets and passes their vertices to rays_triangles_intersections which determines whether they fall on the sunlight direction vector (starting at the facet position). If they do, the facet is in shadow. 
    
    It returns the illumination factor for the facet at that time step. 0 if the facet is in shadow, 1 if it is not.

    TODO: Use Monte Carlo or even distribution ray tracing for faster shadowing calculations.
    '''

    # Ensure triangles_vertices is an array of shape (m, 3, 3)
    triangles_vertices = shape_model_vertices[visible_facet_indices]

    # Call the intersection function
    intersections, t_values = rays_triangles_intersection(
        subject_positions,
        sunlight_directions,
        triangles_vertices
    )

    # Check for any intersection
    if intersections.any():
        return 0  # The facet is in shadow
        
    return 1 # The facet is not in shadow

def calculate_brdf_values(shape_model, rotation_matrices, rotated_sunlight_directions, brdf_lut, start_idx, end_idx, visible_facets_list):
    n_timesteps = len(rotation_matrices)
    brdf_values = {}
    n_facets_in_chunk = end_idx - start_idx
    
    # Collect all unique facet indices needed for this chunk
    needed_facets = set(range(start_idx, end_idx))  # Start with facets in chunk
    for i in range(start_idx, end_idx):
        needed_facets.update(visible_facets_list[i])  # Add their visible facets
    needed_facets = sorted(needed_facets)  # Convert to sorted list
    
    # Create mapping from global to local indices
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(needed_facets)}
    
    # Pre-rotate only the needed facets
    n_needed_facets = len(needed_facets)
    rotated_normals = np.zeros((n_needed_facets, n_timesteps, 3))
    rotated_positions = np.zeros((n_needed_facets, n_timesteps, 3))
    
    # Pre-compute rotated values for needed facets
    for local_idx, global_idx in enumerate(needed_facets):
        normal_i = shape_model[global_idx].normal
        position_i = shape_model[global_idx].position
        for t in range(n_timesteps):
            rotated_normals[local_idx, t] = np.dot(rotation_matrices[t], normal_i)
            rotated_positions[local_idx, t] = np.dot(rotation_matrices[t], position_i)
    
    # Process facets in this chunk
    for i in range(start_idx, end_idx):
        local_i = global_to_local[i]
        visible_facets = visible_facets_list[i]
        brdf_values[i] = np.zeros((len(visible_facets), n_timesteps))
        
        sun_cos = np.einsum('tj,tj->t', rotated_sunlight_directions, rotated_normals[local_i])
        illuminated_timesteps = sun_cos > 0
        
        if not illuminated_timesteps.any():
            continue
            
        inc_deg = np.degrees(np.arccos(sun_cos[illuminated_timesteps]))
        
        for j, target_idx in enumerate(visible_facets):
            local_target = global_to_local[target_idx]
            
            direction_vectors = (rotated_positions[local_target, illuminated_timesteps] - 
                               rotated_positions[local_i, illuminated_timesteps])
            
            # Rest of the BRDF calculation remains the same...

def process_scattering_chunk(start_idx, end_idx, input_light, visible_facets_list, 
                           view_factors_list, timesteps_per_day, albedo, iteration,
                           brdf_lut=None, shape_model=None, rotation_matrices=None, 
                           rotated_sunlight_directions=None):
    """
    Process a chunk of facets for scattering calculation.
    """
    chunk_scattered_light = np.zeros_like(input_light)
    
    # Only calculate BRDF on first iteration
    if iteration == 0 and brdf_lut is not None:
        brdf_values = calculate_brdf_values(
            shape_model, rotation_matrices, rotated_sunlight_directions,
            brdf_lut, start_idx, end_idx, visible_facets_list
        )
    else:
        brdf_values = None  # Use default Lambertian scattering for subsequent iterations
    
    for i in range(start_idx, end_idx):
        visible_facets = visible_facets_list[i]
        view_factors = view_factors_list[i]

        for t in range(timesteps_per_day):
            current_light = input_light[i, t]
            
            if current_light > 0:
                for j, (vf_idx, vf) in enumerate(zip(visible_facets, view_factors)):
                    brdf = brdf_values[i][j, t] if brdf_values is not None else 1.0
                    chunk_scattered_light[vf_idx, t] += (
                        brdf * current_light * vf * albedo / np.pi
                    )
                    
    return chunk_scattered_light

def apply_scattering(thermal_data, shape_model, simulation, config, 
                    rotation_matrices, rotated_sunlight_directions):
    """
    Apply scattering using BRDF lookup tables. Works with any number of jobs (including 1).
    """
    # Initialize BRDF lookup table
    brdf_lut = BRDFLookupTable(config.scattering_lut)
    
    original_insolation = thermal_data.insolation.copy()
    total_scattered_light = np.zeros_like(original_insolation)
    n_facets = len(shape_model)
    
    # Convert lists to numpy arrays
    visible_facets_list = [np.array(x) for x in thermal_data.visible_facets]
    view_factors_list = [np.array(x) for x in thermal_data.secondary_radiation_view_factors]
    
    # Get number of jobs and create chunks
    actual_n_jobs = config.validate_jobs()
    
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]
    
    for iteration in range(config.n_scatters):
        if iteration == 0:
            input_light = original_insolation
        else:
            input_light = scattered_light
        
        # Create parallel executor
        parallel = Parallel(n_jobs=actual_n_jobs, verbose=0)
        delayed_funcs = [
            delayed(process_scattering_chunk)(
                start_idx, end_idx,
                input_light,
                visible_facets_list,
                view_factors_list,
                simulation.timesteps_per_day,
                simulation.albedo, 
                iteration,
                brdf_lut if iteration == 0 else None,
                shape_model if iteration == 0 else None,
                rotation_matrices if iteration == 0 else None,
                rotated_sunlight_directions if iteration == 0 else None
            )
            for start_idx, end_idx in chunks
        ]
        
        # Process chunks with real-time progress bar
        scattered_light = np.zeros_like(original_insolation)
        with tqdm(total=len(chunks), desc=f"Iteration {iteration + 1}/{config.n_scatters}") as pbar:
            for chunk_result in parallel(delayed_funcs):
                scattered_light += chunk_result
                pbar.update(1)
                
        total_scattered_light += scattered_light
    
    thermal_data.insolation = original_insolation + total_scattered_light
    return thermal_data
