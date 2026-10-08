module AGS_GymEnv

using BeamDistributions, BeamTracking
using Beamlines
import BeamTracking: chargeof, massof, C_LIGHT
import Statistics: mean
import Random: default_rng
using PythonCall

include("ags_apertures.jl")

# =================================================================
# Configuration Constants

# Fixed tune controls (configurable but fixed for now)
const HTUNE_CTRL_fixed = 500
const VTUNE_CTRL_fixed = 500

# Action bounds for dipole corrector currents
const I_dhc_bounds = (-25.0, 25.0)
const I_dvc_bounds = (-25.0, 25.0)

# Default misalignment sigma
const DEFAULT_MISALIGN_SIGMA = 3.4e-4

# Fixed beam parameters (hidden state)
const BEAM_CENTROID = (-0.0146, -0.0011, 0.0, 0.0, 0.0, 0.0)
const BEAM_EIGENEMITTANCES = (2.0e-6/3, 2.0e-6/3)
const BEAM_SIGMA_Z = 2.5
const BEAM_SIGMA_DELTA = 1.0e-4
const BEAM_DISPERSION = (-2.14, -0.15)
const BEAM_BETA1 = 20.17
const BEAM_ALPHA1 = 1.57
const BEAM_BETA2 = 11.77
const BEAM_ALPHA2 = 1.02
const BEAM_COUPLING = (q1q2=0.00, q1p2=0.00, p1q2=0.00, p1p2=0.00)
const BEAM_BUNCH_POPULATION = 2.5e10

# Electrical RMS noise constants (set to zero for deterministic RL training)
const _dS = 0.0
const _dX = 0.0
const _dY = 0.0
const _pue_cal = 2.5e9

# =================================================================
# Environment State (global, initialized on module load)

# These are set when the environment is initialized
var"HTUNE_CTRL_it" = HTUNE_CTRL_fixed
var"VTUNE_CTRL_it" = VTUNE_CTRL_fixed

# Global beam reference (recreated on each step for consistency)
global _ref = ReferenceParticle(
    charge_e=chargeof(ags.species_ref),
    rest_energy_eV=massof(ags.species_ref),
    momentum_eVc=ags.pc_ref
)

"""
    get_state_size()

Get the size of the state vector.
State = 73 BPMs × (S, Dx, Dy) + 48 I_dhc + 48 I_dvc = 219 + 96 = 315
"""
function get_state_size()
    return 73 * 3 + 48 + 48
end

"""
    get_action_size()

Get the size of the action vector.
Action = 48 I_dhc + 48 I_dvc = 96
"""
function get_action_size()
    return 48 + 48
end

"""
    get_action_bounds()

Get the bounds for the action space as arrays.
"""
function get_action_bounds()
    n = get_action_size()
    low = fill(I_dhc_bounds[1], n)
    high = fill(I_dhc_bounds[2], n)
    return low, high
end

"""
    get_control_names()

Get the names of all dipole corrector controls in a consistent order.
Returns an array of names like ["a02_dhc", "a02_dvc", ...]
"""
function get_control_names()
    names = String[]
    for sec in 'a':'l'
        for pos in ["02", "08", "12", "18"]
            name = string(sec, pos)
            push!(names, name * "_dhc")
            push!(names, name * "_dvc")
        end
    end
    return names
end

"""
    get_bpm_names()

Get the names of all BPM elements in sorted order.
"""
function get_bpm_names()
    names = [obj.name for obj in values(pue)]
    return sort(names)
end

"""
    set_action(action::Vector{Float64})

Set the dipole corrector currents from a flat action vector.
Action order follows get_control_names(): [a02_dhc, a02_dvc, b02_dhc, b02_dvc, ...]
"""
function set_action(action::Vector{Float64})
    n_dhc = 48

    # Get sorted control names
    dhc_names = sort(collect(keys(I_dhc)))
    dvc_names = sort(collect(keys(I_dvc)))

    # Set I_dhc values
    for (i, name) in enumerate(dhc_names)
        I_dhc[name] = action[i]
    end

    # Set I_dvc values
    for (i, name) in enumerate(dvc_names)
        I_dvc[name] = action[n_dhc + i]
    end

    return nothing
end

"""
    measure_observables()

Run the simulation and measure observables at all BPMs.
Returns a dictionary of BPM name => (S, Dx, Dy) values.
"""
function measure_observables()
    # Recreate beam with fixed parameters
    frame = PhaseSpaceFrame(id="gym_env", reference_particle=_ref)

    beam = normal_coupling_gaussian(frame=frame,
        centroid=BEAM_CENTROID,
        eigenemittances=BEAM_EIGENEMITTANCES,
        sigma_z=BEAM_SIGMA_Z,
        sigma_delta=BEAM_SIGMA_DELTA,
        dispersion=BEAM_DISPERSION,
        beta1=BEAM_BETA1, alpha1=BEAM_ALPHA1,
        beta2=BEAM_BETA2, alpha2=BEAM_ALPHA2,
        coupling=BEAM_COUPLING,
        bunch_population=BEAM_BUNCH_POPULATION
    )

    bunch_C = bunch_charge_C(beam)

    # Sample particles
    particles = sample(beam, 2^10;
        sampler=SobolRmsTorusSampler(exact_moments=true)
    )

    bunch = Bunch(
        transpose(coordinates(particles)),
        species = Species("proton"),
        p_over_q_ref = _ref.momentum_eVc / _ref.charge_e / C_LIGHT
    )

    N = length(bunch.state)

    # Initialize observables dictionary
    observables = Dict(
        obj.name => Dict(
            :S  => randn(default_rng()) * _dS,
            :Dx => randn(default_rng()) * _dX,
            :Dy => randn(default_rng()) * _dY,
        )
        for obj in values(pue)
    )

    S = bunch_C

    for ele in ags.line
        iszero(S) && break

        track!(bunch, ele)

        # This element is not a pickup, so there is nothing to measure.
        haskey(observables, ele.name) || continue

        alive_inds = findall(==(0x01), bunch.state)
        n_alive = length(alive_inds)
        alive_fraction = n_alive / N

        S = bunch_C * alive_fraction
        signal = S * _pue_cal

        obs = observables[ele.name]
        obs[:S] += signal

        if n_alive > 0
            x_mean  = mean(@view bunch.coords.v[alive_inds, 1])
            px_mean = mean(@view bunch.coords.v[alive_inds, 2])
            y_mean  = mean(@view bunch.coords.v[alive_inds, 3])

            obs[:Dx] += x_mean * signal
            obs[:Dy] += y_mean * signal
        end
    end

    return observables
end

"""
    get_state()

Get the current state as a flat array.
State = [BPM observables (S, Dx, Dy for 73 BPMs), control currents (96)]
Total: 315 values
"""
function get_state()
    obs = measure_observables()

    state = Float64[]

    # Add BPM observables in sorted order
    bpm_names = sort(collect(keys(obs)))
    for name in bpm_names
        push!(state, obs[name][:S])
        push!(state, obs[name][:Dx])
        push!(state, obs[name][:Dy])
    end

    # Add current control values in sorted order
    for name in sort(collect(keys(I_dhc)))
        push!(state, I_dhc[name])
    end
    for name in sort(collect(keys(I_dvc)))
        push!(state, I_dvc[name])
    end

    return state
end

"""
    get_reward()

Compute the reward based on beam centering at BPMs.
Since beam survival is ~constant (all particles survive), the reward
focuses on minimizing the RMS beam offset across all BPMs.
- Dx, Dy (beam position) penalize off-center beam — closer to zero is better
Reward = -lambda * sqrt(mean(Dx^2 + Dy^2))
The agent's goal is to steer the beam as close to center as possible at all BPMs.
"""
function get_reward()
    obs = measure_observables()
    lambda = 1000.0
    total_offset = sum(obs[name][:Dx]^2 + obs[name][:Dy]^2 for name in keys(obs))
    n_bpm = length(keys(obs))
    rms_offset = sqrt(total_offset / n_bpm)
    return -lambda * rms_offset
end

"""
    step(action::Vector{Float64})

Take a single step in the environment.
1. Set the action (control currents)
2. Measure observables (done once, reused for state & reward)
3. Compute state and reward
Returns: (state::Vector{Float64}, reward::Float64, done::Bool)
"""
function step(action::Vector{Float64})
    # Set the controls
    set_action(action)

    # Measure observables once and reuse for both state and reward
    obs = measure_observables()

    # Compute reward (RMS beam offset penalty)
    total_offset = sum(obs[name][:Dx]^2 + obs[name][:Dy]^2 for name in keys(obs))
    n_bpm = length(keys(obs))
    rms_offset = sqrt(total_offset / n_bpm)
    reward = -1000.0 * rms_offset

    # Build state vector
    state = Float64[]

    # Add BPM observables in sorted order
    bpm_names = sort(collect(keys(obs)))
    for name in bpm_names
        push!(state, obs[name][:S])
        push!(state, obs[name][:Dx])
        push!(state, obs[name][:Dy])
    end

    # Add current control values in sorted order
    for name in sort(collect(keys(I_dhc)))
        push!(state, I_dhc[name])
    end
    for name in sort(collect(keys(I_dvc)))
        push!(state, I_dvc[name])
    end

    # Single-step optimization, so always done
    done = true

    return state, reward, done
end

"""
    reset()

Reset the environment to initial state.
Resets all control currents to zero.
Returns: (state::Vector{Float64}, reward::Float64)
"""
function reset()
    # Reset control currents
    for name in keys(I_dhc)
        I_dhc[name] = 0.0
    end
    for name in keys(I_dvc)
        I_dvc[name] = 0.0
    end

    # Get initial state
    state = get_state()
    reward = get_reward()

    return state, reward
end

"""
    initialize!(; h_tune=0.0, v_tune=0.0, misalign_sigma=0.00034)

Initialize the environment with specific parameters.
This sets up misalignments and tune controls.
"""
function initialize!(; h_tune::Float64 = HTUNE_CTRL_fixed,
                       v_tune::Float64 = VTUNE_CTRL_fixed,
                       misalign_sigma::Float64 = DEFAULT_MISALIGN_SIGMA)
    # Reset tune controls
    var"HTUNE_CTRL_it" = h_tune
    var"VTUNE_CTRL_it" = v_tune

    # Apply misalignments
    rng = default_rng()
    qh_misalignments = misalign_sigma * randn(length(qh))
    qv_misalignments = misalign_sigma * randn(length(qv))
    qp_misalignments = misalign_sigma * randn(length(qp))

    setproperty!.(values(qh), :x_offset, qh_misalignments)
    setproperty!.(values(qv), :x_offset, qv_misalignments)
    setproperty!.(values(qp), :x_offset, qp_misalignments)

    return nothing
end

# Export functions for PythonCall to access
export step, reset, get_state, get_reward, set_action,
       get_state_size, get_action_size, get_action_bounds,
       get_control_names, get_bpm_names, initialize!

end # module AGS_GymEnv
