"""
Thin Julia facade for the AGS injection environment.

All PHYSICS lives in tracking.jl (machine/apertures via ags_apertures.jl,
beam, and the tracking kernel `measure_observables!`) and is kept identical to
the physics branch — this file must NOT contain machine or measurement logic,
so physics updates land in the RL environment without edits here.

What remains here is the minimal set of primitives the Python AGSEnv
(ags_env.py) marshals through: redrawing main-magnet misalignments, assigning
corrector currents by name, setting the tune controls, and one single-pass
ring measurement. Reward, state assembly, action layout, bounds, and all other
RL configuration are Python-side.
"""
module AGS_GymEnv

using PythonCall  # pyconvert() for the Python<->Julia boundary

include("tracking.jl")  # THE single source of physics (includes ags_apertures.jl)

# =====================================================================
# Primitives exposed to the Python RL wrapper (no RL logic below)
# =====================================================================

"""
    redraw_misalignments!(sigma)

Redraw main-magnet horizontal misalignments, mirroring the setup block in
tracking.jl (`setproperty!.(values(mm_*), :x_offset, sigma * randn)`), but with
caller-controlled sigma. Called by the Python env at construction, so each
process draws a fresh frozen lattice (unseeded, by design).
"""
function redraw_misalignments!(sigma::Real)
    for grp in (mm_ad, mm_af, mm_bd, mm_bf, mm_cd, mm_cf)
        setproperty!.(values(grp), :x_offset, sigma * randn(length(grp)))
    end
    return nothing
end

function _assign!(d::AbstractDict, names::Vector{String}, vals::Vector{Float64})
    length(names) == length(vals) || error("name/value length mismatch")
    for (n, v) in zip(names, vals)
        haskey(d, n) || error("unknown control: $n")
        d[n] = v
    end
    return nothing
end

"""
    set_currents!(dhc_names, dhc_vals, dvc_names, dvc_vals)

Assign corrector currents by name (the caller/Python owns the action layout;
this is pure marshalling). Accepts Python sequences; converts via pyconvert.
"""
function set_currents!(dhc_names, dhc_vals, dvc_names, dvc_vals)
    _assign!(I_dhc, pyconvert(Vector{String}, dhc_names),
             pyconvert(Vector{Float64}, dhc_vals))
    _assign!(I_dvc, pyconvert(Vector{String}, dvc_names),
             pyconvert(Vector{Float64}, dvc_vals))
    return nothing
end

"""Zero every corrector current (the RL episode's "no action" state)."""
function zero_currents!()
    for name in keys(I_dhc)
        I_dhc[name] = 0.0
    end
    for name in keys(I_dvc)
        I_dvc[name] = 0.0
    end
    return nothing
end

"""Set the horizontal/vertical tune-control globals consumed by ags_apertures.jl."""
function set_tunes!(h::Real, v::Real)
    global var"HTUNE_CTRL_it" = Float64(h)
    global var"VTUNE_CTRL_it" = Float64(v)
    return nothing
end

"""Sorted control-supply names (e.g. "a02") — the canonical action/state order."""
dhc_names() = sort(collect(keys(I_dhc)))
dvc_names() = sort(collect(keys(I_dvc)))

"""Sorted BPM element names — the canonical observation order."""
bpm_names() = sort([obj.name for obj in values(pue)])

"""
    measure(; dS=0.0, dX=0.0, dY=0.0)

Run the ring once with the current machine/beam/controls and return the raw
observables dict (BPM name => Dict(:S,:Dx,:Dy)) from tracking.jl's kernel.
Electrical noise defaults to zero for deterministic RL; pass dS/dX/dY to enable.
"""
measure(; dS = 0.0, dX = 0.0, dY = 0.0) =
    measure_observables!(beam, ags, pue; dS = dS, dX = dX, dY = dY)

export redraw_misalignments!, set_currents!, zero_currents!, set_tunes!,
       dhc_names, dvc_names, bpm_names, measure

end # module
