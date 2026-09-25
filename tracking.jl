using BeamDistributions, BeamTracking
import BeamTracking: chargeof, massof, C_LIGHT
import Statistics: mean
import Random: default_rng
include("ags_apertures.jl")


# =================================================================
# Set Machine Parameters

mm_ad_misalignments = 1e-2*randn(length(mm_ad))
mm_af_misalignments = 1e-2*randn(length(mm_af))
mm_bd_misalignments = 1e-2*randn(length(mm_bd))
mm_bf_misalignments = 1e-2*randn(length(mm_bf))
mm_cd_misalignments = 1e-2*randn(length(mm_cd))
mm_cf_misalignments = 1e-2*randn(length(mm_cf))

setproperty!.(values(mm_ad), :x_offset, mm_ad_misalignments)
setproperty!.(values(mm_af), :x_offset, mm_af_misalignments)
setproperty!.(values(mm_bd), :x_offset, mm_bd_misalignments)
setproperty!.(values(mm_bf), :x_offset, mm_bf_misalignments)
setproperty!.(values(mm_cd), :x_offset, mm_cd_misalignments)
setproperty!.(values(mm_cf), :x_offset, mm_cf_misalignments)


#=
qh_misalignments = 3.4e-4*randn(length(qh))
qv_misalignments = 3.4e-4*randn(length(qv))
qp_misalignments = 3.4e-4*randn(length(qp))

# Can loop over `qh`, `qv`, `qp` but this is easier to read
setproperty!.(values(qh), :x_offset, qh_misalignments)
setproperty!.(values(qv), :x_offset, qv_misalignments)
setproperty!.(values(qp), :x_offset, qp_misalignments)
=#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






# =================================================================
# Set Beam Parameters

ref = ReferenceParticle(
    charge_e=chargeof(ags.species_ref), 
    rest_energy_eV=massof(ags.species_ref), 
    momentum_eVc=ags.pc_ref
)

frame = PhaseSpaceFrame(id="example", reference_particle=ref)

beam = normal_coupling_gaussian(frame=frame,
    centroid=(-0.0146, -0.0011, 0.0, 0.0, 0.0, 0.0),
    eigenemittances=(2.0e-6/3, 2.0e-6/3), sigma_z=2.5,
    sigma_delta=1.0e-4, dispersion=(-2.14, -0.15),
    beta1=20.17, alpha1=1.57, beta2=11.77, alpha2=1.02,
    coupling=(q1q2=0.00, q1p2=0.00, p1q2=0.00, p1p2=0.00),
    bunch_population = 2.5e10
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





# =================================================================
# Set Controls

#HTUNE_CTRL_it = 100*(2*rand()-1)
#VTUNE_CTRL_it = 100*(2*rand()-1)

for ele_name in keys(I_dhc) 
    I_dhc[ele_name] = 10*(2*rand()-1)
end

for ele_name in keys(I_dvc) 
    I_dvc[ele_name] = 10*(2*rand()-1)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# =================================================================
# Measure and Get Observables


# Electrical RMS noise: σ_S, σ_Dx, σ_Dy
_dS = _dX = _dY = 1e-3

# Calibration factor: signal-to-bunch_charge_C ratio
_pue_cal = 2.5e9
bunch_C = bunch_charge_C(beam)


function measure_observables!(
    beam,
    ring,
    bpm;
    dS      = _dS,
    dX      = _dX,
    dY      = _dY,
    pue_cal = _pue_cal,
    rng     = default_rng(),
    verbose = false,
)

    particles = sample(beam, 2^10;
        sampler=SobolRmsTorusSampler(exact_moments=true)
    )

    bunch = Bunch(
        transpose(coordinates(particles)), 
        species = Species("proton"), 
        p_over_q_ref = ref.momentum_eVc/ref.charge_e/C_LIGHT
    )

    N = length(bunch.state)

    observables = Dict(
        obj.name => Dict(
            :S  => randn(rng) * dS,
            :Dx => randn(rng) * dX,
            :Dy => randn(rng) * dY,
        )
        for obj in values(bpm)
    )

    S = bunch_charge_C(beam)

    for ele in ring.line
        iszero(S) && break

        track!(bunch, ele)

        # This element is not a pickup, so there is nothing to measure.
        haskey(observables, ele.name) || continue

        alive_inds = findall(==(0x01), bunch.state)
        n_alive = length(alive_inds)
        alive_fraction = n_alive / N

        S = bunch_C * alive_fraction
        signal = S * pue_cal

        obs = observables[ele.name]
        obs[:S] += signal

        if n_alive > 0
            x_mean  = mean(@view bunch.coords.v[alive_inds, 1])
            px_mean = mean(@view bunch.coords.v[alive_inds, 2])
            y_mean  = mean(@view bunch.coords.v[alive_inds, 3])

            obs[:Dx] += x_mean * signal
            obs[:Dy] += y_mean * signal

            if verbose
                println("Fraction at $(ele.name): $alive_fraction")
                println(
                    "(x, px) = (",
                    round(x_mean; sigdigits=3),
                    ", ",
                    round(px_mean; sigdigits=3),
                    ")",
                )
            end
        elseif verbose
            println("Fraction at $(ele.name): 0.0")
            println("No surviving particles")
        end
    end

    return observables
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
