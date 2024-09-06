# simulating excited 1d wave systems with distributed resonances
# in frequency domain using the invariant imbedding technique
# The modularized version

module WS

using Printf: Floats
using Base: datatype_fielddesc_type
using Printf
using LinearAlgebra

import Base.@kwdef

export WaveSystem,
    LoadedSystem,
    Resonance,
    SourceCarrier,
    bands_disp_point,
    get_field,
    show_info

const weak_tolerance = 1.0e-5
const test_tolerance = 1.0e-3
const band_tolerance = 0.1 # 1.0e-4
#const imeps = 0.0075im # decay at resonances
const imeps = 0.00001im # decay at resonances

const FVector = Vector{Float64}
const CVector = Vector{ComplexF64}
const CMatrix = Matrix{ComplexF64}

# Boundary matrices
# TODO: make them the part of the simulation
# Outgoing boundary conditions
const G_L_outgoing ::CMatrix = [[1, 0] [0, 0]]
const G_R_outgoing ::CMatrix = [[0, 0] [0, 1]]

const BoundaryConditions = Dict{Symbol, CMatrix}

set_boundary_conditions(mat_A::CMatrix, mat_B::CMatrix) = Dict(:A => mat_A, :B => mat_B)

const outgoing_Gs ::BoundaryConditions = set_boundary_conditions(G_L_outgoing, G_R_outgoing)

# Complex unit matrix
Ic ::Matrix{ComplexF64} = [[1, 0] [0, 1]]


@kwdef struct Resonance
    position  ::Float64         # positions of resonances
    frequency ::Float64         # resonance frequencies
    kappa     ::Float64         # coupling with resonances
end

############################################################
##
##  Transfer matrix methods
##
############################################################

function res_response(res ::Resonance, freq ::Float64) ::ComplexF64
    return 1/(freq^2 - res.frequency^2 + imeps * freq)
end


function alpha_point2(res ::Resonance, freq ::Float64) ::ComplexF64
    # evaluate the alpha factor for point resonances
    return res.kappa * freq * res_response(res, freq)/2.0
end

function alpha_point(res ::Resonance, freq ::Float64) ::ComplexF64
    # evaluate the alpha factor for point resonances
    return res.kappa * (freq + imeps) * res_response(res, freq)/2.0
end

"""
    point_transfer(res ::Resonance, freq ::Float64)::Array{ComplexF64}

The transfer matrix through a resonance in the basis
of plain wave solutions (in the Cauchy basis T_C = 1 + γ σ_-)
This is an example of a declaration of such functions
"""
function point_transfer(res ::Resonance, freq ::Float64)::CMatrix
    alpha::ComplexF64 = alpha_point(res, freq)
    phase = 2 * freq * res.position
    t2::CMatrix = [-1.0 -imexp(-phase); imexp(phase) 1.0]

    return Ic + 1im .* alpha .* t2
end

# TEMP: I am confused about the performance of the built-in exp function
function imexp(phase::Float64)::ComplexF64
    return cos(phase) + 1im*sin(phase)
end

# find the inverse of 2x2 matrix `mat` using the explicit expression
function l_inverse(mat::CMatrix) ::CMatrix
    det = mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]
    return Matrix{ComplexF64}([mat[2, 2] -mat[1, 2]; -mat[2, 1] mat[1, 1]])./det
end

########################################################
##
## Spectral methods
##
########################################################

# (β, ω)
const SpectralPoint = Tuple{ComplexF64, Float64}

"""
    beta_disp_point(freqs ::FVector,
                    d ::Float64, res ::Resonance) ::CVector

Evaluate β's corresponding to the list of frequencies `freqs`.
The system is characterized by period `d` and a representative resonance
`res`. For the given frequency, we have

cos(β d) = cos(k d) + α(ω) sin(k d)

When the r.h.s. is outside of [-1, 1], the corresponding β's are imaginary.
"""
function beta_disp_point(freqs ::FVector,
                         d ::Float64,
                         res ::Resonance) ::CVector
    phases = freqs .* d
    af = x -> alpha_point(res, x)
    disp_rhs::CVector = cos.(phases) + af.(freqs) .* sin.(phases)
    betas::CVector = acos.(disp_rhs) ./ d
    # we want to make imaginary beta in the upper plane
    map(x -> imag(x) < 0 ? identity(x) : conj(x), betas)
    return betas
end

"""
    bands_disp_point(freqs ::Vector{Float64}, d ::Float64, res ::Resonance)

For the provided array of frequencies `freqs` evaluate the band diagram β(ω) of
a system with period `d` with the resonance `res`.

Return tuple (bands, gaps) with
    bands :: Vector{Vector{SpectralPoint}}
    gaps :: Vector{Vector{SpectralPoint}}

The structure of arrays `bands` and `gaps` is as follows

bands = [band_1, band_2, ...]
band_1 = [ [β_1, ω_1], [β_2, ω_2], ...]]

with real β_k (Im(β_k) < band_tolerance)

The `gaps` contain pairs with imaginary β_k (Im(β_k) > band_tolerance).
"""
function bands_disp_point(freqs ::FVector, d ::Float64, res ::Resonance)
    bands = Vector{Vector{SpectralPoint}}([])
    gaps = Vector{Vector{SpectralPoint}}([])

    length(freqs) != 0 || return (bands, gaps)

    betas = beta_disp_point(freqs, d, res)

    collection::Vector{SpectralPoint} = []
    current::Symbol = abs(imag(betas[1])) < band_tolerance ? :band : :gap
    receptor = Dict(:band => bands, :gap => gaps)
    
    for (β, ω) in zip(betas, freqs)
        if abs(imag(β)) < band_tolerance
            if current == :band
                push!(collection, (β, ω))
            else
                push!(receptor[current], collection)
                current = :band
                collection = [(β, ω)]
            end
        elseif current == :gap
            push!(collection, (β, ω))
        else
            push!(receptor[current], collection)
            current = :gap
            collection = [(β, ω)]
        end            
    end
    push!(receptor[current], collection)
    return (bands, gaps)
end

########################################################
##
## Underlying system methods
##
########################################################

"""
    struct WaveSystem

Provide an interface between to the system-level internal representation.

At the lowest level, the transfer matrix representation resides. In this
representation, the field is described by the state vectors (in the basis
of linearly independent free solutions) specified at "bounds" points:
immediately after resonances (x_k) and the terminal points (X_A and X_B).
These state vectors are found using a variation of the transfer matrix
approach. Hence, the interface holds the transfer matricess across the
intervals defined by bounds and the S-matrices at bounds.
"""
@kwdef struct WaveSystem
    k ::Float64                 # frequency (wavenumber)
    # outside of [XA ,XB] the system is the free carrier
    XA ::Float64
    XB ::Float64

    # matrices describing the boundary conditions
    # :A -> G_L, :B -> G_R
    # default: outgoing boundary conditions
    boundary_G ::BoundaryConditions

    res ::Vector{Resonance}     # resonances

    # bounds = [X_A, x^(k), X_B], x^(k) in Resonances
    bounds ::FVector
    
    # the list of transfer matrices across the intervals defined by bounds
    # TM_list = {T_k} : k in {Resonances} U {XB}
    TM_list ::Vector{CMatrix}

    # the list of S matrices at x_k + 0, where x_k ∈ bounds
    SM_list ::Vector{CMatrix}

    # Various logging information for debugging
    log ::Tuple
end

"""
    WaveSystem(frequency ::Float64,
           interval ::Array{Float64, 2},
           resonances ::Array{Resonance})

The common interface to a constructor of `WaveSystem` presuming the
outgoing boundary conditions

INPUT:
    frequency ::Float64           - simulation frequency
    interval ::Array{Float64, 2}  - simulation interval [XA, XB]
    resonances ::Array{Resonance} - list of resonances
"""
function WaveSystem(frequency ::Float64,
                    interval ::FVector,
                    resonances ::Vector{Resonance})
    return WaveSystem(frequency, interval, outgoing_Gs, resonances)
end

function WaveSystem(frequency ::Float64,
                    interval ::FVector,
                    bound_Gs ::BoundaryConditions,
                    resonances ::Vector{Resonance})
    length(interval) > 2 &&
        error("Incorrect specification of the interval in $interval")
    XA = interval[1]
    XB = interval[2]

    # we drop the sources and resonances outside of the interval [XA, XB]
    sort!(resonances, by = r -> r.position)
    num_res_init = length(resonances)
    resonances = resonances[findall(x -> XA < x.position < XB, resonances)]
    if length(resonances) < num_res_init
        @warn "Resonances outside of the interval were discarded"
    end

    # Build the list of transfer matrices across resonances
    TM_list::Vector{CMatrix} =
        [point_transfer(res, frequency) for res in resonances]
    # and the transfer matrix across the right boundary at X_B
    push!(TM_list, Ic)

    # collect sources between resonances
    bounds::FVector = vcat([XA], [r.position for r in resonances], [XB])

    return WaveSystem(frequency, XA, XB, bound_Gs, bounds, resonances, TM_list)
end

"""
    WaveSystem(frequency ::Float64,
               XA ::Float64, XB ::Float64,
               bound_Gs ::Dict( Symbol, CMatrix),
               bounds :: Vector{Float64},
               resonances ::Vector{Resonance},
               TM_list ::Vector{CMatrix})

Constructor working with sanitized list of resonances and resonance transfer
matrices.
Builds the list of S-matrices and finds states at the intervals boundaries.
"""
function WaveSystem(frequency ::Float64,
                    XA ::Float64, XB ::Float64,
                    bound_Gs ::BoundaryConditions,
                    bounds :: FVector,
                    resonances ::Vector{Resonance},
                    TM_list ::Vector{CMatrix})

    (G_L, G_R) = (bound_Gs[:A], bound_Gs[:B])
    num_points::Int64 = length(bounds)

    # SKM_list[M][n] = S_{m - n + 1}(M)
    # with M extending from 1 (XA) to number of Resonances + 2 (XB)
    # S_1(1) corresponds to S_{X_A} = 1/(G_L + G_R), therefore,
    # S_.(n) is S_.(x_{n-1} + 0) for x_k ∈ Resonances
    SKM_tot_list::Vector{Vector{CMatrix}} = [[l_inverse(G_L + G_R)]]

    for len_ind in 2:num_points
        # Populate S_{M - 1}(M) and S_{M}(M)
        SN = SKM_tot_list[len_ind-1][1]
        Sn_1 = l_inverse(Ic - SN * G_R * (Ic - TM_list[len_ind - 1])) * SN
        push!(SKM_tot_list, [TM_list[len_ind - 1] * Sn_1, Sn_1])

        for p = 3:len_ind
            Sp = SKM_tot_list[len_ind-1][p-1] *
                 (Ic + G_R * (Ic - TM_list[len_ind - 1]) * SKM_tot_list[len_ind][2])
            push!(SKM_tot_list[len_ind], Sp)
        end
    end

    # From the total ladder of the S-matrices, we need only one final row
    # in the order corresponding to the order of bound points
    SKM_list::Vector{CMatrix} = [Smat for Smat in reverse(SKM_tot_list[num_points])]

    # The last two S-matrices are identical since they are related
    # by the trivial transfer matrix
    @assert(sum(abs.(SKM_list[end] - SKM_list[end - 1])) < weak_tolerance,
            "The simple test for equality of the last two S-matrices failed!")

    # More detailed test for the consistency
    for (i, transm) in enumerate(TM_list)
        dist = sqrt(sum(abs2, SKM_list[i + 1] - transm * SKM_list[i]))
        if dist > test_tolerance/10
            println("ERROR: The transfer test for S-matrices failed at " *
                "frequency $frequency and interval $i")
            println("Involved matrices are")
            println("S[$i] :\n $(matr_to_str(SKM_list[i]))")
            println("S[$(i + 1)] :\n $(matr_to_str(SKM_list[i + 1]))")
            println("T[$i] :\n $(matr_to_str(transm))")            
            println("The subsequent mismatches are")
            if i < length(TM_list)
                for j in (i + 1):(length(TM_list))
                    distj = sqrt(sum(abs2, SKM_list[j + 1] - transm * SKM_list[j]))
                    print("($j, $distj) ")
                end
            end
            error("Insufficient precision")
        end
    end

    # Here, we call the actual constructor
    return WaveSystem(k = frequency, XA = XA, XB = XB,
                      res = resonances,
                      boundary_G = bound_Gs,
                      bounds = bounds,
                      TM_list = TM_list, SM_list = SKM_list,
                      log = ())
end

########################################################
##
## Loaded system methods
##
########################################################

struct SourceCarrier
    position ::Float64
    amplitude::ComplexF64
end

# The information about how the source updates the state
# In the basis of linearly independent solutions (plane waves)
# ΔΨ = (e^{-ikx}, - e^{ikx})F(x)/2ik
@kwdef struct SourceVector
    position ::Float64
    delta    ::CVector
end

"""
    struct LoadedSystem

An interface to the inhomogeneous (loaded) system.

The state vectors are specified at the "essential" points. These are all
the points, where the state vectors change. Therefore, in addition to the
bounds points in the underlying WaveSystem, we have the points with
sources, where the state vectors experience the respective jumps given by
the source vector (in the same basis).

The continuous form of the field at an arbitrary set of points is
straightforward to restore since between the essential points the state
vectors are constant.
"""
@kwdef struct LoadedSystem
    ws ::WaveSystem
    # the sources grouped according to ws.bounds
    src_coll ::Vector{Vector{SourceVector}}
    # Esential points: EP_list = bounds U {Carrier sources}
    EP_list ::FVector
    # states in the basis of free solutions
    # ψ_e(r^(k) + 0) : r^{(k)} in EP_list
    states ::Vector{CVector}
    # Various logging information for debugging
    log ::Tuple
end

"""
    split_ext(carrier_src::Vector{SourceVector},
              bounds ::Vector{Float64}) ::Vector{Vector{SourceVector}}

Return an array of `Vector{SourceVector}` containing carrier
excitations located within the intervals identified by `bounds` array.
"""
function split_ext(carrier_src::Vector{SourceVector},
                   bounds ::FVector) ::Vector{Vector{SourceVector}}

    collector = Vector{Vector{SourceVector}}([])
    for i in 1:(length(bounds) - 1)
        insides ::Vector{SourceVector} =
            carrier_src[findall(x -> bounds[i] <= x.position < bounds[i + 1],
                                carrier_src)]
        push!(collector, insides)
    end
    return collector
end

"""
    src_vector(k::Float64, src::SourceCarrier) ::CVector

Return the source vector (e^{-ikx} , -e^{ikx})/2ik at point `pos` corresponding to
frequency `freq`
"""
function src_vector(k::Float64, src::SourceCarrier) ::CVector
    phase::Float64 = k * src.position
    return src.amplitude .* CVector([imexp(-phase), -imexp(phase)]./(2im * k))
end

function LoadedSystem(ws ::WaveSystem, carrier_src ::Vector{SourceCarrier})
    (G_L, G_R) = (ws.boundary_G[:A], ws.boundary_G[:B])
    num_car_src_init = length(carrier_src)
    sort!(carrier_src, by = r -> r.position)
    carrier_src = carrier_src[findall(x -> ws.XA < x.position < ws.XB, carrier_src)]
    if length(carrier_src) < num_car_src_init
        println("WARNING: Carrier sources outside of the interval were discarded")
    end
    # Basic validation
    @assert(length(carrier_src) > 0,
            "The invoked constructor does not support free systems " *
              "(no sources fall within the specified interval)")

    # convert the list of sources to the list of vector updates
    # NOTE: currently deals only with the carrier sources
    src_list::Vector{SourceVector} =
        [SourceVector(position = x.position,
                      delta = src_vector(ws.k, x))
        for x in carrier_src]
    src_coll = split_ext(src_list, ws.bounds)

    # Build the cumulative source within bound-defined intervals
    fm_list ::Vector{CVector} = []
    for (ind, src_batch) in enumerate(src_coll)
        Tm1 = ws.TM_list[ind]
        fm_add ::CVector = [0, 0]
        for source in src_batch
            fm_add += source.delta
        end
        push!(fm_list, Tm1 * fm_add)
    end

    # Construct a particular solution of the inhomogeneous problem
    # u_n^f = S_n \sum_{1 < m ≤ n} S_m^{-1} f_{m - 1}
    # with u_1^f = 0, and f_m = T_{m+1} Σ_k source.delta[k],
    # where k runs over sources between x^(m) and x^{m+1}
    uf_list ::Vector{CVector} = [[0, 0]]
    f_acc::CVector = [0, 0]
    for n in 2:length(ws.bounds)
        f_acc += l_inverse(ws.SM_list[n]) * fm_list[n - 1]
        push!(uf_list, ws.SM_list[n] * f_acc)
    end
    
    # states at ws.bounds
    state_bounds_list::Vector{CVector} = uf_list
    for (num, Smat) in enumerate(ws.SM_list)
        state_bounds_list[num] -= Smat * G_R * uf_list[end]
    end

    # Verifying the solution
    for interval in 1:(length(state_bounds_list) - 1)
        mismatch = state_bounds_list[interval + 1] -
            ws.TM_list[interval] * state_bounds_list[interval] - fm_list[interval]
        dist = sqrt(Float64(conj(transpose(mismatch)) * mismatch))
        if dist > test_tolerance
            out_str = "Solution test failed at frequency $(ws.k) :" *
                "interval $interval ($(ws.bounds[interval]), $(ws.bounds[interval + 1]))\n" *
                " Ψ_1 = $(state_to_str(state_bounds_list[interval]))\n" *
                " Ψ_+ = $(state_to_str(state_bounds_list[interval + 1])) " *
                " with mismatch $dist:\n"
            @warn out_str
        end
    end 

    # The essential points and states there
    EP_list :: FVector = []
    state_list :: Vector{CVector} = []
    for i in 1:(length(ws.bounds) - 1)
        push!(EP_list, ws.bounds[i])
        state = state_bounds_list[i]
        push!(state_list, state)
        for src_carr in src_coll[i]
            push!(EP_list, src_carr.position)
            state += src_carr.delta
            push!(state_list, state)
        end
    end
    push!(EP_list, ws.bounds[end])
    push!(state_list, state_bounds_list[end])

    return LoadedSystem(ws = ws,
                        src_coll = src_coll,
                        EP_list = EP_list,
                        states = state_list,
                        log = ())
end

########################################################
##
## Field methods
##
########################################################

"""
    get_states(ls::LoadedSystem, pts::FVector)::Tuple{FVector, Vector{CVector}}

Return the state vectors in the basis of solutions of the free system at
the (sanitized) coordinates provided in `points` for the loaded system
`ls`.

The list of coordinates is cleaned to keep points strictly inside of
the system interval and to eliminate points coinciding with resonances
(amplitudes are discontinuous there).

OUTPUT: (pts :: FVector, ampls :: Vector{CVector})
        pts - cleaned list of points
        ampls - a 2-vector of amplitudes for each element in `pts` 
"""
function get_states(ls::LoadedSystem, pts::FVector)::Tuple{FVector, Vector{CVector}}
    # clean up the observation points
    num_pts_init = length(pts)
    sort!(pts)
    pts = pts[findall(x -> ls.ws.XA < x < ls.ws.XB, pts)]
    if length(pts) < num_pts_init
        out_str = "WARNING: observation points outside of the system interval are ignored\n" *
            "WARNING: $(length(pts)) are kept out of $num_pts_init"
        @info out_str
    end

    groups::Vector{Int} = [] # number of observers within the EP_list intervals
    for i in 1:(length(ls.EP_list) - 1)
        push!(groups,
              length(pts[findall(x -> ls.EP_list[i] <= x < ls.EP_list[i + 1],
                                 pts)]))
    end
    
    ampls::Vector{CVector} = [] # collects amplitudes at observers
    for (ind, state) in enumerate(ls.states[1:(end-1)])
        append!(ampls, [state for _ in 1:groups[ind]])
    end
    @assert(length(pts) == length(ampls),
            "The lists of oberservers (len(pts) = $(length(pts))) " *
                "and states (len(ampls) = $(length(ampls))) do not match!")

    return (pts, ampls)
end

function get_field(ls::LoadedSystem, points::FVector)
    (pts, ampls) = get_states(ls, points)

    field = [transpose(ampls[ind]) * [imexp(phase), imexp(-phase)]
              for (ind, phase) in enumerate(ls.ws.k .* pts)]
    return (pts, field)
end

function get_field_resonances(ls::LoadedSystem, res_numbers::Vector{Int}) ::CVector
    sort!(res_numbers)
    max_res_num = length(ls.ws.res)
    if minimum(res_numbers) < 1 || maximum(res_numbers) > max_res_num
        out_str = "WARNING: the list of resonances is truncated"
        @info out_str
    end
    res_numbers = res_numbers[findall(x -> 0 < x <= max_res_num, res_numbers)]

    freq = ls.ws.k
    ampls = ls.states[res_numbers]
    pts = [ls.ws.res[num].position for num in res_numbers]
    field = [transpose(ampls[ind]) * [imexp(phase), imexp(-phase)]
             for (ind, phase) in enumerate(freq .* pts)]

    res_fields = [-field[num] * ls.ws.res[num].frequency^2 * res_response(ls.ws.res[num], freq) for num in res_numbers]
    return res_fields
end

############################################################
##
##  Service methods
##
############################################################

function f_to_str(f::Float64) ::String
    return @sprintf("%.4f", f)
end

function c_to_str(f::ComplexF64) ::String
    return @sprintf("%.4f + i %.4f", real(f), imag(f))
end

function state_to_str(state::CVector) ::String
    s1_str = c_to_str(state[1])
    s2_str = c_to_str(state[2])        
    return "( $s1_str ,  $s2_str )"
end

function matr_to_str(matr::CMatrix) ::String
    s11_str = c_to_str(matr[1, 1])
    s12_str = c_to_str(matr[1, 2])
    s21_str = c_to_str(matr[2, 1])
    s22_str = c_to_str(matr[2, 2])
    return "( $s11_str ,  $s12_str \n  $s21_str, $s22_str)"
end

"""
    show_info(ls::LoadedSystem; level = 0)

print information contained in loaded system `ls`. The output information
is controlled by `level`.

Level of details
      0 (default) - output all fields
  Next levels are incremental
      1 - frequency, interval, boundary states
      2 - essential points
      3 - after-EP states
      4 - positions and properties of resonances
      5 - points and parameters of excitations
"""
function show_info(ls::LoadedSystem; level = 0)
    cut_off_log = 1
    println("\n************ WAVE SYSTEM REPORT ************\n")
    println("** System:")
    println("Frequency: $(ls.ws.k)")
    println("Central interval: [$(ls.ws.XA), $(ls.ws.XB)]")

    println("\n** Boundary states:")
    println("ψ_A = $(state_to_str(ls.states[1]))\n")
    println("ψ_B = $(state_to_str(ls.states[end]))")

    level == cut_off_log && return
    cut_off_log += 1

    println("\n** Essential points:")
    println("There are $(length(ls.EP_list)) essential points with $(length(ls.states)) after-EP states")

    level == cut_off_log && return
    cut_off_log += 1

    println("\n** After-EP states:")
    for (count, state) in enumerate(ls.states)
        println("\t$count : ψ($(f_to_str(ls.EP_list[count]))) = $(state_to_str(state))\n")
    end

    level == cut_off_log && return
    cut_off_log += 1

    println("** Resonances:")
    println("The system contains $(length(ls.ws.res)) resonances:")
    for (count, res) in enumerate(ls.ws.res)
        println("\t$count : x = $(f_to_str(res.position)) Ω = $(res.frequency) κ = $(res.kappa)")
    end

    level == cut_off_log && return
    cut_off_log += 1

    println("\n** Sources:")
    if length(ls.src_coll) == 0
        println("There are 0 carrier sources")
    else
        groups = [length(g) for g in ls.src_coll]
        bounds = vcat([ls.ws.XA], [r.position for r in ls.ws.res], [ls.ws.XB])
        println("There are $(sum(groups)) carrier sources")
        for (ind, g) in enumerate(ls.src_coll)
            left_b = bounds[ind]
            right_b = bounds[ind + 1]
            println("\tGroup : $ind in [$(f_to_str(left_b)), $(f_to_str(right_b))]")

            if length(g) == 0
                println("\t\t[]")
                continue
            end

            for (count_src, src) in enumerate(g)
                println("\t\t$count_src : r = $(f_to_str(src.position)) F = $(src.delta)")
            end
        end
    end
end

Base.show(io::IO, ls::LoadedSystem) = show_info(ls)

end # module ends here
