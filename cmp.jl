# cmp.jl - Julia backend for cmp Stata package
# Translated from cmp.mata
# Copyright (C) 2007-24 David Roodman
# Licensed under GNU GPL v3

module CMP

using LinearAlgebra, Distributions, SparseArrays, GHK, StatsFuns

export CmpModel, lf1!, gf1!, cmp_init!, setNumREDraws!

# Model type constants
const cmp_cont = 1
const cmp_left = 2
const cmp_right = 3
const cmp_probit = 4
const cmp_oprobit = 5
const cmp_mprobit = 6
const cmp_int = 7
const cmp_probity1 = 8
const cmp_frac = 10
const mprobit_ind_base = 20
const roprobit_ind_base = 40

# Type aliases for cleaner code
const Mat = Matrix{Float64}
const VecMat = Vector{Matrix{Float64}}
const MatMat = Matrix{Matrix{Float64}}
const VecVecMat = Vector{Vector{Matrix{Float64}}}

# Multinomial probit group info
mutable struct MprobitGroup
    d::Int          # dimension - 1
    out::Int        # eq of chosen alternative
    in_::Vector{Int}   # eqs of remaining alternatives
    res::Vector{Int}   # indices in ECens to hold relative differences

    MprobitGroup() = new(0, 0, Int[], Int[])
end

# Score indices for different parameter groups
mutable struct Scores
    ThetaScores::Vector{Int}    # cols of master score matrix for theta params
    CutScores::Vector{Int}      # cols for cut parameters
    TScores::VecMat             # for each level
    SigScores::VecMat           # only at top level
    GammaScores::VecMat

    Scores() = new(Int[], Int[], Mat[], Mat[], Mat[])
end

# Subview: info associated with subsets of data defined by indicator combinations
mutable struct Subview
    EUncens::Mat
    pECens::Mat
    pF::Mat
    pEt::Mat
    pFt::Mat
    Fi::Vector{Float64}

    theta::VecMat
    y::VecMat
    Lt::VecMat
    Ut::VecMat
    yL::VecMat

    dOmega_dGamma::MatMat
    Scores_::Vector{Vector{Scores}}  # one col for each level, one for each draw
    Yi::Mat
    subsample::Vector{Float64}
    SubsampleInds::Vector{Int}
    one2N::Vector{Int}

    GHKStart::Int
    GHKStartTrunc::Int
    d_uncens::Int
    d_cens::Int
    d2_cens::Int
    d_two_cens::Int
    d_oprobit::Int
    d_trunc::Int
    d_frac::Int
    NFracCombs::Int
    N::Int

    NumCuts::Int
    vNumCuts::Vector{Int}
    dSig_dLTSig::Mat
    N_perm::Int
    CensLTInds::Vector{Int}
    WeightProduct::Vector{Float64}
    TheseInds::Vector{Float64}

    uncens::Vector{Int}
    two_cens::Vector{Int}
    oprobit::Vector{Int}
    cens::Vector{Int}
    cens_nonrobase::Vector{Int}
    trunc::Vector{Int}
    one2d_trunc::Vector{Int}
    frac::Vector{Int}
    censnonfrac::Vector{Int}
    cens_uncens::Vector{Int}

    SigIndsUncens::Vector{Int}
    SigIndsTrunc::Vector{Int}
    SigIndsCensUncens::Vector{Int}
    CutInds::Vector{Int}
    NotBaseEq::Vector{Bool}

    QSig::Mat
    Sig::Mat
    Omega::Mat
    QE::Mat
    QEinvGamma::Mat
    invGammaQSigD::Mat

    dCensNonrobase::Int
    J_d_uncens_d_cens_0::Mat
    J_d_cens_d_0::Mat
    J_d2_cens_d2_0::Mat
    J_N_1_0::Mat

    dphi_dE::Mat
    dPhi_dE::Mat
    dPhi_dSig::Mat
    dPhi_dcuts::Mat
    dPhi_dF::Mat
    dPhi_dpF::Mat
    dPhi_dEt::Mat
    dphi_dSig::Mat
    dPhi_dSigt::Mat
    dPhi_dpE_dSig::Mat
    _dPhi_dpE_dSig::Mat
    _dPhi_dpF_dSig::Mat
    dPhi_dpF_dSig::Mat
    EDE::Mat

    dPhi_dpE::VecMat
    dPhi_dpSig::VecMat
    XU::VecVecMat
    id::VecMat

    roprobit_QE::VecMat
    roprobit_Q_Sig::VecMat
    mprobit::Vector{MprobitGroup}
    halfDmatrix::Mat
    FracCombs::Mat
    frac_QE::VecMat
    frac_QSig::VecMat
    yProd::VecMat

    next::Union{Subview, Nothing}

    function Subview()
        sv = new()
        sv.EUncens = Mat(undef, 0, 0)
        sv.pECens = Mat(undef, 0, 0)
        sv.pF = Mat(undef, 0, 0)
        sv.pEt = Mat(undef, 0, 0)
        sv.pFt = Mat(undef, 0, 0)
        sv.Fi = Float64[]
        sv.theta = Mat[]
        sv.y = Mat[]
        sv.Lt = Mat[]
        sv.Ut = Mat[]
        sv.yL = Mat[]
        sv.dOmega_dGamma = MatMat(undef, 0, 0)
        sv.Scores_ = Vector{Scores}[]
        sv.Yi = Mat(undef, 0, 0)
        sv.subsample = Float64[]
        sv.SubsampleInds = Int[]
        sv.one2N = Int[]
        sv.GHKStart = 0
        sv.GHKStartTrunc = 0
        sv.d_uncens = 0
        sv.d_cens = 0
        sv.d2_cens = 0
        sv.d_two_cens = 0
        sv.d_oprobit = 0
        sv.d_trunc = 0
        sv.d_frac = 0
        sv.NFracCombs = 0
        sv.N = 0
        sv.NumCuts = 0
        sv.vNumCuts = Int[]
        sv.dSig_dLTSig = Mat(undef, 0, 0)
        sv.N_perm = 0
        sv.CensLTInds = Int[]
        sv.WeightProduct = Float64[]
        sv.TheseInds = Float64[]
        sv.uncens = Int[]
        sv.two_cens = Int[]
        sv.oprobit = Int[]
        sv.cens = Int[]
        sv.cens_nonrobase = Int[]
        sv.trunc = Int[]
        sv.one2d_trunc = Int[]
        sv.frac = Int[]
        sv.censnonfrac = Int[]
        sv.cens_uncens = Int[]
        sv.SigIndsUncens = Int[]
        sv.SigIndsTrunc = Int[]
        sv.SigIndsCensUncens = Int[]
        sv.CutInds = Int[]
        sv.NotBaseEq = Bool[]
        sv.QSig = Mat(undef, 0, 0)
        sv.Sig = Mat(undef, 0, 0)
        sv.Omega = Mat(undef, 0, 0)
        sv.QE = Mat(undef, 0, 0)
        sv.QEinvGamma = Mat(undef, 0, 0)
        sv.invGammaQSigD = Mat(undef, 0, 0)
        sv.dCensNonrobase = 0
        sv.J_d_uncens_d_cens_0 = Mat(undef, 0, 0)
        sv.J_d_cens_d_0 = Mat(undef, 0, 0)
        sv.J_d2_cens_d2_0 = Mat(undef, 0, 0)
        sv.J_N_1_0 = Mat(undef, 0, 0)
        sv.dphi_dE = Mat(undef, 0, 0)
        sv.dPhi_dE = Mat(undef, 0, 0)
        sv.dPhi_dSig = Mat(undef, 0, 0)
        sv.dPhi_dcuts = Mat(undef, 0, 0)
        sv.dPhi_dF = Mat(undef, 0, 0)
        sv.dPhi_dpF = Mat(undef, 0, 0)
        sv.dPhi_dEt = Mat(undef, 0, 0)
        sv.dphi_dSig = Mat(undef, 0, 0)
        sv.dPhi_dSigt = Mat(undef, 0, 0)
        sv.dPhi_dpE_dSig = Mat(undef, 0, 0)
        sv._dPhi_dpE_dSig = Mat(undef, 0, 0)
        sv._dPhi_dpF_dSig = Mat(undef, 0, 0)
        sv.dPhi_dpF_dSig = Mat(undef, 0, 0)
        sv.EDE = Mat(undef, 0, 0)
        sv.dPhi_dpE = Mat[]
        sv.dPhi_dpSig = Mat[]
        sv.XU = VecMat[]
        sv.id = Mat[]
        sv.roprobit_QE = Mat[]
        sv.roprobit_Q_Sig = Mat[]
        sv.mprobit = MprobitGroup[]
        sv.halfDmatrix = Mat(undef, 0, 0)
        sv.FracCombs = Mat(undef, 0, 0)
        sv.frac_QE = Mat[]
        sv.frac_QSig = Mat[]
        sv.yProd = Mat[]
        sv.next = nothing
        return sv
    end
end

# Random effects level info
mutable struct RE
    R::Int           # number of draws
    d::Int           # number of RCs and REs
    d2::Int          # triangular number of d
    one2d::Vector{Int}
    one2R::Vector{Int}
    J1R0::Vector{Float64}
    JN12::Vector{Float64}
    JN1pQuadX::VecMat
    HasRC::Bool
    J_N_NEq_0::Mat
    REInds::Vector{Int}
    RCInds::VecMat
    Eqs::Vector{Int}
    REEqs::Vector{Int}
    GammaEqs::Vector{Int}
    NEq::Int
    NEff::Vector{Int}
    X::VecMat
    U::VecMat
    pXU::MatMat
    TotalEffect::MatMat
    Sig::Mat
    T::Mat
    invGamma::Mat
    D::Mat
    dSigdParams::Mat
    NSigParams::Int
    N::Int
    one2N::Vector{Int}
    J_N_1_0::Vector{Float64}
    J_N_0_0::Vector{Float64}
    IDRanges::Matrix{Int}
    IDRangeLengths::Vector{Int}
    IDRangesGroup::Matrix{Int}
    Subscript::VecMat
    id::Mat
    sig::Vector{Float64}
    rho::Vector{Float64}
    covAcross::Int
    covWithin::Vector{Int}
    FixedSigs::Vector{Float64}
    FixedRhos::Mat
    theta::VecMat
    Weights::Vector{Float64}
    ToAdapt::Vector{Float64}
    lnNumREDraws::Float64
    lnLlimits::Float64
    lnLByDraw::Mat
    plnL::Vector{Float64}
    QuadW::Vector{Float64}
    QuadX::Vector{Float64}
    QuadMean::VecMat
    QuadSD::VecMat
    lnnormaldenQuadX::Vector{Float64}
    QuadXAdapt::Dict{Any, Any}
    AdaptivePhaseThisIter::Bool
    AdaptiveShift::Float64
    Rho::Mat
    RCk::Vector{Int}

    function RE()
        re = new()
        re.R = 0
        re.d = 0
        re.d2 = 0
        re.one2d = Int[]
        re.one2R = Int[]
        re.J1R0 = Float64[]
        re.JN12 = Float64[]
        re.JN1pQuadX = Mat[]
        re.HasRC = false
        re.J_N_NEq_0 = Mat(undef, 0, 0)
        re.REInds = Int[]
        re.RCInds = Mat[]
        re.Eqs = Int[]
        re.REEqs = Int[]
        re.GammaEqs = Int[]
        re.NEq = 0
        re.NEff = Int[]
        re.X = Mat[]
        re.U = Mat[]
        re.pXU = MatMat(undef, 0, 0)
        re.TotalEffect = MatMat(undef, 0, 0)
        re.Sig = Mat(undef, 0, 0)
        re.T = Mat(undef, 0, 0)
        re.invGamma = Mat(undef, 0, 0)
        re.D = Mat(undef, 0, 0)
        re.dSigdParams = Mat(undef, 0, 0)
        re.NSigParams = 0
        re.N = 0
        re.one2N = Int[]
        re.J_N_1_0 = Float64[]
        re.J_N_0_0 = Float64[]
        re.IDRanges = Matrix{Int}(undef, 0, 0)
        re.IDRangeLengths = Int[]
        re.IDRangesGroup = Matrix{Int}(undef, 0, 0)
        re.Subscript = Mat[]
        re.id = Mat(undef, 0, 0)
        re.sig = Float64[]
        re.rho = Float64[]
        re.covAcross = 0
        re.covWithin = Int[]
        re.FixedSigs = Float64[]
        re.FixedRhos = Mat(undef, 0, 0)
        re.theta = Mat[]
        re.Weights = Float64[]
        re.ToAdapt = Float64[]
        re.lnNumREDraws = 0.0
        re.lnLlimits = 0.0
        re.lnLByDraw = Mat(undef, 0, 0)
        re.plnL = Float64[]
        re.QuadW = Float64[]
        re.QuadX = Float64[]
        re.QuadMean = Mat[]
        re.QuadSD = Mat[]
        re.lnnormaldenQuadX = Float64[]
        re.QuadXAdapt = Dict{Any, Any}()
        re.AdaptivePhaseThisIter = false
        re.AdaptiveShift = 0.0
        re.Rho = Mat(undef, 0, 0)
        re.RCk = Int[]
        return re
    end
end

# Main model container
mutable struct CmpModel
    REs::Vector{RE}
    base::RE
    subviews::Vector{Subview}

    y::VecMat
    Lt::VecMat
    Ut::VecMat
    yL::VecMat
    Theta::Mat

    d::Int           # number of equations
    L::Int           # number of levels
    _todo::Int       # 0 = likelihood only, 1 = likelihood + scores
    ghkDraws::Int
    ghkScramble::Int
    REScramble::Int
    REAnti::Int
    NumRoprobitGroups::Int
    MaxCuts::Int
    NSimEff::Int

    MprobitGroupInds::Matrix{Int}
    RoprobitGroupInds::Matrix{Int}
    NumREDraws::Vector{Int}
    NonbaseCases::Vector{Int}
    reverse::Bool
    ghkType::String
    REType::String

    Gamma::Mat
    pOmega::Mat
    dSig_dT::Mat
    WeightProduct::Vector{Float64}

    ghk2DrawSet::Any  # GHK draw set object
    ghkAnti::Bool
    NumCuts::Int
    HasGamma::Bool
    SigXform::Int
    d_cens::Int

    Eqs::Mat
    GammaId::Mat
    NumEff::Mat
    vNumCuts::Vector{Int}
    cuts::Mat
    G::Vector{Int}
    GammaIndByEq::Vector{Vector{Int}}
    GammaInd::Matrix{Int}
    dOmega_dGamma::MatMat

    trunceqs::Vector{Int}
    intregeqs::Vector{Int}
    Quadrature::Bool
    AdaptivePhaseThisEst::Bool
    WillAdapt::Bool
    QuadTol::Float64
    QuadIter::Int
    Adapted::Bool
    AdaptNextTime::Bool

    Lastb::Vector{Float64}
    LastlnLLastIter::Float64
    LastlnLThisIter::Float64
    LastIter::Int

    Idd::Mat
    vKd::Vector{Float64}
    vIKI::Vector{Float64}
    vLd::Vector{Float64}

    indicators::Mat
    S0::Mat
    Scores_::Scores

    indVars::Vector{String}
    LtVars::Vector{String}
    UtVars::Vector{String}
    yLVars::Vector{String}

    ThisDraw::Vector{Int}
    h::Float64
    X::VecMat
    sTScores::VecMat
    sGammaScores::VecMat

    function CmpModel()
        m = new()
        m.REs = RE[]
        m.base = RE()
        m.subviews = Subview[]
        m.y = Mat[]
        m.Lt = Mat[]
        m.Ut = Mat[]
        m.yL = Mat[]
        m.Theta = Mat(undef, 0, 0)
        m.d = 0
        m.L = 0
        m._todo = 0
        m.ghkDraws = 0
        m.ghkScramble = 0
        m.REScramble = 0
        m.REAnti = 1
        m.NumRoprobitGroups = 0
        m.MaxCuts = 0
        m.NSimEff = 0
        m.MprobitGroupInds = Matrix{Int}(undef, 0, 0)
        m.RoprobitGroupInds = Matrix{Int}(undef, 0, 0)
        m.NumREDraws = Int[]
        m.NonbaseCases = Int[]
        m.reverse = false
        m.ghkType = ""
        m.REType = ""
        m.Gamma = Mat(undef, 0, 0)
        m.pOmega = Mat(undef, 0, 0)
        m.dSig_dT = Mat(undef, 0, 0)
        m.WeightProduct = Float64[]
        m.ghk2DrawSet = nothing
        m.ghkAnti = false
        m.NumCuts = 0
        m.HasGamma = false
        m.SigXform = 0
        m.d_cens = 0
        m.Eqs = Mat(undef, 0, 0)
        m.GammaId = Mat(undef, 0, 0)
        m.NumEff = Mat(undef, 0, 0)
        m.vNumCuts = Int[]
        m.cuts = Mat(undef, 0, 0)
        m.G = Int[]
        m.GammaIndByEq = Vector{Int}[]
        m.GammaInd = Matrix{Int}(undef, 0, 0)
        m.dOmega_dGamma = MatMat(undef, 0, 0)
        m.trunceqs = Int[]
        m.intregeqs = Int[]
        m.Quadrature = false
        m.AdaptivePhaseThisEst = false
        m.WillAdapt = false
        m.QuadTol = 0.0
        m.QuadIter = 0
        m.Adapted = false
        m.AdaptNextTime = false
        m.Lastb = Float64[]
        m.LastlnLLastIter = 0.0
        m.LastlnLThisIter = 0.0
        m.LastIter = 0
        m.Idd = Mat(undef, 0, 0)
        m.vKd = Float64[]
        m.vIKI = Float64[]
        m.vLd = Float64[]
        m.indicators = Mat(undef, 0, 0)
        m.S0 = Mat(undef, 0, 0)
        m.Scores_ = Scores()
        m.indVars = String[]
        m.LtVars = String[]
        m.UtVars = String[]
        m.yLVars = String[]
        m.ThisDraw = Int[]
        m.h = 0.0
        m.X = Mat[]
        m.sTScores = Mat[]
        m.sGammaScores = Mat[]
        return m
    end
end

# Set number of random effect draws
function setNumREDraws!(model::CmpModel, t::AbstractVector)
    model.NumREDraws = vcat([1], t .* model.REAnti)
    nothing
end

# Vectorize lower triangle of symmetric matrix (Mata's vech)
function vech(A::AbstractMatrix)
    n = size(A, 1)
    result = Float64[]
    for j in 1:n
        for i in j:n
            push!(result, A[i, j])
        end
    end
    return result
end

# Inverse of vech: reconstruct symmetric matrix from vectorized lower triangle
function invvech(v::AbstractVector)
    n = Int((-1 + sqrt(1 + 8*length(v))) / 2)
    A = zeros(n, n)
    k = 1
    for j in 1:n
        for i in j:n
            A[i, j] = v[k]
            A[j, i] = v[k]
            k += 1
        end
    end
    return A
end

# Panel setup: returns matrix of start/end indices for each group
function panelsetup(id::AbstractVector)
    n = length(id)
    if n == 0
        return Matrix{Int}(undef, 0, 2)
    end

    starts = Int[1]
    for i in 2:n
        if id[i] != id[i-1]
            push!(starts, i)
        end
    end

    ngroups = length(starts)
    info = Matrix{Int}(undef, ngroups, 2)
    for g in 1:ngroups-1
        info[g, 1] = starts[g]
        info[g, 2] = starts[g+1] - 1
    end
    info[ngroups, 1] = starts[ngroups]
    info[ngroups, 2] = n

    return info
end

# Panel sum: sum X within panels defined by info
function panelsum(X::AbstractMatrix, info::AbstractMatrix)
    ngroups = size(info, 1)
    result = zeros(ngroups, size(X, 2))
    for g in 1:ngroups
        result[g, :] = sum(X[info[g,1]:info[g,2], :], dims=1)
    end
    return result
end

function panelsum(X::AbstractMatrix, W::AbstractVector, info::AbstractMatrix)
    ngroups = size(info, 1)
    result = zeros(ngroups, size(X, 2))
    for g in 1:ngroups
        rows = info[g,1]:info[g,2]
        result[g, :] = sum(W[rows] .* X[rows, :], dims=1)
    end
    return result
end

function panelsum(X::AbstractVector, info::AbstractMatrix)
    ngroups = size(info, 1)
    result = zeros(ngroups)
    for g in 1:ngroups
        result[g] = sum(X[info[g,1]:info[g,2]])
    end
    return result
end

# Quadratic row sum of log normal density: a - 0.5 * sum(X.^2, dims=2)
# This is: a + rowsum(lnnormalden(X))
function quadrowsum_lnnormalden(X::AbstractMatrix, a::Real)
    ln2pi_2 = 0.91893853320467267  # ln(2*pi)/2
    return (a - ln2pi_2 * size(X, 2)) .- 0.5 .* vec(sum(X .* X, dims=2))
end

# Elimination matrix: maps vec(A) to vech(A) for symmetric A
# Sparse version from old/cmp.jl for efficiency
function Lmatrix(d::Int)
    d2 = d * (d + 1) ÷ 2
    sparse(1:d2, [i*d + j for i in 0:d-1 for j in i+1:d], ones(Int, d2))
end

# Duplication matrix: maps vech(A) to vec(A) for symmetric A
# Sparse version from old/cmp.jl for efficiency
function Dmatrix(d::Int)
    d2 = d^2
    sparse(1:d2, [(c = min(i, j); max(i, j) + (2d - c - 1) * c ÷ 2 + 1) for i in 0:d-1 for j in 0:d-1], ones(Int, d2))
end

# Transform QE (error correction) to QSig (covariance correction)
# QSig = L * (QE ⊗ QE)' * D
function QE2QSig(QE::AbstractMatrix)
    n_out = size(QE, 1)
    n_in = size(QE, 2)
    return Lmatrix(n_out) * kron(QE, QE)' * Dmatrix(n_in)
end

# Derivative of Sigma w.r.t. lnsig's and atanhrho's
function dSigdsigrhos(SigXform::Int, sig::AbstractVector, Sig::AbstractMatrix,
                      rho::AbstractVector, Rho::AbstractMatrix)
    _d = length(sig)
    _d2 = _d + length(rho)
    D = Matrix{Float64}(I, _d2, _d2)

    for k in 1:_d
        t = zeros(_d, _d)
        t2 = SigXform != 0 ? Sig[k, :] : (_d > 1 ? Rho[k, :] .* sig : sig)
        t[k, :] = t2
        t[:, k] = t[:, k] + t2
        D[:, k] = vech(t)
    end

    if _d > 1
        k = _d + 1
        for j in 1:_d
            for i in j+1:_d
                t = zeros(_d, _d)
                t[i, j] = sig[i] * sig[j]
                D[:, k] = vech(t)
                k += 1
            end
        end
        if SigXform != 0
            # Datanh = cosh^2
            t = cosh.(rho)
            D[:, _d+1:end] = D[:, _d+1:end] ./ (t .* t)'
        end
    end

    return D
end

# Compute normal(F) - normal(E) while maximizing precision
# When F+E < 0, compute via reflection to avoid catastrophic cancellation
function normal2(E::AbstractVector, F::AbstractVector)
    sign = (F .+ E) .< 0
    sign = 2 .* sign .- 1  # -1 or 1
    return abs.(normcdf.(sign .* F) .- normcdf.(sign .* E))
end

# Bivariate normal CDF using Genz (2004) algorithm
# Based on Drezner & Wesolowsky (1989)
function binormalGenz(dh::AbstractVector, dk::AbstractVector, r::Real,
                      sign::AbstractVector=ones(length(dh)))
    # Gauss-Legendre weights and abscissae for 10-point rule
    w = [0.2491470458134028, 0.2491470458134028, 0.2334925365383548,
         0.2334925365383548, 0.2031674267230659, 0.2031674267230659,
         0.1600783285433462, 0.1600783285433462, 0.1069393259953184,
         0.1069393259953184, 0.0471753363865118, 0.0471753363865118]
    x = [-0.1252334085114689, 0.1252334085114689, -0.3678314989981802,
         0.3678314989981802, -0.5873179542866175, 0.5873179542866175,
         -0.7699026741943047, 0.7699026741943047, -0.9041172563704749,
         0.9041172563704749, -0.9815606342467192, 0.9815606342467192]

    n = length(dh)
    bvn = zeros(n)

    # Handle correlation = 0 case
    if abs(r) < eps()
        return normcdf.(dh) .* normcdf.(sign .* dk)
    end

    # Handle correlation = ±1 cases
    if abs(r) >= 1 - eps()
        if r > 0
            return normcdf.(min.(dh, dk))
        else
            return max.(0, normcdf.(dh) .- normcdf.(-dk))
        end
    end

    # General case using Gauss-Legendre quadrature
    h = dh
    k = sign .* dk
    hk = h .* k

    if abs(r) < 0.925
        # Standard quadrature
        hs = (h .* h .+ k .* k) ./ 2
        asr = asin(r)
        for i in 1:12
            sn = sin(asr * (1 + x[i]) / 2)
            bvn .+= w[i] .* exp.((sn .* hk .- hs) ./ (1 - sn^2))
            sn = sin(asr * (1 - x[i]) / 2)
            bvn .+= w[i] .* exp.((sn .* hk .- hs) ./ (1 - sn^2))
        end
        bvn = bvn .* asr ./ (4π) .+ normcdf.(-h) .* normcdf.(-k)
    else
        # Use alternate formula for high correlations
        if r < 0
            k = -k
            hk = -hk
        end

        ass = (1 - r) * (1 + r)
        a = sqrt(ass)
        bs = (h .- k).^2
        c = (4 - hk) / 8
        d = (12 - hk) / 16
        asr = -(bs ./ ass .+ hk) / 2

        @. bvn = a * exp(asr) * (1 - c * (bs - ass) * (1 - d * bs / 5) / 3 + c * d * ass^2 / 5)

        if -hk[1] < 100  # Approximate check
            b = sqrt(bs)
            @. bvn -= exp(-hk / 2) * sqrt(2π) * normcdf(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
        end

        a = a / 2
        for i in 1:12
            for is in [-1, 1]
                xs = (a * (is * x[i] + 1))^2
                rs = sqrt(1 - xs)
                asr = -(bs ./ xs .+ hk) / 2
                @. bvn += a * w[i] * exp(asr) * (exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs -
                                                   (1 + c * xs * (1 + d * xs)))
            end
        end
        bvn = -bvn / (2π)

        if r > 0
            bvn .+= normcdf.(-max.(h, k))
        else
            bvn = -bvn
            @. bvn[h < 0] += normcdf(k[h < 0]) - normcdf(h[h < 0])
            @. bvn[h >= 0] += normcdf(-h[h >= 0]) - normcdf(-k[h >= 0])
        end
    end

    return max.(0, min.(1, bvn))
end

# binormal2: integral of bivariate normal from -infinity to E1, F2 to E2
# Done to maximize precision as in normal2()
function binormal2(E1::AbstractVector, E2::AbstractVector, F2::AbstractVector, rho::Real)
    sign = (E2 .+ F2) .< 0
    sign = 2 .* sign .- 1
    return abs.(binormalGenz(E1, sign .* E2, rho, sign) .- binormalGenz(E1, sign .* F2, rho, sign))
end

# Divide matrix/vector by scalar, optimizing for c=1,-1
function Mdivs(X::AbstractArray, c::Real)
    c == 1 ? X : (c == -1 ? -X : X / c)
end

# Get column(s) from matrix - return whole matrix if selecting all columns
function getcol(A::AbstractMatrix, p::AbstractVector)
    length(p) == size(A, 2) ? A : A[:, p]
end

function getcol(A::AbstractMatrix, p::Int)
    A[:, p]
end

# neg_half_E_Dinvsym_E: compute -0.5 * inner product of errors weighted by
# derivative of inverse of a symmetric matrix
function neg_half_E_Dinvsym_E!(E_invX::AbstractMatrix, one2N::AbstractVector, EDE::AbstractMatrix)
    N = size(E_invX, 2)
    if N > 0
        l = size(EDE, 2)
        for j in N:-1:1
            E_invX_j = one2N == (:) ? E_invX[:, j] : E_invX[one2N, j]
            EDE[:, l] = E_invX_j .* E_invX_j .* 0.5
            l -= 1
            for i in j+1:N
                E_invX_i = one2N == (:) ? E_invX[:, i] : E_invX[one2N, i]
                EDE[:, l-N+j+i] = E_invX_i .* E_invX_j
            end
            l = l - N + j
        end
    end
    return EDE
end

# Vectorized binormal: accepts general covariance matrix, not just rho parameter
function vecbinormal(X::AbstractMatrix, Sig::AbstractMatrix, one2N::AbstractVector,
                     todo::Int)
    SigDiag = diag(Sig)
    sqrtSigDiag = sqrt.(SigDiag)

    Xhat = X ./ sqrtSigDiag'
    rho = Sig[1,2] / (sqrtSigDiag[1] * sqrtSigDiag[2])

    # Handle missing values -> large positive
    Xhat_1 = one2N == (:) ? Xhat[:, 1] : Xhat[one2N, 1]
    Xhat_2 = one2N == (:) ? Xhat[:, 2] : Xhat[one2N, 2]
    Xhat_1 = coalesce.(Xhat_1, 1e6)
    Xhat_2 = coalesce.(Xhat_2, 1e6)

    Phi = binormalGenz(Xhat_1, Xhat_2, rho)

    dPhi_dX = nothing
    dPhi_dSig = nothing

    if todo != 0
        phi = replace(normalden.(Xhat), NaN => 0.0)

        # Each X_ with the other partialled out, then renormalized to s.d. 1
        sqrt1mrho2 = sqrt(1 - rho^2)
        X_ = Xhat * [1 -rho; -rho 1] / sqrt1mrho2

        X_2 = one2N == (:) ? X_[:, 2] : X_[one2N, 2]
        X_1 = one2N == (:) ? X_[:, 1] : X_[one2N, 1]

        phi_1 = one2N == (:) ? phi[:, 1] : phi[one2N, 1]
        phi_2 = one2N == (:) ? phi[:, 2] : phi[one2N, 2]

        dPhi_dSig = phi_1 .* replace(normalden.(X_2), NaN => 0.0) / sqrt(det(Sig))

        dPhi_dX = phi .* hcat(replace(normcdf.(X_2), NaN => 1.0),
                               replace(normcdf.(X_1), NaN => 1.0)) ./ sqrtSigDiag'

        X_filled = replace(X, NaN => 0.0)
        dPhi_dSigDiag = (X_filled .* dPhi_dX .+ Sig[1,2] .* dPhi_dSig) ./ (-2 .* SigDiag')

        dPhi_dSigDiag_1 = one2N == (:) ? dPhi_dSigDiag[:, 1] : dPhi_dSigDiag[one2N, 1]
        dPhi_dSigDiag_2 = one2N == (:) ? dPhi_dSigDiag[:, 2] : dPhi_dSigDiag[one2N, 2]

        dPhi_dSig = hcat(dPhi_dSigDiag_1, dPhi_dSig, dPhi_dSigDiag_2)
    end

    return Phi, dPhi_dX, dPhi_dSig
end

# Log likelihood for continuous variables
# Sig is the assumed covariance for the full error set
# inds marks the observed variables assumed to have a joint normal distribution
function lnLContinuous(EUncens::AbstractMatrix, Omega::AbstractMatrix,
                       uncens::AbstractVector, one2N::AbstractVector, todo::Int)
    Omega_uncens = Omega[uncens, uncens]
    C = inv(cholesky(Omega_uncens).L)

    phi = quadrowsum_lnnormalden(EUncens * C', sum(log.(diag(C))))

    dphi_dE = nothing
    dphi_dSig = nothing

    if todo != 0
        invSig = C' * C
        t = -EUncens * invSig
        dphi_dE = t

        # Score w.r.t. Sigma elements
        # This is a simplified version - full implementation would need more
        dphi_dSig = nothing  # TODO: implement full derivative
    end

    return phi, dphi_dE, dphi_dSig
end

# Log likelihood for truncation - denominator correction
function lnLTrunc(pEt::AbstractMatrix, pFt::AbstractMatrix, Omega::AbstractMatrix,
                  trunc::AbstractVector, one2d_trunc::AbstractVector,
                  one2N::AbstractVector, todo::Int,
                  ghk2DrawSet::Any, ghkAnti::Bool, GHKStartTrunc::Int)
    d_trunc = length(trunc)
    Omega_trunc = Omega[trunc, trunc]

    # Use vecmultinormal for truncation likelihood
    Phi, dPhi_dEt, dPhi_dFt, dPhi_dSigt = vecmultinormal(
        pEt, pFt, Omega_trunc, d_trunc, one2d_trunc, one2N, todo,
        ghk2DrawSet, ghkAnti, GHKStartTrunc, 1)

    return Phi, dPhi_dEt, dPhi_dFt, dPhi_dSigt
end

# Multivariate normal CDF for a vector of observations
# Computes P(L_1 <= x_1 <= U_1, L_2 <= x_2 <= U_2, ...)
function vecmultinormal(E::AbstractMatrix, F::AbstractMatrix, Sig::AbstractMatrix,
                        d::Int, bounded::AbstractVector, one2N::AbstractVector,
                        todo::Int, ghk2DrawSet::Any, ghkAnti::Bool, GHKStart::Int,
                        N_perm::Int)

    dPhi_dE = nothing
    dPhi_dF = nothing
    dPhi_dSig = nothing

    if d == 1
        sqrtSig = sqrt(Sig[1,1])
        E_scaled = E[:, 1] ./ sqrtSig

        if !isempty(bounded)
            F_scaled = F[:, 1] ./ sqrtSig
            Phi = normal2(F_scaled, E_scaled)
            if todo != 0 && N_perm == 1
                dPhi_dE = replace(normalden.(E, 0, sqrtSig), NaN => 0.0) ./ Phi
                dPhi_dF = -replace(normalden.(F, 0, sqrtSig), NaN => 0.0) ./ Phi
                dPhi_dSig = (sum(dPhi_dE .* E, dims=2) .+ sum(dPhi_dF .* F, dims=2)) ./ (-2 * Sig[1,1])
            end
        else
            Phi = normcdf.(E_scaled)
            if todo != 0 && N_perm == 1
                dPhi_dE = replace(normalden.(E, 0, sqrtSig), NaN => 0.0) ./ Phi
                dPhi_dSig = dPhi_dE .* E ./ (-2 * Sig[1,1])
            end
        end

        if N_perm == 1
            return log.(Phi), dPhi_dE, dPhi_dF, dPhi_dSig
        end
        return Phi, dPhi_dE, dPhi_dF, dPhi_dSig
    end

    if d == 2
        if !isempty(bounded)
            # Bivariate case with bounds - simplified implementation
            Phi, dPhi_dE, dPhi_dSig = vecbinormal(E, Sig, one2N, todo)
        else
            Phi, dPhi_dE, dPhi_dSig = vecbinormal(E, Sig, one2N, todo)
        end
    else
        # d > 2: Use GHK simulator
        # This is where GHK.jl comes in
        if ghk2DrawSet !== nothing
            # Use GHK.jl for high-dimensional integration
            # ghk!(ghk2DrawSet, F, E, Sig, ...)
            error("GHK integration for d > 2 not yet implemented - use GHK.jl")
        else
            error("Need GHK simulator for d > 2")
        end
    end

    if N_perm == 1
        if todo != 0
            dPhi_dE = dPhi_dE ./ Phi
            dPhi_dSig = dPhi_dSig ./ Phi
            if !isempty(bounded) && dPhi_dF !== nothing
                dPhi_dF = dPhi_dF ./ Phi
            end
        end
        return log.(Phi), dPhi_dE, dPhi_dF, dPhi_dSig
    end
    return Phi, dPhi_dE, dPhi_dF, dPhi_dSig
end

# Utility: row sum with quadrature precision
function quadrowsum(X::AbstractMatrix)
    vec(sum(X, dims=2))
end

# Sparse grid quadrature sequences
# Generates nodes and weights for nested sparse grids integration with Gaussian weights
# Based on Heiss and Winschel
function SpGrGetSeq(d::Int, norm::Int)
    if d == 1
        return reshape([norm], 1, 1)
    end
    retval = [norm - d + 1  ones(Int, 1, d-1)]
    for i in norm-d:-1:1
        subseq = SpGrGetSeq(d-1, norm-i)
        retval = vcat(retval, hcat(fill(i, size(subseq, 1)), subseq))
    end
    return retval
end

# Set column of matrix to values (in-place)
function setcol!(M::AbstractMatrix, col::Int, vals::AbstractVector)
    M[:, col] = vals
    nothing
end

function setcol!(M::AbstractMatrix, col::Int, vals::AbstractMatrix)
    M[:, col] = vec(vals)
    nothing
end

# Make matrix symmetric by copying lower triangle to upper
function makesymmetric!(A::AbstractMatrix)
    n = size(A, 1)
    for j in 1:n
        for i in j+1:n
            A[j, i] = A[i, j]
        end
    end
    nothing
end

# Paste values into matrix and advance index (for score accumulation)
function pasteandadvance!(M::AbstractMatrix, k::Ref{Int}, vals::AbstractMatrix)
    ncols = size(vals, 2)
    M[:, k[]:k[]+ncols-1] = vals
    k[] += ncols
    nothing
end

# Score accumulation helper
function scoreaccum!(acc::AbstractMatrix, r::Int, weights::AbstractVector, scores::AbstractMatrix)
    if r == 1
        acc .= weights .* scores
    else
        acc .+= weights .* scores
    end
    nothing
end

# Multiply X by vector element-wise (column broadcast), optionally in-place
function xdotv(X::AbstractMatrix, v::AbstractVector)
    return X .* v
end

# Log-likelihood for censored variables
function lnLCensored!(model::CmpModel, v::Subview, todo::Int)
    uncens = v.uncens
    cens = v.cens
    d_cens = v.d_cens
    d_two_cens = v.d_two_cens
    N_perm = v.N_perm
    ThisNumCuts = v.NumCuts

    # Partial continuous variables out of the censored ones
    if v.d_uncens > 0
        Omega_uncens = v.Omega[uncens, uncens]
        Omega_uncens_cens = v.Omega[uncens, cens]
        invSig_uncens = inv(cholesky(Omega_uncens).L * cholesky(Omega_uncens).L')
        Sig_uncens_cens = Omega_uncens_cens
        beta = invSig_uncens * Sig_uncens_cens

        t = v.EUncens * beta
        pE = v.pECens .- t  # partial out errors from upper bounds
        pF = d_two_cens > 0 ? v.pF .- t : zeros(0, 0)  # partial out errors from lower bounds
        pSig = v.Omega[cens, cens] - Sig_uncens_cens' * beta  # corresponding covariance
    else
        pE = v.pECens
        pF = d_two_cens > 0 ? v.pF : zeros(0, 0)
        pSig = v.Omega[cens, cens]
    end

    # For now, simplified implementation for common cases
    # Full implementation would handle roprobit, fractional probit, etc.

    bounded = v.two_cens
    Phi, dPhi_dE, dPhi_dF, dPhi_dSig = vecmultinormal(
        pE, pF, pSig, v.dCensNonrobase, bounded, v.one2N, todo,
        model.ghk2DrawSet, model.ghkAnti, v.GHKStart, N_perm)

    if todo != 0
        # Translate scores w.r.t. partialled errors to unpartialled ones
        if v.d_uncens > 0
            # Complex derivative chain rule - simplified here
            v.dPhi_dE[v.one2N, v.cens_uncens] = dPhi_dE
            v.dPhi_dSig[v.one2N, v.SigIndsCensUncens] = dPhi_dSig
        else
            v.dPhi_dE[v.one2N, v.cens_uncens] = dPhi_dE
            v.dPhi_dSig[v.one2N, v.SigIndsCensUncens] = dPhi_dSig
        end

        if d_two_cens > 0
            if v.d_uncens > 0
                v.dPhi_dF = dPhi_dF
            else
                v.dPhi_dF[v.one2N, v.cens_uncens] = dPhi_dF
            end
            v.dPhi_dE .+= v.dPhi_dF
        end
    end

    return Phi
end

# Build total effects of random effects and coefficients at a given level
function buildtotaleffects!(model::CmpModel, l::Int)
    REs = model.REs
    RE = REs[l]
    NumREDraws = model.NumREDraws

    for r in NumREDraws[l+1]:-1:1
        if RE.HasRC
            pUT = zeros(RE.N, RE.NEq)
            if !isempty(RE.REInds)
                pUT[:, RE.REEqs] = RE.U[r] * RE.T[:, RE.REInds]
            end
        else
            pUT = RE.U[r] * RE.T
        end

        # Random coefficients
        for eq in RE.NEq:-1:1
            if RE.RCk[eq] > 0
                RCInds_eq = RE.RCInds[eq]
                pUT[:, eq] .+= vec(sum((RE.U[r] * RE.T[:, RCInds_eq]) .* RE.X[eq], dims=2))
            end
        end

        # Apply Gamma transformation if needed
        if model.HasGamma
            for eq in length(RE.GammaEqs):-1:1
                _eq = RE.GammaEqs[eq]
                RE.TotalEffect[r, _eq] = pUT * RE.invGamma[:, eq]
            end
        else
            for eq in length(RE.GammaEqs):-1:1
                _eq = RE.GammaEqs[eq]
                RE.TotalEffect[r, _eq] = pUT[:, eq:eq]
            end
        end
    end
    nothing
end

# Main likelihood evaluator
# Translated from cmp.mata lf1() function (lines 1190-1695)
function lf1!(model::CmpModel, todo::Int, b::AbstractVector)
    d = model.d
    L = model.L
    REs = model.REs
    base = model.base

    # Initialize output
    N = base.N
    lnf = zeros(N)
    S = todo != 0 ? model.S0 : zeros(0, 0)

    # === Extract parameters ===
    # Linear predictors (theta) for each equation are pre-computed by Stata's mleval
    # and passed to Julia via the cmp_lf1.ado wrapper. They are stored in:
    #   model.REs[end].theta[eq] for each equation eq

    # Auxiliary parameters (gamma, cuts, sig, rho) are extracted by the Stata wrapper
    # using mleval and stored in _cmp_aux_params vector. Read them with incrementing index.
    aux_params = isdefined(Main, :_cmp_aux_params) ? Main._cmp_aux_params : Float64[]
    aux_idx = Ref(1)  # Use Ref for mutable counter

    # Helper function to read next auxiliary parameter
    function next_aux()
        val = aux_params[aux_idx[]]
        aux_idx[] += 1
        return val
    end

    # Gamma parameters (if any simultaneous equations)
    Gamma = model.Gamma
    GammaInd = model.GammaInd
    indicators = model.indicators
    for j in 1:size(GammaInd, 1)
        if !isempty(aux_params)
            Gamma[Int(GammaInd[j, 1]), Int(GammaInd[j, 2])] = -next_aux()
        end
    end

    # Cut parameters for ordered probit
    vNumCuts = model.vNumCuts
    cuts = model.cuts
    trunceqs = model.trunceqs
    Lt = model.Lt
    Ut = model.Ut
    for eq1 in 1:d
        if !isempty(vNumCuts) && eq1 <= length(vNumCuts) && vNumCuts[eq1] > 0
            for cut in 2:vNumCuts[eq1]+1
                if !isempty(aux_params)
                    cuts[cut, eq1] = next_aux()
                    # Check truncation bounds (translated from Mata lines 1215-1217)
                    if !isempty(trunceqs) && trunceqs[eq1] != 0
                        if !isempty(Lt) && !isempty(Lt[eq1]) && !isempty(Ut) && !isempty(Ut[eq1])
                            # Validity check for truncation
                            Lt_eq = Lt[eq1]
                            Ut_eq = Ut[eq1]
                            if any((indicators[:, eq1] .!= 0) .&
                                   ((Lt_eq .< Inf .&& cuts[cut, eq1] .< Lt_eq) .| (cuts[cut, eq1] .> Ut_eq)))
                                return (fill(NaN, N), S)
                            end
                        end
                    end
                end
            end
        end
    end

    # === Random effects parameters (sig, rho) for each level ===
    SigXform = model.SigXform

    for l in 1:L
        RE = REs[l]
        RE.sig = Float64[]
        RE.rho = Float64[]

        lnsigWithin = 0.0
        lnsigAcross = 0.0

        # Exchangeable across levels?
        if RE.covAcross == 0 && !isempty(aux_params)
            lnsigWithin = lnsigAcross = next_aux()
        end

        for eq1 in 1:RE.NEq
            # Exchangeable within but not across?
            if !isempty(RE.covWithin) && RE.Eqs[eq1] <= length(RE.covWithin) &&
               RE.covWithin[RE.Eqs[eq1]] == 0 && RE.covAcross != 0
                lnsigWithin = lnsigAcross
            end

            for c1 in 1:RE.NEff[eq1]
                if !isempty(RE.FixedSigs) && RE.Eqs[eq1] <= length(RE.FixedSigs) &&
                   (ismissing(RE.FixedSigs[RE.Eqs[eq1]]) || isnan(RE.FixedSigs[RE.Eqs[eq1]]))
                    # Need to read a new sig parameter?
                    if !isempty(RE.covWithin) && RE.Eqs[eq1] <= length(RE.covWithin) &&
                       RE.covWithin[RE.Eqs[eq1]] != 0 && RE.covAcross != 0 && !isempty(aux_params)
                        lnsigWithin = next_aux()
                    end
                    if SigXform == 0 && lnsigWithin == 0
                        return (fill(NaN, N), S)  # Invalid parameters
                    end
                    sig_val = SigXform != 0 ? exp(lnsigWithin) : lnsigWithin
                    push!(RE.sig, sig_val)
                elseif !isempty(RE.FixedSigs) && RE.Eqs[eq1] <= length(RE.FixedSigs)
                    push!(RE.sig, RE.FixedSigs[RE.Eqs[eq1]])
                else
                    # Default sig value if not fixed
                    sig_val = SigXform != 0 ? exp(lnsigWithin) : lnsigWithin
                    push!(RE.sig, sig_val)
                end
            end
        end

        # Correlation parameters
        atanhrhoWithin = 0.0
        atanhrhoAcross = 0.0

        # Exchangeable across?
        if RE.covAcross == 0 && RE.d > 1 && !isempty(aux_params)
            atanhrhoAcross = next_aux()
        end

        for eq1 in 1:RE.NEq
            if !isempty(RE.covWithin) && RE.Eqs[eq1] <= length(RE.covWithin)
                if RE.covWithin[RE.Eqs[eq1]] == 2  # independent?
                    atanhrhoWithin = 0.0
                elseif RE.covWithin[RE.Eqs[eq1]] == 0 && RE.NEff[eq1] > 1 && !isempty(aux_params)
                    # Exchangeable within
                    atanhrhoWithin = next_aux()
                end
            end

            for c1 in 1:RE.NEff[eq1]
                for c2 in c1+1:RE.NEff[eq1]
                    if !isempty(RE.covWithin) && RE.Eqs[eq1] <= length(RE.covWithin) &&
                       RE.covWithin[RE.Eqs[eq1]] == 1 && !isempty(aux_params)
                        # Unstructured within
                        atanhrhoWithin = next_aux()
                    end
                    push!(RE.rho, atanhrhoWithin)
                end

                for eq2 in eq1+1:RE.NEq
                    for c2 in 1:RE.NEff[eq2]
                        if !isempty(RE.FixedRhos) &&
                           RE.Eqs[eq2] <= size(RE.FixedRhos, 1) && RE.Eqs[eq1] <= size(RE.FixedRhos, 2) &&
                           (ismissing(RE.FixedRhos[RE.Eqs[eq2], RE.Eqs[eq1]]) ||
                            isnan(RE.FixedRhos[RE.Eqs[eq2], RE.Eqs[eq1]]))
                            if RE.covAcross == 1 && !isempty(aux_params)
                                # Unstructured across
                                atanhrhoAcross = next_aux()
                            end
                            push!(RE.rho, atanhrhoAcross)
                        elseif !isempty(RE.FixedRhos) &&
                               RE.Eqs[eq2] <= size(RE.FixedRhos, 1) && RE.Eqs[eq1] <= size(RE.FixedRhos, 2)
                            push!(RE.rho, RE.FixedRhos[RE.Eqs[eq2], RE.Eqs[eq1]])
                        else
                            push!(RE.rho, atanhrhoAcross)
                        end
                    end
                end
            end
        end
    end

    # === Handle Gamma transformation if present ===
    invGamma = model.HasGamma ? inv(lu(Gamma)) : I(d)
    if model.HasGamma && any(isnan.(invGamma))
        return (fill(NaN, N), S)  # Gamma not invertible
    end

    # === Build covariance matrices T, Sig for each level ===
    for l in 1:L
        RE = REs[l]
        if RE.d == 1
            RE.T = fill(RE.sig[1], 1, 1)
            RE.Sig = RE.T .^ 2
        else
            k = 0
            for j in 1:RE.d
                for i in j+1:RE.d
                    k += 1
                    if SigXform != 0
                        if RE.rho[k] > 100
                            RE.Rho[i, j] = 1.0
                        elseif RE.rho[k] < -100
                            RE.Rho[i, j] = -1.0
                        else
                            RE.Rho[i, j] = tanh(RE.rho[k])
                        end
                    else
                        RE.Rho[i, j] = RE.rho[k]
                    end
                end
            end
            makesymmetric!(RE.Rho)

            # T = cholesky(Rho)' .* sig
            try
                RE.T = cholesky(RE.Rho).L' .* RE.sig'
            catch
                return (fill(NaN, N), S)  # Rho not positive definite
            end

            # Sig = (sig * sig') .* Rho
            RE.Sig = (RE.sig * RE.sig') .* RE.Rho
        end

        if todo != 0
            RE.D = dSigdsigrhos(SigXform, RE.sig, RE.Sig, RE.rho, RE.Rho) * RE.dSigdParams
        end

        if model.HasGamma
            RE.invGamma = invGamma[RE.Eqs, RE.GammaEqs]
        end

        if l < L
            buildtotaleffects!(model, l)
        end
    end

    # === Compute Omega for each subview ===
    pOmega = model.HasGamma ? invGamma' * base.Sig * invGamma : base.Sig

    for v in model.subviews
        v.Omega = v.QE' * pOmega * v.QE

        if todo != 0
            if model.HasGamma
                v.QEinvGamma = v.QE' * invGamma
            else
                v.QEinvGamma = v.QE'
            end
        end
    end

    # === Main likelihood computation ===
    base.plnL = lnf

    # Simplified main loop - full implementation would handle:
    # - Multiple draws for random effects
    # - Adaptive quadrature
    # - Multiple subviews
    # - Mprobit error transformations

    for v in model.subviews
        # Compute errors for this subview
        # (would loop over equations and compute E = y - theta for each type)

        # Compute likelihood
        if v.d_cens > 0
            lnL = lnLCensored!(model, v, todo)
            if v.d_uncens > 0
                phi, dphi_dE, dphi_dSig = lnLContinuous(v.EUncens, v.Omega, v.uncens, v.one2N, todo)
                lnL = lnL .+ phi
            end
        else
            lnL, dphi_dE, dphi_dSig = lnLContinuous(v.EUncens, v.Omega, v.uncens, v.one2N, todo)
        end

        # Truncation correction
        if v.d_trunc > 0
            Phi_trunc, dPhi_dEt, dPhi_dFt, dPhi_dSigt = lnLTrunc(
                v.pEt, v.pFt, v.Omega, v.trunc, v.one2d_trunc, v.one2N, todo,
                model.ghk2DrawSet, model.ghkAnti, v.GHKStartTrunc)
            lnL = lnL .- Phi_trunc
        end

        lnf[v.SubsampleInds] = lnL

        # Score computation would go here when todo==1
    end

    return (lnf, S)
end

function gf1!(model::CmpModel, todo::Int, b::AbstractVector)
    # Group evaluator - calls lf1! then aggregates to group level
    # For survey data, this computes group-level likelihood

    lnf, S = lf1!(model, todo, b)

    # Aggregate to group level using panel structure
    # (Implementation depends on group structure)

    return (lnf, S)
end

# Helper: get indices for vech of submatrix
function vSigInds(inds::AbstractVector, d::Int)
    if isempty(inds)
        return Int[]
    end
    result = Int[]
    for (idx, i) in enumerate(inds)
        for j in 1:idx
            # Convert to vech index: for d×d matrix, vech index of (i,j) where i>=j
            # is sum(d-k+1 for k=1 to j-1) + (i-j+1) = j*(2d-j+1)/2 - d + i
            ii, jj = max(inds[idx], inds[j]), min(inds[idx], inds[j])
            vech_idx = jj * (2*d - jj + 1) ÷ 2 - d + ii
            push!(result, vech_idx)
        end
    end
    return result
end

# Initialize model before estimation - simplified version that reads data from Stata
# This is called from cmp.ado without parameters; data is read using sf_get_var()
function cmp_init!(model::CmpModel)
    d = model.d

    # sf_get_var is provided by julia.ado in Main module
    sf_get_var = Main.sf_get_var

    # Read indicator matrix from Stata
    # The indicator variable names are stored in model.indVars
    N = 0
    indicators = zeros(0, d)
    if !isempty(model.indVars)
        for (eq, varname) in enumerate(model.indVars)
            if !isempty(varname)
                col = sf_get_var(varname)
                if N == 0
                    N = length(col)
                    indicators = zeros(N, d)
                end
                indicators[:, eq] = col
            end
        end
    end

    # Read dependent variables and bounds
    y = [isempty(model.yLVars) || eq > length(model.yLVars) || isempty(model.yLVars[eq]) ?
         zeros(0, 0) : reshape(sf_get_var(model.yLVars[eq]), :, 1) for eq in 1:d]
    Lt = [isempty(model.LtVars) || eq > length(model.LtVars) || isempty(model.LtVars[eq]) ?
          zeros(0, 0) : reshape(sf_get_var(model.LtVars[eq]), :, 1) for eq in 1:d]
    Ut = [isempty(model.UtVars) || eq > length(model.UtVars) || isempty(model.UtVars[eq]) ?
          zeros(0, 0) : reshape(sf_get_var(model.UtVars[eq]), :, 1) for eq in 1:d]

    # Use the full initialization
    return cmp_init_full!(model, indicators, y, Lt, Ut, y)  # yL = y for now
end

# Initialize model before estimation - full version with explicit parameters
# Translated from cmp.mata cmp_init() function (lines 1734-2264)
function cmp_init_full!(model::CmpModel, indicators::AbstractMatrix,
                   y::Vector, Lt::Vector, Ut::Vector, yL::Vector;
                   id::Vector{<:AbstractMatrix}=Matrix{Float64}[],
                   Eqs::AbstractMatrix=ones(1, 1),
                   NumEff::AbstractMatrix=ones(1, 1),
                   GammaId::AbstractMatrix=Matrix{Float64}(I, model.d, model.d),
                   GammaIndByEq::Vector{Vector{Int}}=Vector{Int}[],
                   vNumCuts::Vector{Int}=Int[],
                   trunceqs::Vector{Int}=Int[],
                   intregeqs::Vector{Int}=Int[])

    d = model.d
    L = model.L

    # Initialize random effects structures
    model.REs = [RE() for _ in 1:L]
    model.base = model.REs[L]
    base = model.base

    one2d = 1:d
    d2 = d * (d + 1) ÷ 2

    # Initialize Gamma matrix (I - Gamma for simultaneous equations)
    model.Gamma = Matrix{Float64}(I, d, d)

    # Initialize cuts for ordered probit
    model.MaxCuts = isempty(vNumCuts) ? 0 : maximum(vNumCuts)
    model.cuts = fill(1.701e38, model.MaxCuts + 2, d)  # maxfloat
    model.cuts[1, :] .= -1.701e38  # minfloat
    model.vNumCuts = isempty(vNumCuts) ? zeros(Int, d) : vNumCuts
    model.trunceqs = isempty(trunceqs) ? zeros(Int, d) : trunceqs
    model.intregeqs = isempty(intregeqs) ? zeros(Int, d) : intregeqs

    # Store data references
    model.y = y
    model.Lt = Lt
    model.Ut = Ut
    model.yL = yL

    # Initialize HasGamma flag
    model.HasGamma = !isempty(GammaIndByEq) && any(length.(GammaIndByEq) .> 0)
    model.GammaIndByEq = GammaIndByEq

    if model.HasGamma
        model.Idd = Matrix{Float64}(I, d*d, d*d)
        model.vLd = vec(sum(Lmatrix(d) .* (1:d*d)', dims=2))
        Kd = kron(I(d), I(d))  # Commutation matrix approximation
        model.vKd = vec(sum(Kd .* (1:d*d), dims=1))
    end

    model.ThisDraw = ones(Int, L)

    # Setup each random effects level
    for l in L:-1:1
        RE = model.REs[l]
        RE.Eqs = findall(Eqs[:, l] .!= 0)
        RE.NEq = length(RE.Eqs)
        RE.NEff = [Int(NumEff[l, eq]) for eq in RE.Eqs]
        RE.GammaEqs = model.HasGamma ? findall((GammaId * Eqs[:, l]) .!= 0) : RE.Eqs
        RE.d = sum(RE.NEff)
        RE.one2d = collect(1:RE.d)
        RE.theta = [zeros(0, 0) for _ in 1:d]
        RE.Rho = Matrix{Float64}(I, RE.d, RE.d)
        RE.d2 = RE.d * (RE.d + 1) ÷ 2

        # Covariance structure parameters (default to unstructured)
        RE.covAcross = 1  # 0=exchangeable, 1=unstructured, 2=independent
        RE.covWithin = ones(Int, d)
        RE.FixedSigs = fill(NaN, d)
        RE.FixedRhos = fill(NaN, d, d)
    end

    # Setup base level data dimensions
    model.indicators = indicators
    base.N = size(indicators, 1)
    base.one2N = base.N < 10000 ? collect(1:base.N) : Int[]
    base.J_N_1_0 = zeros(base.N)
    base.J_N_0_0 = zeros(base.N, 0)
    model.Theta = zeros(base.N, d)

    # Build dSigdParams for each level (derivative structure)
    for l in L:-1:1
        RE = model.REs[l]
        if model._todo != 0 || model.HasGamma
            # Simplified: assume unstructured covariance
            RE.NSigParams = RE.d2
            RE.dSigdParams = Matrix{Float64}(I, RE.d2, RE.d2)
        end
    end

    # Setup score structure if computing gradients
    if model._todo != 0
        model.Scores_ = Scores()
        model.G = zeros(Int, d)

        if model.HasGamma
            for m in 1:d
                if !isempty(GammaIndByEq) && m <= length(GammaIndByEq)
                    model.G[m] = length(GammaIndByEq[m])
                end
            end
        end

        model.Scores_.ThetaScores = collect(1:d)
        model.NumCuts = sum(model.vNumCuts)
        if model.NumCuts > 0
            model.Scores_.CutScores = collect(d+1:d+model.NumCuts)
        end

        cols = d + 1 + model.NumCuts
        model.Scores_.SigScores = [zeros(0, 0) for _ in 1:L]
        for l in 1:L
            RE = model.REs[l]
            if RE.NSigParams > 0
                model.Scores_.SigScores[l] = zeros(1, RE.NSigParams)
                # Store column indices
                cols += RE.NSigParams
            end
        end

        model.S0 = zeros(base.N, cols - 1)
    end

    # Setup random effects draws for levels below top
    for l in L-1:-1:1
        RE = model.REs[l]
        if l <= length(id) && !isempty(id[l])
            RE.id = id[l]
            RE.N = Int(maximum(RE.id))
        else
            RE.N = base.N
            RE.id = reshape(collect(1.0:base.N), base.N, 1)
        end

        RE.R = model.NumREDraws[l+1]
        RE.one2N = RE.N < 10000 ? collect(1:RE.N) : Int[]
        RE.J_N_1_0 = zeros(RE.N)
        RE.one2R = collect(1:RE.R)

        # Initialize draw matrices
        RE.U = [zeros(base.N, RE.d) for _ in 1:RE.R]
        RE.TotalEffect = [zeros(0, 0) for _ in 1:RE.R, _ in 1:d]

        # Setup panel structure
        RE.IDRanges = panelsetup(vec(RE.id))
        RE.IDRangeLengths = RE.IDRanges[:, 2] .- RE.IDRanges[:, 1] .+ 1

        # Generate random draws (simplified - random draws)
        NDraws = RE.R ÷ model.REAnti
        U = randn(RE.N * NDraws, RE.d)

        # Distribute draws to groups
        S = repeat(collect(1:NDraws), inner=RE.N)
        for r in NDraws:-1:1
            mask = S .== r
            if sum(mask) > 0
                RE.U[r] = U[mask, :]
            end
            if model.REAnti == 2 && r + RE.R ÷ 2 <= RE.R
                RE.U[r + RE.R ÷ 2] = -RE.U[r]
            end
        end

        RE.lnNumREDraws = log(RE.R)
        RE.lnLlimits = log(floatmin()) + 1
        RE.lnLByDraw = zeros(RE.N, RE.R)
    end

    # Build subviews - one for each unique indicator combination
    model.subviews = Subview[]
    remaining = collect(1:base.N)
    d_cens = 0
    d_ghk = 0
    ghk_nobs = 0

    while !isempty(remaining)
        t = remaining[1]
        v = Subview()

        # Find rows with same indicator pattern
        TheseInds = indicators[t, :]
        v.TheseInds = TheseInds
        v.subsample = vec(all(indicators .== TheseInds', dims=2))
        v.SubsampleInds = findall(v.subsample)
        remaining = setdiff(remaining, v.SubsampleInds)

        v.N = length(v.SubsampleInds)
        v.one2N = v.N < 10000 ? collect(1:v.N) : Int[]

        # Classify equations by type
        v.uncens = findall(TheseInds .== cmp_cont)
        v.d_uncens = length(v.uncens)

        v.oprobit = findall(TheseInds .== cmp_oprobit)
        v.d_oprobit = length(v.oprobit)

        v.trunc = findall(model.trunceqs .!= 0)
        v.d_trunc = length(v.trunc)
        v.one2d_trunc = collect(1:v.d_trunc)

        # Censored equations (excluding mprobit base)
        v.cens = findall((TheseInds .> cmp_cont) .& (TheseInds .< Inf) .&
                         ((TheseInds .< mprobit_ind_base) .| (TheseInds .>= roprobit_ind_base)))
        v.d_cens = length(v.cens)
        d_cens = max(d_cens, v.d_cens)

        # Equations that are doubly censored
        v.two_cens = findall((TheseInds .== cmp_oprobit) .| (TheseInds .== cmp_int) .|
                             (((TheseInds .== cmp_left) .| (TheseInds .== cmp_right) .|
                               (TheseInds .== cmp_probit) .| (TheseInds .== cmp_probity1)) .&
                              (model.trunceqs .!= 0)))
        v.d_two_cens = length(v.two_cens)

        # Transformation matrix QE
        signs = 2 .* ((TheseInds .== cmp_right) .| (TheseInds .== cmp_probity1) .|
                      (TheseInds .== cmp_frac)) .- 1
        v.QE = Diagonal(signs)

        # Subsample data
        v.theta = [zeros(0, 0) for _ in 1:d]
        v.y = [isempty(y[i]) ? zeros(0, 0) : y[i][v.SubsampleInds, :] for i in 1:d]
        v.Lt = [isempty(Lt[i]) ? zeros(0, 0) : Lt[i][v.SubsampleInds, :] for i in 1:d]
        v.Ut = [isempty(Ut[i]) ? zeros(0, 0) : Ut[i][v.SubsampleInds, :] for i in 1:d]
        v.yL = [isempty(yL[i]) ? zeros(0, 0) : yL[i][v.SubsampleInds, :] for i in 1:d]

        # GHK setup for high-dimensional integration
        if v.d_cens > 2
            v.GHKStart = ghk_nobs + 1
            ghk_nobs += v.N
        end
        if v.d_trunc > 2
            v.GHKStartTrunc = ghk_nobs + 1
            ghk_nobs += v.N
        end
        d_ghk = max(d_ghk, v.d_trunc, v.d_cens)

        # Allocate error matrices
        if v.d_uncens > 0
            v.EUncens = zeros(v.N, v.d_uncens)
        end
        if v.d_cens > 0
            v.pECens = zeros(v.N, v.d_cens)
        end
        if v.d_two_cens > 0 || v.d_trunc > 0
            v.pF = zeros(v.N, v.d_cens)
            if v.d_trunc > 0
                v.pEt = zeros(v.N, v.d_trunc)
                v.pFt = zeros(v.N, v.d_trunc)
            end
        end

        # NotBaseEq: equations that are not mprobit base alternatives
        v.NotBaseEq = (TheseInds .< mprobit_ind_base) .| (TheseInds .>= roprobit_ind_base)

        # Score-related setup
        if model._todo != 0
            v.cens_uncens = vcat(v.cens, v.uncens)
            v.SigIndsUncens = vSigInds(v.uncens, d)
            v.SigIndsCensUncens = vSigInds(v.cens_uncens, d)
            if v.d_trunc > 0
                v.SigIndsTrunc = vSigInds(v.trunc, d)
            end

            v.J_d_uncens_d_cens_0 = zeros(v.d_uncens, v.d_cens)
            v.J_d_cens_d_0 = zeros(v.d_cens, d)
            v.J_d2_cens_d2_0 = zeros(v.d_cens * (v.d_cens + 1) ÷ 2, d2)
            v.J_N_1_0 = zeros(v.N, 1)

            if v.d_uncens > 0
                v.dphi_dE = zeros(v.N, d)
                v.dphi_dSig = zeros(v.N, d2)
                v.EDE = zeros(v.N, v.d_uncens * (v.d_uncens + 1) ÷ 2)
            else
                v.dPhi_dE = zeros(v.N, d)
            end

            if v.d_two_cens > 0 || v.d_trunc > 0
                v.dPhi_dpF = zeros(v.N, v.d_cens)
                if v.d_uncens == 0
                    v.dPhi_dF = zeros(v.N, d)
                end
                if v.d_trunc > 0
                    v.dPhi_dEt = zeros(v.N, d)
                    v.dPhi_dSigt = zeros(v.N, d2)
                end
            end

            if v.d_cens > 0 && v.d_uncens == 0
                v.dPhi_dSig = zeros(v.N, d2)
            end
            if v.d_cens > 0 && v.d_uncens > 0
                v.dPhi_dpE_dSig = zeros(v.N, d2)
                v._dPhi_dpE_dSig = zeros(v.N, (v.d_cens + v.d_uncens) * (v.d_cens + v.d_uncens + 1) ÷ 2)
            end

            if model.NumCuts > 0
                v.dPhi_dcuts = zeros(v.N, model.NumCuts)
            end

            v.QSig = QE2QSig(Matrix(v.QE))'
            v.dSig_dLTSig = Matrix(Dmatrix(v.d_cens + v.d_uncens))

            # ID mapping for multi-level models
            v.id = [zeros(0, 0) for _ in 1:L-1]
            for l in 1:L-1
                if l <= length(id) && !isempty(id[l])
                    v.id[l] = id[l][v.SubsampleInds, :]
                end
            end
        end

        # Initialize NFracCombs
        v.d_frac = 0
        v.NFracCombs = 1
        v.N_perm = 1

        push!(model.subviews, v)
    end

    model.d_cens = d_cens

    # Setup GHK draws if needed
    if ghk_nobs > 0 && model.ghkDraws > 0
        # Would initialize GHK draw set here using GHK.jl
        # model.ghk2DrawSet = GHKDraws(ghk_nobs, model.ghkDraws, d_ghk, ...)
    end

    return 0  # Success
end

# Helper: normalden with mean and standard deviation
function normalden(x::Real, mean::Real, sd::Real)
    pdf(Normal(mean, sd), x)
end

function normalden(x::Real)
    pdf(Normal(), x)
end

function normalden(x::AbstractArray)
    pdf.(Normal(), x)
end

function normalden(x::AbstractArray, mean::Real, sd::Real)
    pdf.(Normal(mean, sd), x)
end

end # module CMP
