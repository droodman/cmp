# cmp.jl - Julia backend for cmp Stata package
# Copyright (C) 2007-24 David Roodman
# Licensed under GNU GPL v3

module CMP

using LinearAlgebra, Distributions, SparseArrays, StatsFuns, IrrationalConstants, LoopVectorization, GHK, stataplugininterface

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
    ThetaScores::Vector{Int16}    # cols of master score matrix for theta params
    CutScores::Vector{Int16}      # cols for cut parameters
    TScores::VecMat             # for each level
    SigScores::VecMat           # only at top level
    GammaScores::VecMat

    Scores() = new(Int16[], Int16[], Mat[], Mat[], Mat[])
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

    next::Union{Subview,Nothing}

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
    IDRanges::Vector{UnitRange{Int64}}
    IDRangeLengths::Vector{Int}
    IDRangesGroup::Vector{UnitRange{Int64}}
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
    QuadXAdapt::Dict{Any,Any}
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
        re.IDRanges = UnitRange{Int64}[]
        re.IDRangeLengths = Int[]
        re.IDRangesGroup = UnitRange{Int64}[]
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
        re.QuadXAdapt = Dict{Any,Any}()
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
    yVars::Vector{String}

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
        m.yVars = String[]
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
function vech!(dest::AbstractVector{T}, X::AbstractMatrix{T}) where T
	d = nrows(X)
	k = 0
	j₀, j₁ = 1, d
	@inbounds for i ∈ 1:d
		for j ∈ j₀:j₁
			k += 1
			dest[k] = X[j]
		end
		j₀ += d+1
		j₁ += d
	end
	dest
end
function vech(X::AbstractMatrix{T}) where T
	d = nrows(X)
	dest = Vector{T}(undef, Int(d*(d+1)÷2))
	vech!(dest, X)
end

# Inverse of vech: reconstruct symmetric matrix from vectorized lower triangle
function invvech!(dest::AbstractMatrix{T}, X::AbstractVector{T}, d::Integer) where T
	k = 0
	j₀, j₁ = 1, d
	@inbounds for i ∈ 1:d
		for j ∈ j₀:j₁
			k += 1
			dest[j] = X[k]
		end
		j₀ += d+1
		j₁ += d
	end
	dest
end
function invvech(X::AbstractVector{T}, d::Integer) where T
	dest = zeros(T, d, d)
	invvech!(dest, X, d)
end

# Panel setup: returns matrix of start/end indices for each group
function panelsetup(X::AbstractArray{S} where S, colinds::AbstractVector{T} where T<:Integer)
  N = nrows(X)
	iszero(N) && return(Vector{UnitRange{Int64}}(undef,0))
  info = Vector{UnitRange{Int64}}(undef, N)
  lo = p = 1
  @inbounds for hi ∈ 2:N
    for j ∈ colinds
      if X[hi,j] ≠ X[lo,j]
        info[p] = lo:hi-1
        lo = hi
        p += 1
        break
      end
  	end
  end
  info[p] = lo:N
  resize!(info, p)
  info
end

# Single-argument version for a simple vector (assumes already sorted)
function panelsetup(X::AbstractVector{S}) where S
  N = length(X)
  iszero(N) && return(Vector{UnitRange{Int64}}(undef,0))
  info = Vector{UnitRange{Int64}}(undef, N)
  lo = p = 1
  @inbounds for hi ∈ 2:N
    if X[hi] ≠ X[lo]
      info[p] = lo:hi-1
      lo = hi
      p += 1
    end
  end
  info[p] = lo:N
  resize!(info, p)
  info
end

function panelsum!(dest::AbstractVecOrMat, X::AbstractVecOrMat, info::AbstractVector{UnitRange{T}} where T<:Integer)
	iszero(length(dest)) && return
	J = CartesianIndices(axes(X)[2:end])
	eachindexJ = eachindex(J)
	@inbounds for g in eachindex(info)
		f, l = first(info[g]), last(info[g])
		fl = f+1:l
		if f<l
			for j ∈ eachindexJ
				Jj = J[j]
				tmp = X[f,Jj]
				@tturbo warn_check_args=false for i ∈ fl
					tmp += X[i,Jj]
				end
				dest[g,Jj] = tmp
			end
		else
			@simd for j ∈ eachindexJ
				dest[g,J[j]] = X[f,J[j]]
			end
		end
	end
end
function panelsum!(dest::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}, wt::AbstractVector{T}, info::AbstractVector{UnitRange{S}} where S<:Integer) where T
	iszero(length(dest)) && return
	iszero(length(wt)) &&return panelsum!(dest, X, info)
	if iszero(length(info)) || nrows(info)==nrows(X)
		dest .= X .* wt
		return
	end
	J = CartesianIndices(axes(X)[2:end])
	eachindexJ = eachindex(J)
	@inbounds for g in eachindex(info)
		f, l = first(info[g]), last(info[g])
    fl = f+1:l
		_wt = wt[f]
		if f<l
			for j ∈ eachindexJ
				Jj = J[j]
				tmp = X[f,Jj] * _wt
				@tturbo warn_check_args=false for i ∈ fl
					tmp += X[i,Jj] * wt[i]
				end
				dest[g,Jj] = tmp
			end
		else
			for j ∈ eachindexJ
				dest[g,J[j]] = X[f,J[j]] * _wt
			end
		end
	end
end
function panelsum(X::AbstractVecOrMat{T}, wt::AbstractVector{T}, info::AbstractVector{UnitRange{S}} where S<:Integer) where T
	dest = isa(X, AbstractVector{T}) ? Vector{T}(undef, iszero(length(info)) ? nrows(X) : length(info)           ) :
		                                 Matrix{T}(undef, iszero(length(info)) ? nrows(X) : length(info), ncols(X))
	if iszero(length(info)) || length(info)==length(X)
		dest .= X .* wt
	else
		panelsum!(dest, X, wt, info)
	end
	dest
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
    sparse(1:d2, [i * d + j for i ∈ 0:d-1 for j ∈ i+1:d], ones(Int, d2))
end

# Duplication matrix: maps vech(A) to vec(A) for symmetric A
# Sparse version from old/cmp.jl for efficiency
function Dmatrix(d::Int)
    d2 = d^2
    sparse(1:d2, [(c = min(i, j); max(i, j) + (2d - c - 1) * c ÷ 2 + 1) for i ∈ 0:d-1 for j ∈ 0:d-1], ones(Int, d2))
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

    for k ∈ 1:_d
        t = zeros(_d, _d)
        t2 = SigXform != 0 ? Sig[k, :] : (_d > 1 ? Rho[k, :] .* sig : sig)
        t[k, :] = t2
        t[:, k] = t[:, k] + t2
        D[:, k] = vech(t)
    end

    if _d > 1
        k = _d + 1
        for j ∈ 1:_d
            for i ∈ j+1:_d
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
        for i ∈ 1:12
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
        bs = (h .- k) .^ 2
        c = (4 - hk) / 8
        d = (12 - hk) / 16
        asr = -(bs ./ ass .+ hk) / 2

        @. bvn = a * exp(asr) * (1 - c * (bs - ass) * (1 - d * bs / 5) / 3 + c * d * ass^2 / 5)

        if -hk[1] < 100  # Approximate check
            b = sqrt(bs)
            @. bvn -= exp(-hk / 2) * sqrt(2π) * normcdf(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
        end

        a = a / 2
        for i ∈ 1:12
            for is ∈ [-1, 1]
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
            @. bvn[h<0] += normcdf(k[h<0]) - normcdf(h[h<0])
            @. bvn[h>=0] += normcdf(-h[h>=0]) - normcdf(-k[h>=0])
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
        for j ∈ N:-1:1
            E_invX_j = one2N == (:) ? E_invX[:, j] : E_invX[one2N, j]
            EDE[:, l] = E_invX_j .* E_invX_j .* 0.5
            l -= 1
            for i ∈ j+1:N
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
    rho = Sig[1, 2] / (sqrtSigDiag[1] * sqrtSigDiag[2])

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
        dPhi_dSigDiag = (X_filled .* dPhi_dX .+ Sig[1, 2] .* dPhi_dSig) ./ (-2 .* SigDiag')

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
        sqrtSig = sqrt(Sig[1, 1])
        E_scaled = E[:, 1] ./ sqrtSig

        if !isempty(bounded)
            F_scaled = F[:, 1] ./ sqrtSig
            Phi = normal2(F_scaled, E_scaled)
            if todo != 0 && N_perm == 1
                dPhi_dE = replace(normalden.(E, 0, sqrtSig), NaN => 0.0) ./ Phi
                dPhi_dF = -replace(normalden.(F, 0, sqrtSig), NaN => 0.0) ./ Phi
                dPhi_dSig = (sum(dPhi_dE .* E, dims=2) .+ sum(dPhi_dF .* F, dims=2)) ./ (-2 * Sig[1, 1])
            end
        else
            Phi = normcdf.(E_scaled)
            if todo != 0 && N_perm == 1
                dPhi_dE = replace(normalden.(E, 0, sqrtSig), NaN => 0.0) ./ Phi
                dPhi_dSig = dPhi_dE .* E ./ (-2 * Sig[1, 1])
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
    retval = [norm - d + 1 ones(Int, 1, d - 1)]
    for i ∈ norm-d:-1:1
        subseq = SpGrGetSeq(d - 1, norm - i)
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
    for j ∈ 1:n
        for i ∈ j+1:n
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

    for r ∈ NumREDraws[l+1]:-1:1
        if RE.HasRC
            pUT = zeros(RE.N, RE.NEq)
            if !isempty(RE.REInds)
                pUT[:, RE.REEqs] = RE.U[r] * RE.T[:, RE.REInds]
            end
        else
            pUT = RE.U[r] * RE.T
        end

        # Random coefficients
        for eq ∈ RE.NEq:-1:1
            if RE.RCk[eq] > 0
                RCInds_eq = RE.RCInds[eq]
                pUT[:, eq] .+= vec(sum((RE.U[r] * RE.T[:, RCInds_eq]) .* RE.X[eq], dims=2))
            end
        end

        # Apply Gamma transformation if needed
        if model.HasGamma
            for eq ∈ length(RE.GammaEqs):-1:1
                _eq = RE.GammaEqs[eq]
                RE.TotalEffect[r, _eq] = pUT * RE.invGamma[:, eq]
            end
        else
            for eq ∈ length(RE.GammaEqs):-1:1
                _eq = RE.GammaEqs[eq]
                RE.TotalEffect[r, _eq] = pUT[:, eq:eq]
            end
        end
    end
    nothing
end

# Build X-U products for score computation (Mata lines 1143-1172)
function buildXU!(model::CmpModel, l::Int)
    REs = model.REs
    RE = REs[l]
    base = model.base

    if RE.HasRC
        # Pre-compute X-U products for random coefficients
        for r ∈ RE.R:-1:1
            k = 0
            e = 0
            for eq1 ∈ 1:RE.NEq
                for c ∈ 1:(RE.RCk[eq1] + (eq1 ∈ RE.REEqs ? 1 : 0))
                    e += 1
                    k += 1
                    Ue = RE.U[r][:, e]

                    if c <= RE.RCk[eq1]
                        RE.pXU[r, k] = Ue .* RE.X[eq1][:, c:RE.RCk[eq1]]
                    else
                        RE.pXU[r, k] = zeros(base.N, 0)
                    end

                    if eq1 ∈ RE.REEqs
                        RE.pXU[r, k] = hcat(RE.pXU[r, k], Ue)
                    end

                    for eq2 ∈ (eq1+1):RE.NEq
                        k += 1
                        if RE.RCk[eq2] > 0
                            RE.pXU[r, k] = Ue .* RE.X[eq2]
                        else
                            RE.pXU[r, k] = zeros(base.N, 0)
                        end
                        if eq2 ∈ RE.REEqs
                            RE.pXU[r, k] = hcat(RE.pXU[r, k], Ue)
                        end
                    end
                end
            end
        end
    else
        # Simpler form for random effects only (no random coefficients)
        for r ∈ RE.R:-1:1
            for j ∈ RE.d:-1:1
                RE.pXU[r, j] = RE.U[r][:, j:j]
            end
        end
    end

    # Copy to subview-level XU
    for v ∈ model.subviews
        for r ∈ RE.R:-1:1
            for j ∈ size(v.XU[l], 2):-1:1
                if !isempty(RE.pXU[r, j])
                    v.XU[l][r, j] = RE.pXU[r, j][v.SubsampleInds, :]
                end
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
    for j ∈ 1:size(GammaInd, 1)
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
    for eq1 ∈ 1:d
        if !isempty(vNumCuts) && eq1 <= length(vNumCuts) && vNumCuts[eq1] > 0
            for cut ∈ 2:vNumCuts[eq1]+1
                if !isempty(aux_params)
                    cuts[cut, eq1] = next_aux()
                    # Check truncation bounds
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

    # === Adaptive quadrature iteration check ===
    # (Translated from Mata lines 1275-1287)
    # Use st_global("ML_ic") to get current iteration since moptimize_result_iterations() not available
    NewIter = false
    if model.WillAdapt
        Iter = parse(Int, st_global("ML_ic"))
        if Iter != model.LastIter
            NewIter = true
            model.LastIter = Iter
            if !model.Adapted
                if model.AdaptNextTime
                    model.Adapted = true
                    model.AdaptivePhaseThisEst = true
                    println("\nPerforming Naylor-Smith adaptive quadrature.")
                else
                    if !isempty(model.Lastb)
                        # Criterion to begin adaptive phase: relative change < 0.1
                        model.AdaptNextTime = maximum(abs.((b .- model.Lastb) ./ (abs.(model.Lastb) .+ 1e-10))) < 0.1
                    end
                    model.Lastb = copy(b)
                end
            end
        end
    end

    # === Random effects parameters (sig, rho) for each level ===
    SigXform = model.SigXform

    for l ∈ 1:L
        RE = REs[l]
        RE.sig = Float64[]
        RE.rho = Float64[]

        lnsigWithin = 0.0
        lnsigAcross = 0.0

        # Exchangeable across levels?
        if RE.covAcross == 0 && !isempty(aux_params)
            lnsigWithin = lnsigAcross = next_aux()
        end

        for eq1 ∈ 1:RE.NEq
            # Exchangeable within but not across?
            if !isempty(RE.covWithin) && RE.Eqs[eq1] <= length(RE.covWithin) &&
               RE.covWithin[RE.Eqs[eq1]] == 0 && RE.covAcross != 0
                lnsigWithin = lnsigAcross
            end

            for c1 ∈ 1:RE.NEff[eq1]
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

        for eq1 ∈ 1:RE.NEq
            if !isempty(RE.covWithin) && RE.Eqs[eq1] <= length(RE.covWithin)
                if RE.covWithin[RE.Eqs[eq1]] == 2  # independent?
                    atanhrhoWithin = 0.0
                elseif RE.covWithin[RE.Eqs[eq1]] == 0 && RE.NEff[eq1] > 1 && !isempty(aux_params)
                    # Exchangeable within
                    atanhrhoWithin = next_aux()
                end
            end

            for c1 ∈ 1:RE.NEff[eq1]
                for c2 ∈ c1+1:RE.NEff[eq1]
                    if !isempty(RE.covWithin) && RE.Eqs[eq1] <= length(RE.covWithin) &&
                       RE.covWithin[RE.Eqs[eq1]] == 1 && !isempty(aux_params)
                        # Unstructured within
                        atanhrhoWithin = next_aux()
                    end
                    push!(RE.rho, atanhrhoWithin)
                end

                for eq2 ∈ eq1+1:RE.NEq
                    for c2 ∈ 1:RE.NEff[eq2]
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
    for l ∈ 1:L
        RE = REs[l]
        if RE.d == 1
            RE.T = fill(RE.sig[1], 1, 1)
            RE.Sig = RE.T .^ 2
        else
            k = 0
            for j ∈ 1:RE.d
                for i ∈ j+1:RE.d
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

    for v ∈ model.subviews
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
    # Full draw loop structure from Mata lf1() lines 1367-1681
    base.plnL = lnf

    # Initialize score accumulators for hierarchical models
    sThetaScores = zeros(0, 0)
    sCutScores = zeros(0, 0)
    sTScores = [zeros(0, 0) for _ in 1:L]
    sGammaScores = [zeros(0, 0) for _ in 1:sum(model.G)]

    # Main draw loop - iterates through all draw combinations for hierarchical RE models
    # For L=1 (non-hierarchical), this loop runs exactly once
    l = L  # Initialize for do-while logic
    while true  # do { ... } while (l) in Mata
        for v ∈ model.subviews
            # Compute errors for this subview (translated from Mata lf1() lines 1369-1444)
            EUncensEq = 0
            ECensEq = 0
            FCensEq = 0
            tEq = 0

            for i ∈ 1:d
                if v.TheseInds[i] == cmp_mprobit  # handle mprobit below
                    ECensEq += 1
                    FCensEq += 1
                elseif v.NotBaseEq[i]
                    # Copy theta from base level to subview
                    # Reshape to column vector for consistent array operations
                    v.theta[i] = reshape(base.theta[i][v.SubsampleInds], :, 1)

                    if v.TheseInds[i] > 0 && isfinite(v.TheseInds[i])
                        if v.TheseInds[i] == cmp_cont
                            # Continuous: E = y - theta
                            EUncensEq += 1
                            v.EUncens[:, EUncensEq] = v.y[i] .- v.theta[i]
                        else
                            ECensEq += 1
                            if v.TheseInds[i] == cmp_left || v.TheseInds[i] == cmp_int
                                # Left censored/interval: E = y - theta
                                v.pECens[:, ECensEq] = v.y[i] .- v.theta[i]
                            elseif v.TheseInds[i] == cmp_right
                                # Right censored: E = theta - y
                                v.pECens[:, ECensEq] = v.theta[i] .- v.y[i]
                            elseif v.TheseInds[i] == cmp_probit
                                # Probit y=0: E = -theta
                                v.pECens[:, ECensEq] = -v.theta[i]
                            elseif v.TheseInds[i] == cmp_probity1 || v.TheseInds[i] == cmp_frac
                                # Probit y=1 or fractional: E = theta
                                v.pECens[:, ECensEq] = v.theta[i]
                            elseif v.TheseInds[i] == cmp_oprobit
                                # Ordered probit: E = cuts[y+1] - theta
                                y_int = Int.(v.y[i])
                                if !isempty(trunceqs) && trunceqs[i] != 0
                                    # Truncated oprobit
                                    t = v.y[i] .> model.vNumCuts[i]
                                    v.pECens[:, ECensEq] = (t .* v.Ut[i] .+ (1 .- t) .* cuts[y_int .+ 1, i]) .- v.theta[i]
                                else
                                    v.pECens[:, ECensEq] = cuts[y_int .+ 1, i] .- v.theta[i]
                                end
                            else
                                # Roprobit: E = -theta
                                v.pECens[:, ECensEq] = -v.theta[i]
                            end

                            # Handle lower bounds for two-sided censored equations
                            if !isempty(v.pF) && ECensEq <= length(model.NonbaseCases) && model.NonbaseCases[ECensEq] != 0
                                FCensEq += 1
                                if v.TheseInds[i] == cmp_int
                                    # Interval regression lower bound
                                    v.pF[:, FCensEq] = v.yL[i] .- v.theta[i]
                                elseif v.TheseInds[i] == cmp_oprobit
                                    # Ordered probit lower bound
                                    if !isempty(trunceqs) && trunceqs[i] != 0
                                        t = v.y[i]
                                        v.pF[:, FCensEq] = (t .* v.Lt[i] .+ (1 .- t) .* cuts[y_int, i]) .- v.theta[i]
                                    else
                                        v.pF[:, FCensEq] = cuts[y_int, i] .- v.theta[i]
                                    end
                                elseif !isempty(trunceqs) && trunceqs[i] != 0
                                    if v.TheseInds[i] == cmp_left
                                        v.pF[:, FCensEq] = v.Lt[i] .- v.theta[i]
                                    elseif v.TheseInds[i] == cmp_right
                                        v.pF[:, FCensEq] = v.theta[i] .- v.Ut[i]
                                    elseif v.TheseInds[i] == cmp_probit
                                        v.pF[:, FCensEq] = v.Lt[i] .- v.theta[i]
                                    elseif v.TheseInds[i] == cmp_probity1
                                        v.pF[:, FCensEq] = v.theta[i] .- v.Ut[i]
                                    end
                                end
                            end
                        end

                        # Truncation bounds
                        if !isempty(trunceqs) && trunceqs[i] != 0
                            tEq += 1
                            if v.TheseInds[i] == cmp_left
                                v.pEt[:, tEq] = v.Ut[i] .- v.theta[i]
                                v.pFt[:, tEq] = v.pF[:, FCensEq]
                            elseif v.TheseInds[i] == cmp_right
                                v.pEt[:, tEq] = v.theta[i] .- v.Lt[i]
                                v.pFt[:, tEq] = v.pF[:, FCensEq]
                            elseif v.TheseInds[i] == cmp_probit
                                v.pEt[:, tEq] = v.Ut[i] .- v.theta[i]
                                v.pFt[:, tEq] = v.pF[:, FCensEq]
                            elseif v.TheseInds[i] == cmp_probity1
                                v.pEt[:, tEq] = v.theta[i] .- v.Lt[i]
                                v.pFt[:, tEq] = v.pF[:, FCensEq]
                            elseif v.TheseInds[i] ∈ (cmp_cont, cmp_oprobit, cmp_int)
                                v.pEt[:, tEq] = v.Ut[i] .- v.theta[i]
                                v.pFt[:, tEq] = v.Lt[i] .- v.theta[i]
                            end
                        end
                    end
                end
            end

            # Mprobit error computation (relative-difference errors)
            # Translated from Mata lines 1447-1452
            MprobitGroupInds = model.MprobitGroupInds
            for j ∈ size(MprobitGroupInds, 1):-1:1
                if !isempty(v.mprobit) && j <= length(v.mprobit) && v.mprobit[j].d > 0
                    out_eq = v.mprobit[j].out
                    out = base.theta[out_eq][v.SubsampleInds]
                    for ii ∈ v.mprobit[j].d:-1:1
                        in_eq = v.mprobit[j].in[ii]
                        res_idx = v.mprobit[j].res[ii]
                        v.pECens[:, res_idx] = out .- base.theta[in_eq][v.SubsampleInds]
                    end
                end
            end

            # Replace missing values with large numbers (like Mata's maxfloat)
            if !isempty(v.pECens)
                replace!(x -> ismissing(x) || isnan(x) ? 1.701e38 : x, v.pECens)
            end
            if !isempty(v.pF)
                replace!(x -> ismissing(x) || isnan(x) ? -1.701e38 : x, v.pF)
            end
            if !isempty(v.pEt)
                replace!(x -> ismissing(x) || isnan(x) ? 1.701e38 : x, v.pEt)
            end
            if !isempty(v.pFt)
                replace!(x -> ismissing(x) || isnan(x) ? -1.701e38 : x, v.pFt)
            end

            # Compute likelihood
            if v.d_cens > 0
                lnL_v = lnLCensored!(model, v, todo)
                if v.d_uncens > 0
                    phi, dphi_dE, dphi_dSig = lnLContinuous(v.EUncens, v.Omega, v.uncens, v.one2N, todo)
                    lnL_v = lnL_v .+ phi
                end
            else
                lnL_v, dphi_dE, dphi_dSig = lnLContinuous(v.EUncens, v.Omega, v.uncens, v.one2N, todo)
            end

            # Truncation correction
            if v.d_trunc > 0
                Phi_trunc, dPhi_dEt, dPhi_dFt, dPhi_dSigt = lnLTrunc(
                    v.pEt, v.pFt, v.Omega, v.trunc, v.one2d_trunc, v.one2N, todo,
                    model.ghk2DrawSet, model.ghkAnti, v.GHKStartTrunc)
                lnL_v = lnL_v .- Phi_trunc
            end

            base.plnL[v.SubsampleInds] = lnL_v

            # Score computation for single-level models (L=1)
            if todo != 0 && L == 1
                pdlnL_dtheta = v.d_cens > 0 ?
                    (v.d_uncens > 0 ? v.dphi_dE .+ v.dPhi_dE : v.dPhi_dE) :
                    v.dphi_dE
                pdlnL_dSig = v.d_cens > 0 ?
                    (v.d_uncens > 0 ? v.dphi_dSig .+ v.dPhi_dSig : v.dPhi_dSig) :
                    v.dphi_dSig

                if v.d_trunc > 0
                    pdlnL_dtheta = pdlnL_dtheta .- v.dPhi_dEt
                    pdlnL_dSig = pdlnL_dSig .- v.dPhi_dSigt
                end

                pdlnL_dtheta = pdlnL_dtheta * v.QEinvGamma

                # Store theta scores
                S[v.SubsampleInds, model.Scores_.ThetaScores] = pdlnL_dtheta

                # Store cut scores
                if model.NumCuts > 0
                    S[v.SubsampleInds, model.Scores_.CutScores] = v.dPhi_dcuts
                end

                # Store Sig scores
                if !isempty(base.D) && size(base.D, 2) > 0
                    S[v.SubsampleInds, collect(model.Scores_.SigScores[L])] = pdlnL_dSig * v.invGammaQSigD
                end

                # Gamma scores would go here for simultaneous equations models
            end
        end

        # Multi-level likelihood aggregation (Mata lines 1533-1680)
        # For hierarchical models, aggregate likelihoods across draws at each level
        for l_agg ∈ (L-1):-1:1
            RE = REs[l_agg]
            ThisDraw = model.ThisDraw

            # Aggregate lnL to group level
            RE.lnLByDraw[:, ThisDraw[l_agg+1]] = panelsum(REs[l_agg+1].plnL, REs[l_agg+1].Weights, RE.IDRangesGroup)

            if ThisDraw[l_agg+1] < RE.R
                # Increment draw counter
                model.ThisDraw[l_agg+1] += 1
            else
                # Finished all draws at this level - aggregate and reset
                if model.Adapted
                    RE.lnLByDraw .+= RE.AdaptiveShift
                end

                # Compute weights proportional to L (not lnL) for the group
                t = RE.lnLlimits .- extrema.(eachrow(RE.lnLByDraw))
                lnLmin = [x[1] for x in t]
                lnLmax = [x[2] for x in t]
                t = lnLmin .* (lnLmin .> 0) .- lnLmax
                shift = t .* (t .< 0) .+ lnLmax
                L_g = replace(exp.(RE.lnLByDraw .+ shift), NaN => 0.0)

                if model.Quadrature
                    L_g = L_g .* RE.QuadW'
                end

                RE.plnL = vec(sum(L_g, dims=2))  # Sum of likelihoods across draws

                if todo != 0 || (model.AdaptivePhaseThisEst && model.WillAdapt)
                    L_g = replace(L_g ./ RE.plnL, NaN => 0.0)  # Normalize as weights
                end

                # Adaptive quadrature update (Naylor-Smith adaptation) - Mata lines 1554-1618
                if model.AdaptivePhaseThisEst && NewIter
                    ThisDrawKey = model.ThisDraw[1:l_agg]

                    # Get or initialize adaptive quadrature points for this draw combination
                    if !haskey(RE.QuadXAdapt, ThisDrawKey)
                        RE.QuadXAdapt[ThisDrawKey] = [copy(RE.QuadX) for _ in 1:RE.N]
                    end
                    ThisQuadXAdapt = RE.QuadXAdapt[ThisDrawKey]

                    if RE.d == 1
                        # Optimized 1-D case
                        for j ∈ RE.N:-1:1
                            if RE.ToAdapt[j] > 0
                                t = L_g[j, :]
                                QuadXAdapt_j = ThisQuadXAdapt[j]

                                # Weighted mean
                                RE.QuadMean[j] = t' * QuadXAdapt_j

                                # Weighted standard deviation
                                C_vec = QuadXAdapt_j .- RE.QuadMean[j]
                                C = sqrt(t' * (C_vec .* C_vec))

                                if isnan(C) || C == 0
                                    # Diverged - restart but decrement counter
                                    RE.ToAdapt[j] -= 1
                                    ThisQuadXAdapt[j] = copy(RE.QuadX)
                                    RE.AdaptiveShift[j, :] .= 0.0
                                else
                                    RE.QuadSD[j] = [C]
                                    newQuadX = RE.QuadX .* C .+ RE.QuadMean[j]

                                    # Check convergence
                                    if maximum(abs.(QuadXAdapt_j .- newQuadX) ./ (abs.(QuadXAdapt_j) .+ 1e-10)) < model.QuadTol
                                        RE.ToAdapt[j] = 0
                                        continue
                                    end

                                    ThisQuadXAdapt[j] = newQuadX
                                    # Adaptive shift: ln(det(C)) - 0.5*(adapted_x^2 - original_x^2)
                                    ln2pi_2 = 0.91893853320467267
                                    RE.AdaptiveShift[j, :] = (log(C) - ln2pi_2) .- (0.5 .* newQuadX'.^2 .+ RE.lnnormaldenQuadX)
                                end

                                # Update draws for this group
                                for r ∈ RE.R:-1:1
                                    rng = RE.IDRanges[j]
                                    RE.U[r][rng, :] .= ThisQuadXAdapt[j][r]
                                end
                            end
                        end
                    else
                        # Multi-dimensional case
                        for j ∈ RE.N:-1:1
                            if RE.ToAdapt[j] > 0
                                t = L_g[j, :]
                                QuadXAdapt_j = ThisQuadXAdapt[j]

                                # Weighted mean
                                RE.QuadMean[j] = vec(t' * QuadXAdapt_j)

                                # Weighted covariance and Cholesky
                                centered = QuadXAdapt_j .- RE.QuadMean[j]'
                                weighted_cov = (centered .* t)' * centered
                                C_result = cholesky(Symmetric(weighted_cov), check=false)

                                if !issuccess(C_result)
                                    # Diverged - restart
                                    RE.ToAdapt[j] -= 1
                                    ThisQuadXAdapt[j] = copy(RE.QuadX)
                                    RE.AdaptiveShift[j, :] .= 0.0
                                else
                                    C = C_result.L
                                    RE.QuadSD[j] = diag(C)
                                    newQuadX = RE.QuadX * C' .+ RE.QuadMean[j]'

                                    # Check convergence
                                    if maximum(abs.(QuadXAdapt_j .- newQuadX) ./ (abs.(QuadXAdapt_j) .+ 1e-10)) < model.QuadTol
                                        RE.ToAdapt[j] = 0
                                        continue
                                    end

                                    ThisQuadXAdapt[j] = newQuadX
                                    # Adaptive shift
                                    RE.AdaptiveShift[j, :] = vec(quadrowsum_lnnormalden(newQuadX, sum(log.(RE.QuadSD[j])))) .- RE.lnnormaldenQuadX
                                end

                                # Update draws
                                for r ∈ RE.R:-1:1
                                    rng = RE.IDRanges[j]
                                    RE.U[r][rng, :] .= ThisQuadXAdapt[j][r, :]'
                                end
                            end
                        end
                    end

                    # Check if adaptation should continue
                    RE.AdaptivePhaseThisIter = any(RE.ToAdapt .> 0) && (RE.AdaptivePhaseThisIter - 1) % model.QuadIter != 0
                    if RE.AdaptivePhaseThisIter
                        buildtotaleffects!(model, l_agg)
                        if model._todo != 0
                            buildXU!(model, l_agg)
                        end
                    end
                end

                model.ThisDraw[l_agg+1] = 1
            end

            if model.ThisDraw[l_agg+1] > 1 || RE.AdaptivePhaseThisIter
                # Propagate draw changes down the tree
                for _l ∈ l_agg:(L-1)
                    for eq_idx ∈ length(REs[_l].GammaEqs):-1:1
                        _eq = REs[_l].GammaEqs[eq_idx]
                        if !isempty(REs[_l].TotalEffect) && size(REs[_l].TotalEffect, 1) >= model.ThisDraw[_l+1] &&
                           size(REs[_l].TotalEffect, 2) >= _eq && !isempty(REs[_l].TotalEffect[model.ThisDraw[_l+1], _eq])
                            REs[_l+1].theta[_eq] = REs[_l].theta[_eq] .+ REs[_l].TotalEffect[model.ThisDraw[_l+1], _eq]
                        else
                            REs[_l+1].theta[_eq] = REs[_l].theta[_eq]
                        end
                    end
                end
                l = l_agg  # Set l to break out of this loop
                break
            end

            # Score accumulation across draws for hierarchical models (Mata lines 1632-1675)
            if todo != 0
                for v ∈ model.subviews
                    # Get likelihood weights for this subview's observations
                    if !isempty(v.id) && l_agg <= length(v.id) && !isempty(v.id[l_agg])
                        L_gv = L_g[Int.(v.id[l_agg]), RE.one2R]
                    else
                        L_gv = L_g[v.SubsampleInds, :]
                    end

                    # Accumulate scores weighted by likelihood contributions
                    for r ∈ 1:model.NumREDraws[l_agg+1]
                        L_gvr = L_gv[:, r]

                        # Accumulate theta scores
                        if !isempty(v.Scores) && l_agg+1 <= length(v.Scores) &&
                           r <= length(v.Scores[l_agg+1]) && !isempty(v.Scores[l_agg+1][r].ThetaScores)
                            scoreaccum!(sThetaScores, r, L_gvr, v.Scores[l_agg+1][r].ThetaScores)
                        end

                        # Accumulate cut scores
                        if model.NumCuts > 0 && !isempty(v.Scores) && l_agg+1 <= length(v.Scores) &&
                           r <= length(v.Scores[l_agg+1]) && !isempty(v.Scores[l_agg+1][r].CutScores)
                            scoreaccum!(sCutScores, r, L_gvr, v.Scores[l_agg+1][r].CutScores)
                        end

                        # Accumulate T scores for each level
                        for i ∈ L:-1:1
                            if REs[i].NSigParams > 0 && !isempty(v.Scores) && l_agg+1 <= length(v.Scores) &&
                               r <= length(v.Scores[l_agg+1]) && !isempty(v.Scores[l_agg+1][r].TScores) &&
                               i <= length(v.Scores[l_agg+1][r].TScores) && !isempty(v.Scores[l_agg+1][r].TScores[i])
                                scoreaccum!(sTScores[i], r, L_gvr, v.Scores[l_agg+1][r].TScores[i])
                            end
                        end

                        # Accumulate gamma scores
                        for i ∈ length(sGammaScores):-1:1
                            if !isempty(v.Scores) && l_agg+1 <= length(v.Scores) &&
                               r <= length(v.Scores[l_agg+1]) && !isempty(v.Scores[l_agg+1][r].GammaScores) &&
                               i <= length(v.Scores[l_agg+1][r].GammaScores) && !isempty(v.Scores[l_agg+1][r].GammaScores[i])
                                scoreaccum!(sGammaScores[i], r, L_gvr, v.Scores[l_agg+1][r].GammaScores[i])
                            end
                        end
                    end

                    # Final scores at level 1, or store for next level up
                    if l_agg == 1
                        # Apply weight product and store final scores
                        WeightProduct = !isempty(v.WeightProduct) ? v.WeightProduct : ones(v.N)

                        S[v.SubsampleInds, model.Scores_.ThetaScores] = sThetaScores .* WeightProduct

                        if model.NumCuts > 0
                            S[v.SubsampleInds, model.Scores_.CutScores] = sCutScores .* WeightProduct
                        end

                        if base.NSigParams > 0
                            S[v.SubsampleInds, collect(model.Scores_.SigScores[L])] = (sTScores[L] * v.invGammaQSigD) .* WeightProduct
                        end

                        for i ∈ (L-1):-1:1
                            if REs[i].NSigParams > 0
                                S[v.SubsampleInds, collect(model.Scores_.SigScores[i])] = (sTScores[i] * REs[i].D) .* WeightProduct
                            end
                        end

                        # Gamma scores
                        idx = 1
                        for m ∈ 1:d
                            for c ∈ 1:model.G[m]
                                if v.TheseInds[m] != 0
                                    if !isempty(v.dOmega_dGamma) && m <= size(v.dOmega_dGamma, 1) &&
                                       c <= size(v.dOmega_dGamma, 2) && !isempty(v.dOmega_dGamma[m, c])
                                        S[v.SubsampleInds, model.Scores_.GammaScores[idx]] =
                                            (sGammaScores[idx] .+ sTScores[L] * v.dOmega_dGamma[m, c]) .* WeightProduct
                                    else
                                        S[v.SubsampleInds, model.Scores_.GammaScores[idx]] = sGammaScores[idx] .* WeightProduct
                                    end
                                else
                                    S[v.SubsampleInds, model.Scores_.GammaScores[idx]] .= 0.0
                                end
                                idx += 1
                            end
                        end
                    else
                        # Store scores for next level up
                        ThisDraw = model.ThisDraw
                        if !isempty(v.Scores) && l_agg <= length(v.Scores) && ThisDraw[l_agg] <= length(v.Scores[l_agg])
                            v.Scores[l_agg][ThisDraw[l_agg]].ThetaScores = copy(sThetaScores)
                            if model.NumCuts > 0
                                v.Scores[l_agg][ThisDraw[l_agg]].CutScores = copy(sCutScores)
                            end
                            for i ∈ L:-1:1
                                if REs[i].NSigParams > 0
                                    v.Scores[l_agg][ThisDraw[l_agg]].TScores[i] = copy(sTScores[i])
                                end
                            end
                            for i ∈ 1:length(sGammaScores)
                                v.Scores[l_agg][ThisDraw[l_agg]].GammaScores[i] = copy(sGammaScores[i])
                            end
                        end
                    end
                end
            end

            RE.plnL = log.(RE.plnL) .- shift
            if model.Quadrature == 0
                RE.plnL = RE.plnL .- RE.lnNumREDraws
            end

            l = l_agg  # Update l for the while condition
        end

        # Exit condition: when l reaches 0 (all draws processed)
        # For L=1, the inner for loop doesn't run, l stays at L=1, and we exit
        if L == 1 || l == 0
            break
        end
        l = L  # Reset for next iteration
    end

    # Final aggregation for hierarchical models (Mata lines 1683-1694)
    if L > 1
        if !isempty(REs[1].Weights)
            total_lnL = sum(REs[1].Weights .* REs[1].plnL)
        else
            total_lnL = sum(REs[1].plnL)
        end

        if todo == 0
            lnf .= total_lnL / N  # Distribute equally for ml
        end
    end

    return (lnf, S)
end

function gf1!(model::CmpModel, todo::Int, b::AbstractVector)
    # Group evaluator - calls lf1! then aggregates to group level
    # Translated from Mata gf1() lines 1704-1732
    # For survey/clustered data, computes group-level likelihood and scores

    REs = model.REs
    base = model.base
    d = model.d

    # Call lf1! to get observation-level results
    lnf_obs, S_obs = lf1!(model, todo, b)

    # Check for errors
    if any(isnan.(lnf_obs))
        return (lnf_obs, S_obs)
    end

    # Get group-level log-likelihood from top RE level
    lnf = REs[1].plnL

    if todo != 0
        # Need to expand scores by X matrices and aggregate to groups
        K = length(b)  # number of parameters
        n_groups = length(REs[1].IDRanges)
        S = zeros(n_groups, K)

        # X matrices should be populated by the Stata wrapper (cmp_gf1.ado)
        # or we lazily load them here
        if isempty(model.X) || length(model.X) < d
            # X matrices not available - this would be an error in practice
            # In Mata, these come from moptimize_util_indepvars(M, i)
            # For Julia, they should be passed from Stata
            @warn "X matrices not populated for gf1! score expansion"
            return (lnf, S)
        end

        # For each equation, expand scores by X and aggregate to groups
        col_start = 1
        for i ∈ 1:d
            X_i = model.X[i]
            n_vars = size(X_i, 2)
            if n_vars > 0
                # Scores for equation i, expanded by X
                scores_expanded = S_obs[:, i] .* X_i
                # Aggregate to group level
                S[:, col_start:col_start+n_vars-1] = panelsum(scores_expanded, model.WeightProduct, REs[1].IDRanges)
            end
            col_start += n_vars
        end

        # Handle auxiliary parameters (cuts, sig, rho, gamma) if any
        if col_start <= K
            # Auxiliary parameter scores don't need X expansion
            aux_scores = S_obs[:, d+1:end]
            S[:, col_start:K] = panelsum(aux_scores, model.WeightProduct, REs[1].IDRanges)
        end

        return (lnf, S)
    end

    return (lnf, zeros(0, 0))
end

# Helper: get indices for vech of submatrix
function vSigInds(inds::AbstractVector, d::Int)
    if isempty(inds)
        return Int[]
    end
    result = Int[]
    for (idx, i) ∈ enumerate(inds)
        for j ∈ 1:idx
            # Convert to vech index: for d×d matrix, vech index of (i,j) where i>=j
            # is sum(d-k+1 for k=1 to j-1) + (i-j+1) = j*(2d-j+1)/2 - d + i
            ii, jj = max(inds[idx], inds[j]), min(inds[idx], inds[j])
            vech_idx = jj * (2 * d - jj + 1) ÷ 2 - d + ii
            push!(result, vech_idx)
        end
    end
    return result
end

# Initialize model before estimation - simplified version that reads data from Stata
# This is called from cmp.ado without parameters; data is read using st_data()
function cmp_init!(model::CmpModel)
    d = model.d

    # Read indicator matrix from Stata
    # The indicator variable names are stored in model.indVars
    N = 0
    indicators = zeros(0, d)
    if !isempty(model.indVars)
        for (eq, varname) ∈ enumerate(model.indVars)
            if !isempty(varname)
                col = st_data(varname)
                if N == 0
                    N = length(col)
                    indicators = zeros(N, d)
                end
                indicators[:, eq] = col
            end
        end
    end

    # Read dependent variables from yVars (following Mata lines 1790-1799)
    y = [isempty(model.yVars) || eq > length(model.yVars) || isempty(model.yVars[eq]) ?
         zeros(0, 0) : reshape(st_data(model.yVars[eq]), :, 1) for eq ∈ 1:d]
    # Read truncation bounds from LtVars/UtVars
    Lt = [isempty(model.LtVars) || eq > length(model.LtVars) || isempty(model.LtVars[eq]) ?
          zeros(0, 0) : reshape(st_data(model.LtVars[eq]), :, 1) for eq ∈ 1:d]
    Ut = [isempty(model.UtVars) || eq > length(model.UtVars) || isempty(model.UtVars[eq]) ?
          zeros(0, 0) : reshape(st_data(model.UtVars[eq]), :, 1) for eq ∈ 1:d]
    # Read interval regression lower bounds from yLVars (separate from y)
    yL = [isempty(model.yLVars) || eq > length(model.yLVars) || isempty(model.yLVars[eq]) ?
          zeros(0, 0) : reshape(st_data(model.yLVars[eq]), :, 1) for eq ∈ 1:d]

    # Use the full initialization
    return cmp_init_full!(model, indicators, y, Lt, Ut, yL)
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
    model.REs = [RE() for _ ∈ 1:L]
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
        model.Idd = Matrix{Float64}(I, d * d, d * d)
        model.vLd = vec(sum(Lmatrix(d) .* (1:d*d)', dims=2))
        Kd = kron(I(d), I(d))  # Commutation matrix approximation
        model.vKd = vec(sum(Kd .* (1:d*d), dims=1))
    end

    model.ThisDraw = ones(Int, L)

    # Setup each random effects level
    for l ∈ L:-1:1
        RE = model.REs[l]
        RE.Eqs = findall(Eqs[:, l] .!= 0)
        RE.NEq = length(RE.Eqs)
        RE.NEff = [Int(NumEff[l, eq]) for eq ∈ RE.Eqs]
        RE.GammaEqs = model.HasGamma ? findall((GammaId * Eqs[:, l]) .!= 0) : RE.Eqs
        RE.d = sum(RE.NEff)
        RE.one2d = collect(1:RE.d)
        RE.theta = [zeros(0, 0) for _ ∈ 1:d]
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
    for l ∈ L:-1:1
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
            for m ∈ 1:d
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
        model.Scores_.SigScores = [zeros(0, 0) for _ ∈ 1:L]
        for l ∈ 1:L
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
    for l ∈ L-1:-1:1
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
        RE.U = [zeros(base.N, RE.d) for _ ∈ 1:RE.R]
        RE.TotalEffect = [zeros(0, 0) for _ ∈ 1:RE.R, _ ∈ 1:d]

        # Setup panel structure
        RE.IDRanges = panelsetup(vec(RE.id))
        RE.IDRangeLengths = [length(r) for r ∈ RE.IDRanges]

        # Generate random draws (simplified - random draws)
        NDraws = RE.R ÷ model.REAnti
        U = randn(RE.N * NDraws, RE.d)

        # Distribute draws to groups
        S = repeat(collect(1:NDraws), inner=RE.N)
        for r ∈ NDraws:-1:1
            mask = S .== r
            if sum(mask) > 0
                RE.U[r] = U[mask, :]
            end
            if model.REAnti == 2 && r + RE.R ÷ 2 <= RE.R
                RE.U[r+RE.R÷2] = -RE.U[r]
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

        # Number of cut parameters for this subview's oprobit equations
        v.vNumCuts = !isempty(model.vNumCuts) ? [model.vNumCuts[eq] for eq ∈ v.oprobit] : Int[]
        v.NumCuts = sum(v.vNumCuts)

        v.trunc = findall(model.trunceqs .!= 0)
        v.d_trunc = length(v.trunc)
        v.one2d_trunc = collect(1:v.d_trunc)

        # Censored equations (excluding mprobit base)
        v.cens = findall((TheseInds .> cmp_cont) .& (TheseInds .< Inf) .&
                         ((TheseInds .< mprobit_ind_base) .| (TheseInds .>= roprobit_ind_base)))
        v.d_cens = length(v.cens)
        d_cens = max(d_cens, v.d_cens)

        # dCensNonrobase: number of censored equations that are not base alternatives
        # (from Mata line 2031)
        cens_nonrobase_mask = (!isempty(model.NonbaseCases) ? (model.NonbaseCases .!= 0) : trues(d)) .&
                              (TheseInds .> cmp_cont) .& (TheseInds .< Inf) .&
                              ((TheseInds .< mprobit_ind_base) .| (TheseInds .>= roprobit_ind_base))
        v.cens_nonrobase = findall(cens_nonrobase_mask)
        v.dCensNonrobase = length(v.cens_nonrobase)

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
        v.theta = [zeros(0, 0) for _ ∈ 1:d]
        v.y = [isempty(y[i]) ? zeros(0, 0) : y[i][v.SubsampleInds, :] for i ∈ 1:d]
        v.Lt = [isempty(Lt[i]) ? zeros(0, 0) : Lt[i][v.SubsampleInds, :] for i ∈ 1:d]
        v.Ut = [isempty(Ut[i]) ? zeros(0, 0) : Ut[i][v.SubsampleInds, :] for i ∈ 1:d]
        v.yL = [isempty(yL[i]) ? zeros(0, 0) : yL[i][v.SubsampleInds, :] for i ∈ 1:d]

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
            v.id = [zeros(0, 0) for _ ∈ 1:L-1]
            for l ∈ 1:L-1
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

# =============================================================================
# Sparse Grids Integration - Kronecker-Patterson Nested Nodes and Weights
# =============================================================================

# KPN (Kronecker-Patterson Nested) 1D nodes
# These are nested nodes for Gaussian quadrature that allow efficient sparse grids
# Hex constants from Mata converted to Float64
function KPNn1d()
    n1d = Vector{Union{Nothing, Vector{Float64}}}(nothing, 25)
    n1d[1] = [0.0]
    n1d[2] = nothing
    n1d[3] = [0.0, 1.7320508075688772]  # sqrt(3)
    n1d[4] = [0.0, 0.7414959076854169, 1.7320508075688772, 4.183250089986876]
    n1d[5] = nothing
    n1d[6] = nothing
    n1d[7] = nothing
    n1d[8] = [0.0, 0.7414959076854169, 1.7320508075688772, 2.8612795760570582, 4.183250089986876]
    n1d[9] = [0.0, 0.7414959076854169, 1.2304236340273060, 1.7320508075688772, 2.5960831150492023,
              2.8612795760570582, 4.183250089986876, 5.187698957197879, 6.363947888829836]
    n1d[10] = nothing
    n1d[11] = nothing
    n1d[12] = nothing
    n1d[13] = nothing
    n1d[14] = nothing
    n1d[15] = [0.0, 0.7414959076854169, 1.2304236340273060, 1.7320508075688772, 2.5960831150492023,
               2.8612795760570582, 3.2053337944991945, 4.183250089986876, 5.187698957197879, 6.363947888829836]
    n1d[16] = [0.0, 0.2453407083009012, 0.7414959076854169, 1.2304236340273060, 1.7320508075688772,
               2.2336260616769415, 2.5960831150492023, 2.8612795760570582, 3.2053337944991945,
               3.6353185190372783, 4.183250089986876, 5.187698957197879, 6.363947888829836,
               7.123267590839759, 7.937990921564168, 9.133710089388968]
    n1d[17] = [0.0, 0.2453407083009012, 0.7414959076854169, 1.2304236340273060, 1.7320508075688772,
               2.2336260616769415, 2.5960831150492023, 2.8612795760570582, 3.2053337944991945,
               3.6353185190372783, 4.183250089986876, 5.187698957197879, 5.697693884920621,
               6.363947888829836, 7.123267590839759, 7.937990921564168, 9.133710089388968]
    n1d[18] = nothing
    n1d[19] = nothing
    n1d[20] = nothing
    n1d[21] = nothing
    n1d[22] = nothing
    n1d[23] = nothing
    n1d[24] = nothing
    n1d[25] = [0.0, 0.2453407083009012, 0.7414959076854169, 1.2304236340273060, 1.7320508075688772,
               2.2336260616769415, 2.5960831150492023, 2.8612795760570582, 3.2053337944991945,
               3.6353185190372783, 4.183250089986876, 4.7368964527797285, 5.187698957197879,
               5.697693884920621, 6.363947888829836, 7.123267590839759, 7.937990921564168, 9.133710089388968]
    return n1d
end

# KPN (Kronecker-Patterson Nested) 1D weights
function KPNw1d()
    w1d = Vector{Union{Nothing, Vector{Float64}}}(nothing, 25)
    w1d[1] = [1.0]
    w1d[2] = nothing
    w1d[3] = [0.6666666666666666, 0.16666666666666666]
    w1d[4] = [0.4587017028787511, 0.13137860698313633, 0.13855327472974924, 2.229393645534102e-3]
    w1d[5] = nothing
    w1d[6] = nothing
    w1d[7] = nothing
    w1d[8] = [0.2539731157454138, 0.2684013316323094, 0.09463904929459604,
              3.224384718268182e-4, 1.5108525619647e-7]
    w1d[9] = [0.2671275291312547, 0.2521461772251085, 6.974596376472268e-4,
              0.08861574604191467, 3.7915700249622566e-6, 1.3890372963737668e-4,
              1.5108525619647e-7, -1.35685795071e-10, 1.2369091848e-14]
    w1d[10] = nothing
    w1d[11] = nothing
    w1d[12] = nothing
    w1d[13] = nothing
    w1d[14] = nothing
    w1d[15] = [0.3026987966988389, 0.20830505459862068, 0.030517553330883955,
               0.06325878421706788, 0.01138136048419631, -3.07952197296e-4,
               5.7101396749744e-5, 1.5108525619647e-7, 4.8963851969e-11,
               1.0032929e-18]
    w1d[16] = [0.25783927756254784, 6.99308395805e-4, 0.19992541568539815,
               0.06551557039265076, 0.030875686131089916, 1.7529770814e-6,
               0.01038813287982181, -2.761605712e-4, 5.403685494654e-5,
               1.5012025866e-8, 1.50305e-7, 4.93e-11, 9.99e-19, 8.6e-24, -3.0e-29, 3.0e-36]
    w1d[17] = [0.06981845905271833, 0.0413122786927, 0.1765050206831844,
               0.07545166143609986, 0.027325316131568854, 3.7212765658e-4,
               0.00917970668, -2.75817e-5, 2.703e-5, 2.12e-7, 5.58e-8, 5.48e-11,
               -1.1e-16, 1.6e-18, -7.7e-23, 5.3e-29, -6.8e-37]
    w1d[18] = nothing
    w1d[19] = nothing
    w1d[20] = nothing
    w1d[21] = nothing
    w1d[22] = nothing
    w1d[23] = nothing
    w1d[24] = nothing
    w1d[25] = [6.656e-9, 0.19199752, 0.14822903, 0.0921086, 0.0453587, 0.0156785,
               0.00478, 0.00146, 4.14e-5, 8.63e-6, 4.474e-9, 1.06e-10, 1.57e-13,
               2.1e-16, 2.89e-20, 2.5e-25, 1.45e-30, 4.47e-42]
    return w1d
end

# GQN (Gauss Quadrature) 1D weights - classical Gauss-Hermite quadrature
function GQNw1d()
    w1d = Vector{Union{Nothing, Vector{Float64}}}(nothing, 25)
    w1d[1] = [1.0]
    w1d[2] = [0.5]
    w1d[3] = [0.6666666666666666, 0.16666666666666666]
    w1d[4] = [0.45412414523193148, 0.04587585476806854]
    w1d[5] = [0.5333333333333333, 0.2222222222222222, 0.011257411327720691]
    w1d[6] = [0.4088847674916566, 0.08841531786366206, 2.2583032023729127e-3]
    w1d[7] = [0.4571428571428571, 0.24047619047619048, 0.030476190476190476, 3.96825396825397e-4]
    w1d[8] = [0.37305758815493944, 0.11723990186906, 0.009635220120788298, 5.648070174825e-5]
    w1d[9] = [0.40634920634920635, 0.24420109580665757, 0.04993919524979327,
              3.392556653748e-3, 7.640432855233e-6]
    w1d[10] = [0.34642851097304634, 0.13548370298026785, 0.019111580500770284,
               9.9681258451e-4, 2.2293936455e-6]
    w1d[11] = [0.36940836940836940, 0.2440097802966442, 0.06613874607171428,
               7.1057968927e-3, 3.0467753172e-4, 5.4826885714e-8]
    w1d[12] = [0.32402207878737, 0.1470679796254, 0.02903918810826,
               2.3244707399e-3, 6.109e-5, 1.114e-7]
    w1d[13] = [0.34099674099674, 0.24031234031234, 0.0785714285714285, 0.01178571428571,
               7.6923076923e-4, 2.089e-5, 3.30e-8]
    w1d[14] = [0.30392398873952, 0.15443020437, 0.039047619047619, 4.676190476e-3,
               2.5252525e-4, 5.5e-6, 7.1e-9]
    w1d[15] = [0.31746031746, 0.2358, 0.0889, 0.0173, 1.67e-3, 7.62e-5, 1.47e-6, 9.72e-10]
    w1d[16] = [0.28571428571, 0.161, 0.0476, 7.33e-3, 5.66e-4, 2.0e-5, 3.0e-7, 1.5e-9]
    w1d[17] = [0.2976, 0.23, 0.0968, 0.0227, 2.85e-3, 1.79e-4, 5.1e-6, 5.7e-8, 1.8e-10]
    w1d[18] = [0.27, 0.164, 0.0543, 9.8e-3, 9.5e-4, 4.6e-5, 1.0e-6, 9.0e-9, 2.3e-11]
    w1d[19] = [0.28, 0.22, 0.103, 0.028, 4.5e-3, 3.9e-4, 1.7e-5, 3.6e-7, 3.0e-9, 7.4e-12]
    w1d[20] = [0.26, 0.165, 0.06, 0.013, 1.5e-3, 9.4e-5, 3.1e-6, 4.9e-8, 3.2e-10, 6.6e-13]
    w1d[21] = [0.27, 0.21, 0.108, 0.034, 6.5e-3, 7.3e-4, 4.5e-5, 1.4e-6, 1.9e-8, 1.0e-10, 1.7e-13]
    w1d[22] = [0.25, 0.165, 0.065, 0.016, 2.3e-3, 1.9e-4, 8.4e-6, 1.9e-7, 2.0e-9, 8.7e-12, 1.2e-14]
    w1d[23] = [0.26, 0.21, 0.11, 0.039, 8.6e-3, 1.2e-3, 9.2e-5, 3.6e-6, 6.7e-8, 5.4e-10, 1.8e-12, 2.0e-15]
    w1d[24] = [0.24, 0.165, 0.07, 0.019, 3.2e-3, 3.3e-4, 1.9e-5, 5.7e-7, 8.4e-9, 5.5e-11, 1.4e-13, 1.2e-16]
    w1d[25] = [0.25, 0.2, 0.115, 0.044, 1.1e-2, 1.8e-3, 1.8e-4, 1.0e-5, 3.1e-7, 4.6e-9, 3.1e-11, 8.5e-14, 8.1e-17, 2.4e-20]
    return w1d
end

# GQN (Gauss Quadrature) 1D nodes
function GQNn1d()
    n1d = Vector{Union{Nothing, Vector{Float64}}}(nothing, 25)
    n1d[1] = [0.0]
    n1d[2] = [1.0]
    n1d[3] = [0.0, 1.7320508075688772]
    n1d[4] = [0.7414959076854169, 2.3344142183389773]
    n1d[5] = [0.0, 1.3556261799742659, 2.8569700138728056]
    n1d[6] = [0.6167065901925942, 1.8891758777537109, 3.324257433552119]
    n1d[7] = [0.0, 1.1544053947399681, 2.3667594107345413, 3.7504397177257425]
    n1d[8] = [0.5390798113513751, 1.6365190424351079, 2.8024858612875416, 4.144547186125899]
    n1d[9] = [0.0, 1.0232556637891326, 2.0768479786778302, 3.2054290028564703, 4.512745863399781]
    n1d[10] = [0.48493570421477814, 1.4659890943911582, 2.4843258416389542, 3.5818234835519270, 4.859462828332312]
    n1d[11] = [0.0, 0.9282329505125345, 1.8760350201548458, 2.8651231606436447, 3.936166607129977, 5.188000939658097]
    n1d[12] = [0.4444030019441951, 1.3403751971516167, 2.2594644510007993, 3.2237098287700974, 4.271825847932282, 5.500901704467748]
    n1d[13] = [0.0, 0.8566190860303124, 1.7254183795882394, 2.6206899734322148, 3.5634443802816329, 4.590843711998803, 5.800167133570216]
    n1d[14] = [0.4124869684477138, 1.2426862814917308, 2.0887274297323704, 2.9630365798386679, 3.8869245750597002, 4.89527643078399, 6.087409546901289]
    n1d[15] = [0.0, 0.7991290683245479, 1.6067100690287302, 2.4324368270097586, 3.2890824243987665, 4.1962077112690155, 5.1880012243748714, 6.363947888829836]
    n1d[16] = [0.38676060450055735, 1.1638291005549648, 1.9519803457163335, 2.7602450476307, 3.6008486475106274, 4.4929553025200082, 5.4720713645743356, 6.6308781983931386]
    n1d[17] = [0.0, 0.7517952766485296, 1.5094452036688797, 2.2795070805010599, 3.0712086914116158, 3.8972489673146206, 4.7776168795438645, 5.7485814929244475, 6.8891945683788325]
    n1d[18] = [0.3655002483577753, 1.0983955180915013, 1.8397799215086046, 2.5958265797257616, 3.3747365357444902, 4.1880202316294039, 5.0540726854427482, 6.0184318609498935, 7.1394648491464795]
    n1d[19] = [0.0, 0.7120690072331, 1.4288766760783, 2.1556346027572, 2.8992903001629, 3.6698782611498, 4.4831325906584, 5.3650883850774, 6.35406823683, 7.38274]
    n1d[20] = [0.34696415708135, 1.042030920195, 1.7434844887806, 2.4573679234157, 3.1918687508395, 3.9585600046, 4.7754556886, 5.6700138926, 6.683, 7.62]
    n1d[21] = [0.0, 0.678, 1.358, 2.046, 2.748, 3.472, 4.229, 5.037, 5.929, 6.97, 7.85]
    n1d[22] = [0.331, 0.995, 1.663, 2.340, 3.034, 3.754, 4.513, 5.332, 6.250, 7.33, 8.08]
    n1d[23] = [0.0, 0.648, 1.298, 1.953, 2.620, 3.305, 4.015, 4.762, 5.568, 6.473, 7.58, 8.31]
    n1d[24] = [0.317, 0.951, 1.590, 2.236, 2.895, 3.573, 4.280, 5.027, 5.838, 6.756, 7.88, 8.54]
    n1d[25] = [0.0, 0.622, 1.245, 1.873, 2.509, 3.160, 3.831, 4.529, 5.266, 6.065, 6.968, 8.10, 8.76]
    return n1d
end

# Sparse grid Kronecker product for quadrature rules
# Input: n1d - vector of pointers to 1D nodes; w1d - vector of pointers to 1D weights
# Output: (nodes, weights) tuple
function SpGrKronProd(n1d::Vector{Vector{Float64}}, w1d::Vector{Vector{Float64}})
    nodes = n1d[1]
    weights = w1d[1]
    for j ∈ 2:length(n1d)
        nj = n1d[j]
        wj = w1d[j]
        # Kronecker product for nodes: extend each row with each element of nj
        nodes = hcat(repeat(nodes, outer=(length(nj), 1)),
                     kron(nj, ones(size(nodes, 1))))
        # Kronecker product for weights
        weights = kron(wj, weights)
    end
    return (nodes, weights)
end

# Sparse grids for dim-dimensional integration with accuracy level k
# Uses Kronecker-Patterson nested nodes for dim > 2
# Uses Gauss quadrature for dim <= 2 (more efficient for low dimensions)
function SpGr(dim::Int, k::Int)
    if dim <= 2
        # "Sparse" grids only sparser for dim > 2, use non-nested nodes
        n1d = GQNn1d()
        w1d = GQNw1d()
        nodes = n1d[k]
        weights = w1d[k]
        nodes = isnothing(nodes) ? Float64[] : nodes
        weights = isnothing(weights) ? Float64[] : weights
        # Add negative nodes (symmetric around 0)
        nodes = vcat(nodes, -nodes[1+mod(k, 2)+1:end])
        weights = vcat(weights, weights[1+mod(k, 2)+1:end])
        if dim == 1
            return (reshape(nodes, :, 1), weights)
        else
            # Kronecker square for 2D
            n_k = length(nodes)
            nodes_2d = hcat(repeat(nodes, outer=n_k), kron(nodes, ones(n_k)))
            weights_2d = kron(weights, weights)
            return (nodes_2d, weights_2d)
        end
    end

    n1d_all = KPNn1d()
    w1d_all = KPNw1d()

    nodes = zeros(0, dim)
    weights = Float64[]

    # Precompute number of rows for each 1D rule
    R1d = zeros(Int, 25)
    for r ∈ 1:25
        R1d[r] = isnothing(n1d_all[r]) ? 0 : length(n1d_all[r])
    end

    for q ∈ max(0, k - dim):k-1
        r = length(weights)
        bq = (2 * mod(k - q, 2) - 1) * binomial(dim - 1, dim + q - k)
        is = SpGrGetSeq(dim, dim + q)  # all rowvectors in N^D_{D+q}

        Rq = [R1d[is[i, 1]] for i ∈ 1:size(is, 1)]
        for j ∈ 2:dim
            Rq = Rq .* [R1d[is[i, j]] for i ∈ 1:size(is, 1)]
        end

        total_new = sum(Rq)
        nodes = vcat(nodes, zeros(total_new, dim))
        weights = vcat(weights, zeros(total_new))

        # Inner loop collecting product rules
        offset = r
        for j ∈ 1:size(is, 1)
            midx = is[j, :]
            # Get 1D nodes/weights for this combination
            n1d_j = [n1d_all[midx[i]] for i ∈ 1:dim]
            w1d_j = [w1d_all[midx[i]] for i ∈ 1:dim]

            # Skip if any are nothing
            if any(isnothing.(n1d_j)) || any(isnothing.(w1d_j))
                continue
            end

            newnw = SpGrKronProd(n1d_j, w1d_j)
            nrows = size(newnw[1], 1)
            nodes[offset+1:offset+nrows, :] = newnw[1]
            weights[offset+1:offset+nrows] = newnw[2] .* bq
            offset += nrows
        end

        # Combine identical nodes, summing weights
        if size(nodes, 1) > 1
            perm = sortperm(collect(eachrow(nodes)))
            nodes = nodes[perm, :]
            weights = weights[perm]

            # Find unique rows
            keep = vcat([any(nodes[i, :] .!= nodes[i+1, :]) for i ∈ 1:size(nodes, 1)-1], [true])

            # Running sum of weights, then difference
            cumweights = cumsum(weights)
            weights = cumweights[keep] .- vcat([0.0], cumweights[keep][1:end-1])
            nodes = nodes[keep, :]
        end
    end

    # Expand rules to other orthants (symmetric about 0 in each dimension)
    for j ∈ 1:dim
        keep_pos = nodes[:, j] .> 0
        if any(keep_pos)
            t = nodes[keep_pos, :]
            t[:, j] = -t[:, j]
            nodes = vcat(nodes, t)
            weights = vcat(weights, weights[keep_pos])
        end
    end

    return (nodes, weights)
end

# =============================================================================
# Rank-Ordered Probit Support: PermuteTies
# =============================================================================

# Given ranking potentially with ties, return matrix of all un-tied rankings consistent with it
# Each row is one valid ordering
function PermuteTies(v::AbstractVector)
    # Get sorted unique values and their positions
    sorted_unique = sort(unique(v))
    n = length(v)

    # Find groups of tied elements
    groups = [findall(v .== val) for val ∈ sorted_unique]

    # Generate all permutations recursively
    result = _PermuteTies(collect(1:n), groups, 1)
    return permutedims(hcat(result...), (2, 1))
end

function _PermuteTies(indices::Vector{Int}, groups::Vector{Vector{Int}}, group_idx::Int)
    if group_idx > length(groups)
        return [copy(indices)]
    end

    group = groups[group_idx]
    if length(group) <= 1
        return _PermuteTies(indices, groups, group_idx + 1)
    end

    # Generate all permutations of this group
    results = Vector{Int}[]
    for perm ∈ permutations(group)
        new_indices = copy(indices)
        for (i, p) ∈ enumerate(perm)
            new_indices[group[i]] = p
        end
        append!(results, _PermuteTies(new_indices, groups, group_idx + 1))
    end
    return results
end

# Simple permutations generator
function permutations(arr::AbstractVector)
    n = length(arr)
    if n <= 1
        return [arr]
    end
    result = Vector{eltype(arr)}[]
    for i ∈ 1:n
        rest = vcat(arr[1:i-1], arr[i+1:end])
        for p ∈ permutations(rest)
            push!(result, vcat([arr[i]], p))
        end
    end
    return result
end

# =============================================================================
# Bivariate Normal with Bounds (vecbinormal2)
# =============================================================================

# Compute binormal(E1,E2,rho)-binormal(E1,F2,rho) to maximize precision
# If midpoint between E2, F2 is >0, negate E2, F2, rho to take difference of smaller numbers
# infsign: whether to interpret missing in E1 as +infinity (true) or -infinity (false)
# flip: whether to swap indices 1 and 2
function vecbinormal2(E1::AbstractVector, E2::AbstractVector, F2::AbstractVector,
                      Sig::AbstractMatrix, infsign::Bool, flip::Bool,
                      one2N::AbstractVector, todo::Int)
    i1, i2 = flip ? (2, 1) : (1, 2)

    SigDiag = diag(Sig)[[i1, i2]]
    sqrtSigDiag = sqrt.(SigDiag)

    E1hat = E1 ./ sqrtSigDiag[1]
    E2hat = E2 ./ sqrtSigDiag[2]
    F2hat = F2 ./ sqrtSigDiag[2]
    rho = Sig[1, 2] / (sqrtSigDiag[1] * sqrtSigDiag[2])

    # Replace missing with large values
    E1hat_filled = replace(E1hat, NaN => infsign ? 1e6 : -1e6)
    E2hat_filled = replace(E2hat, NaN => 1e6)
    F2hat_filled = replace(F2hat, NaN => -1e6)

    Phi = binormal2(E1hat_filled, E2hat_filled, F2hat_filled, rho)

    dPhi_dE1 = nothing
    dPhi_dE2 = nothing
    dPhi_dF2 = nothing
    dPhi_dSig = nothing

    if todo != 0
        phiE1 = replace(normalden.(E1hat), NaN => 0.0)
        phiE2 = replace(normalden.(E2hat), NaN => 0.0)
        phiF2 = replace(normalden.(F2hat), NaN => 0.0)

        t = sqrt(1 - rho^2)
        E1hat_t = E1hat ./ t
        E2hat_t = E2hat ./ t
        F2hat_t = F2hat ./ t

        # Each with the other partialled out, then renormalized to s.d. 1
        E1E2hat1 = E1hat_t .- rho .* E2hat_t
        E1E2hat2 = E2hat_t .- rho .* E1hat_t
        E1F2hat1 = E1hat_t .- rho .* F2hat_t
        E1F2hat2 = F2hat_t .- rho .* E1hat_t

        detSig = sqrt(det(Sig))
        dPhi_dSigE = phiE1 .* replace(normalden.(E1E2hat2), NaN => 0.0) ./ detSig
        dPhi_dSigF = phiE1 .* replace(normalden.(E1F2hat2), NaN => 0.0) ./ detSig

        dPhi_dXE = hcat(phiE1, phiE2) .* hcat(
            replace(normcdf.(E1E2hat2), NaN => 1.0),
            replace(normcdf.(E1E2hat1), NaN => infsign ? 1.0 : 0.0)
        ) ./ sqrtSigDiag'

        dPhi_dXF = hcat(phiE1, phiF2) .* hcat(
            replace(normcdf.(E1F2hat2), NaN => 0.0),
            replace(normcdf.(E1F2hat1), NaN => infsign ? 1.0 : 0.0)
        ) ./ sqrtSigDiag'

        t_neg = -2 .* SigDiag'
        E1_filled = replace(E1, NaN => 0.0)
        E2_filled = replace(E2, NaN => 0.0)
        F2_filled = replace(F2, NaN => 0.0)

        dPhi_dSigDiagE = (hcat(E1_filled, E2_filled) .* dPhi_dXE .+ Sig[1, 2] .* dPhi_dSigE) ./ t_neg
        dPhi_dSigDiagF = (hcat(E1_filled, F2_filled) .* dPhi_dXF .+ Sig[1, 2] .* dPhi_dSigF) ./ t_neg

        dPhi_dSig = hcat(
            dPhi_dSigDiagE[:, i1] .- dPhi_dSigDiagF[:, i1],
            dPhi_dSigE .- dPhi_dSigF,
            dPhi_dSigDiagE[:, i2] .- dPhi_dSigDiagF[:, i2]
        )
        dPhi_dE1 = dPhi_dXE[:, 1] .- dPhi_dXF[:, 1]
        dPhi_dE2 = dPhi_dXE[:, 2]
        dPhi_dF2 = -dPhi_dXF[:, 2]
    end

    return Phi, dPhi_dE1, dPhi_dE2, dPhi_dF2, dPhi_dSig
end

# =============================================================================
# Derivative of Phi w.r.t. partialled-out errors and covariance (dPhi_dpE_dSig)
# =============================================================================

# Compute product of derivative of Phi w.r.t. partialled-out errors and derivative of
# partialled-out errors w.r.t. original covariance matrix
# Used as part of chain rule to transform scores for partialled-out errors and covariance
# into scores w.r.t. unpartialled ones
function dPhi_dpE_dSig!(E_out::AbstractMatrix, one2N::AbstractVector, beta::AbstractMatrix,
                        invSig_out::AbstractMatrix, Sig_out_in::AbstractMatrix,
                        dPhi_dpE::AbstractMatrix, lin::Int, lout::Int,
                        scores::AbstractMatrix, J_d_uncens_d_cens_0::AbstractMatrix)
    l = lin + lout

    # First part: scores w.r.t. sig_ij where both i,j are in are 0, skip those columns
    l = 1
    for j ∈ 1:lin
        l += lin - j + 1  # skip in-in columns

        # Scores w.r.t. sig_ij where i out and j in
        for i ∈ 1:lout
            neg_dbeta_dSig = copy(J_d_uncens_d_cens_0)
            neg_dbeta_dSig[:, j] = -invSig_out[:, i]
            if one2N == (:)
                scores[:, l] = vec(sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2))
            else
                scores[one2N, l] = vec(sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2))
            end
            l += 1
        end
    end

    # Second part: scores w.r.t. sig_ij where both i,j out
    for j ∈ 1:lout
        beta_j = beta[j, :]
        invSig_out_j = invSig_out[:, j]
        neg_dbeta_dSig = invSig_out_j * (invSig_out_j' * Sig_out_in)

        if one2N == (:)
            scores[:, l] = vec(sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2))
        else
            scores[one2N, l] = vec(sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2))
        end
        l += 1

        for i ∈ j+1:lout
            neg_dbeta_dSig = invSig_out[:, i] * beta_j' + invSig_out_j * beta[i, :]'
            if one2N == (:)
                scores[:, l] = vec(sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2))
            else
                scores[one2N, l] = vec(sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2))
            end
            l += 1
        end
    end

    return scores
end

# =============================================================================
# Post-Estimation Results (SaveSomeResults)
# =============================================================================

# Save some results to Stata after estimation
function SaveSomeResults!(model::CmpModel)
    L = model.L
    REs = model.REs

    # Save MprobitGroupInds and RoprobitGroupInds
    if !isempty(model.MprobitGroupInds)
        sf_store_matrix("e(MprobitGroupEqs)", model.MprobitGroupInds)
    end
    if !isempty(model.RoprobitGroupInds)
        sf_store_matrix("e(ROprobitGroupEqs)", model.RoprobitGroupInds)
    end

    # Save Sigma matrices for each level
    if L == 1
        sf_store_matrix("e(Sigma)", REs[1].Sig)
    else
        for l ∈ L:-1:1
            RE = REs[l]
            suffix = l < L ? string(l) : ""
            sf_store_matrix("e(Sigma" * suffix * ")", RE.Sig)

            # If adaptive quadrature was used, save means and SEs
            if l < L && model.Quadrature && (model.AdaptivePhaseThisEst || model.Adapted)
                means = zeros(RE.N, RE.d)
                ses = zeros(RE.N, RE.d)
                for j ∈ 1:RE.N
                    if !isempty(RE.QuadMean) && j <= length(RE.QuadMean)
                        means[j, :] = RE.QuadMean[j]
                    end
                    if !isempty(RE.QuadSD) && j <= length(RE.QuadSD)
                        ses[j, :] = RE.QuadSD[j]
                    end
                end
                sf_store_matrix("e(REmeans" * string(l) * ")", means * RE.T)
                sf_store_matrix("e(RESEs" * string(l) * ")", ses * RE.T)
            end
        end

        # If weights were used, store effective N
        if !isempty(model.WeightProduct)
            # sf_store_numscalar("e(N)", sum(model.WeightProduct))
        end
    end

    # Handle Gamma (reduced form) results if HasGamma
    if model.HasGamma
        # Complex reduced-form calculations would go here
        # This involves transforming structural parameters to reduced form
        # and computing appropriate covariance matrices
        # For now, this is a placeholder
    end

    return nothing
end

# Helper: store matrix in Stata (placeholder - actual implementation depends on interface)
function sf_store_matrix(name::String, M::AbstractMatrix)
    # This would use stataplugininterface to store in Stata
    # st_matrix(name, M)
    nothing
end

# =============================================================================
# Setter Functions with Logic
# =============================================================================

# Set whether model will use adaptive quadrature
function setWillAdapt!(model::CmpModel, t::Bool)
    model.WillAdapt = t
    model.Lastb = Float64[]  # Reset last parameter vector
    nothing
end

# Set GammaId matrix (for simultaneous equations)
function setGammaI!(model::CmpModel, t::AbstractMatrix)
    model.GammaId = t
    # Compute power of GammaId
    for i ∈ model.d-2:-1:1
        model.GammaId = model.GammaId * t
    end
    nothing
end

# Set Gamma index matrix (for identifying Gamma parameters)
function setGammaInd!(model::CmpModel, t::AbstractMatrix)
    model.GammaInd = t
    model.HasGamma = size(t, 1) > 0
    if model.HasGamma
        model.GammaIndByEq = Vector{Vector{Int}}(undef, model.d)
        for i ∈ 1:model.d
            rows_for_eq = findall(t[:, 2] .== i)
            model.GammaIndByEq[i] = t[rows_for_eq, 1]
        end
    end
    nothing
end

# Set Roprobit group indices and update count
function setRoprobitGroupInds!(model::CmpModel, t::AbstractMatrix)
    model.RoprobitGroupInds = t
    model.NumRoprobitGroups = size(t, 1)
    nothing
end

# Set number of cut points for ordered probit
function setvNumCuts!(model::CmpModel, t::AbstractVector{Int})
    model.vNumCuts = t
    model.NumCuts = sum(t)
    nothing
end

# Set variable names for indicators
function setindVars!(model::CmpModel, t::String)
    model.indVars = split(t)
    nothing
end

# Set variable names for yL (interval regression lower bounds)
function setyLVars!(model::CmpModel, t::String)
    model.yLVars = split(t)
    nothing
end

# Set variable names for Lt (truncation lower bounds)
function setLtVars!(model::CmpModel, t::String)
    model.LtVars = split(t)
    nothing
end

# Set variable names for Ut (truncation upper bounds)
function setUtVars!(model::CmpModel, t::String)
    model.UtVars = split(t)
    nothing
end

# Set model dimensions and initialize
function setd!(model::CmpModel, t::Int)
    model.d = t
    nothing
end

function setL!(model::CmpModel, t::Int)
    model.L = t
    nothing
end

function settodo!(model::CmpModel, t::Int)
    model._todo = t
    nothing
end

function setMaxCuts!(model::CmpModel, t::Int)
    model.MaxCuts = t
    nothing
end

function setReverse!(model::CmpModel, t::Bool)
    model.reverse = t
    nothing
end

function setSigXform!(model::CmpModel, t::Int)
    model.SigXform = t
    nothing
end

function setQuadTol!(model::CmpModel, t::Float64)
    model.QuadTol = t
    nothing
end

function setQuadIter!(model::CmpModel, t::Int)
    model.QuadIter = t
    nothing
end

function setGHKType!(model::CmpModel, t::String)
    model.ghkType = t
    nothing
end

function setGHKAnti!(model::CmpModel, t::Bool)
    model.ghkAnti = t
    nothing
end

function setGHKDraws!(model::CmpModel, t::Int)
    model.ghkDraws = t
    nothing
end

function getGHKDraws(model::CmpModel)
    return model.ghkDraws
end

function setGHKScramble!(model::CmpModel, t::String)
    scramble_types = ["", "sqrt", "negsqrt", "fl"]
    model.ghkScramble = findfirst(==(t), scramble_types) - 1
    nothing
end

function setREType!(model::CmpModel, t::String)
    model.REType = t
    nothing
end

function setREAnti!(model::CmpModel, t::Int)
    model.REAnti = t
    nothing
end

function setREScramble!(model::CmpModel, t::String)
    scramble_types = ["", "sqrt", "negsqrt", "fl"]
    model.REScramble = findfirst(==(t), scramble_types) - 1
    nothing
end

function setQuadrature!(model::CmpModel, t::Bool)
    model.Quadrature = t
    nothing
end

function setEqs!(model::CmpModel, t::AbstractMatrix)
    model.Eqs = t
    nothing
end

function setNumEff!(model::CmpModel, t::AbstractMatrix)
    model.NumEff = t
    nothing
end

function setMprobitGroupInds!(model::CmpModel, t::AbstractMatrix)
    model.MprobitGroupInds = t
    nothing
end

function setNonbaseCases!(model::CmpModel, t::AbstractVector{Int})
    model.NonbaseCases = t
    nothing
end

function settrunceqs!(model::CmpModel, t::AbstractVector{Int})
    model.trunceqs = t
    nothing
end

function setintregeqs!(model::CmpModel, t::AbstractVector{Int})
    model.intregeqs = t
    nothing
end

# =============================================================================
# Utility: vSigInds - map variable indices to vectorized covariance indices
# =============================================================================

# Given indices for variables and dimension of variance matrix,
# return corresponding indices in vectorized variance matrix
# e.g., (1,3) -> ((1,1), (3,1), (3,3)) -> (1, 3, 6)
function vSigInds(inds::AbstractVector{Int}, d::Int)
    if isempty(inds)
        return Int[]
    end

    # Create mapping from (i,j) to vech index
    d2 = d * (d + 1) ÷ 2
    mapping = zeros(Int, d, d)
    k = 1
    for j ∈ 1:d
        for i ∈ j:d
            mapping[i, j] = k
            mapping[j, i] = k
            k += 1
        end
    end

    # Extract indices for selected variables
    n = length(inds)
    result = Int[]
    for j ∈ 1:n
        for i ∈ j:n
            push!(result, mapping[inds[i], inds[j]])
        end
    end
    return result
end

# Insert row into matrix at specified position
function insert_row(X::AbstractMatrix, i::Int, newrow::AbstractVector)
    if i == 1
        return vcat(newrow', X)
    elseif i == size(X, 1) + 1
        return vcat(X, newrow')
    else
        return vcat(X[1:i-1, :], newrow', X[i:end, :])
    end
end

# Multiply matrix by vector, returning pointer-like reference for efficiency
function Xdotv(X::AbstractMatrix, v::AbstractVector)
    isempty(v) ? X : X .* v
end

end # module CMP
