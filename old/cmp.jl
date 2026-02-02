module CMP
# export :.

using LinearAlgebra, Kronecker, Combinatorics, SparseArrays
using Distributions 

@inline nrows(X::AbstractArray) = size(X,1)
@inline ncols(X::AbstractArray) = size(X,2)

const ind_t = UInt8
const eq_t = UInt8  # equation numbers
const iter_t = UInt32
const draw_t = UInt

const cmp_cont::ind_t = 1
const cmp_left::ind_t = 2
const cmp_right::ind_t = 3
const cmp_probit::ind_t = 4
const cmp_oprobit::ind_t = 5
const cmp_mprobit::ind_t = 6
const cmp_int::ind_t = 7
const cmp_probity1::ind_t = 8
const cmp_frac::ind_t = 10
const mprobit_ind_base::ind_t = 20
const roprobit_ind_base::ind_t = 40

mutable struct mprobit_group 
	d::eq_t; out::eq_t  # dimension - 1; eq of chosen alternative
	in::Vector{eq_t}; res::Vector{eq_t}  # eqs of remaining alternatives; indices in ECens to hold relative differences
end

mutable struct scores{T<::AbstractFloat}
	ThetaScores::Vector{eq_t}; CutScores::Vector{eq_t}  # in nonhierarchical models, vectors specifying relevant cols of master score matrix, S
	TScores::Vector{Matrix{T}}; SigScores::Vector{Matrix{T}}; ΓScores::Vector{Matrix{T}}  # SigScores only used at top level, to refer to cols of S. In hierarchical models, TScores[L] holds base Sig scores
end

mutable struct scorescol{T<::AbstractFloat}
	M::Vector{scores{T}}
end

mutable struct subview{T<::AbstractFloat}  # info associated with subsets of data defined by given combinations of indicator values
  EUncens::Matrix{T}
  ECens::Matrix{T}; F::Matrix{T}; Et::Matrix{T}; Ft::Matrix{T}
  Fi::Vector{T}  # temporary var used in lf1(); store here in case setcol(pX, Fi::.) leads to pX=&Fi and Fi should be preserved
	theta::Vector{Matrix{T}}; y::Vector{Matrix{T}}; Lt::Vector{Matrix{T}}; Ut::Vector{Matrix{T}}; yL::Vector{Matrix{T}};
	dΩdΓ::Matrix{Matrix{T}};
	Scores::Vector{scorescol{T}}  # one col for each level, one col for each draw
	Yi::Matrix{UInt16}
	subsample::Vector{Bool}; SubsampleInds::Vector{UInt64}
	d_uncens::eq_t; d_cens::eq_t; d2_cens::eq_t; d_two_cens::eq_t; d_oprobit::eq_t; d_trunc::eq_t; d_frac::eq_t; NFracCombs::UInt; N::UInt
	NumCuts::UInt  # number of cuts in ordered probit eqs relevant for *these* observations
	vNumCuts::Vector{UInt16}  # number of cuts per eq for the eq for *these* observations
	dSig_dLTSig::Matrix{T}  # derivative of Sig w.r.t. its lower triangle
	N_perm::UInt
	CensLTInds::Vector{UInt32}  # indexes of lower triangle of a vectorized square matrix of dimension d_cens
	WeightProduct::Vector{T}
	TheseInds::Vector{ind_t}  # user-provided indicator values
	uncens::Vector{eq_t}; two_cens::Vector{eq_t}; oprobit::Vector{eq_t}; cens::Vector{eq_t}; cens_nonrobase::Vector{eq_t}; trunc::Vector{eq_t}; frac::Vector{eq_t}; censnonfrac::Vector{eq_t}
	cens_uncens::Vector{eq_t}  # one_cens, oprobit, uncens
	SigIndsUncens::Vector{Int16}  # Indexes, within the vectorized upper triangle of Sig, entries for the eqs uncens at these obs
	SigIndsTrunc::Vector{Int16}  # Ditto for trunc obs
	SigIndsCensUncens::Vector{Int16}  # Permutes vectorized upper triangle of Sig to order corresponding to cens eqs first
	CutInds::Vector{Int16}  # Indexes, within full list of oprobit cuts, of those relevant for the equations in these observations
  NotBaseEq::Vector{eq_t}  # indicators of which eqs are not mprobit or roprobit base eqs
	QSig::Matrix{T}  # correction factor for trial cov matrix reflecting scores of passed "error" (XB,-XB,Y-XB, or XB-Y) w.r.t XB, and relative differencing
	Sig::Matrix{T}  # Sig, reflecting that correction
	Ω::Matrix{T}  # invΓ * Sig * invΓ' in Γ models. Slips into place of "Sig".
	QE::Matrix{T}  # correction factors for dlnL/dE
	QEinvΓ::Matrix{T}; invΓQSigD::Matrix{T}
	dCensNonrobase::eq_t
	dphi_dE::Matrix{T}; dPhi_dE::Matrix{T}; dPhi_dSig::Matrix{T}; dPhi_dcuts::Matrix{T}; dPhi_dF::Matrix{T}; dPhi_dpF::Matrix{T}; dPhi_dEt::Matrix{T}; dphi_dSig::Matrix{T}; dPhi_dSigt::Matrix{T}; dPhi_dpE_dSig::Matrix{T}; _dPhi_dpE_dSig::Matrix{T}; _dPhi_dpF_dSig::Matrix{T}; dPhi_dpF_dSig::Matrix{T}; EDE
	dPhi_dpE::Vector{Matrix{T}}; dPhi_dpSig::Vector{Matrix{T}}
	XU::Vector{Vector{Matrix{T}}}
	id::Vector{Matrix{UInt}}  # for each level, colvector of observation indexes that explodes group-level data to obs-level data, within this view
	roprobit_QE::Vector{Vector{T}}  # for each roprobit permutation, matrix that effects roprobit differencing of ECens columns
	roprobit_Q_Sig::Vector{Vector{T}}  # ditto for vech() of Sigma of censored E columns
	mprobit::Vector{mprobit_group}
	halfDmatrix::Matrix{T}
	FracCombs::Matrix{T}
	frac_QE::Vector{Matrix{T}}; frac_QSig::Vector{Matrix{T}}; yProd::Vector{Matrix{T}}  # all products of frac prob y's
	
	next::subview{T}
end

@enum Cov::Int8 unstructured=0 exchangeable=1 independent=2
@enum AdaptState::Int8 converged=0 reset=1 ordinary=2  # 0 = converged; 1 = adaptation needed having been reset because of divergence; 2 = ordinary adaptation needed

mutable struct RE{T<::AbstractFloat}  # info associated with given level of model. Top level also holds various globals as an alternative to storing them as separate externals, references to which are slow
	R::draw_t  # number of draws. (*REs)[l].R = NumREDraws[l+1]
	d::eq_t; d2::eq_t  # number of RCs and REs, corresponding triangular number
  JN1pQuadX::Vector{Matrix{T}}
	HasRC::Bool
	REInds::Vector{eq_t}  # indexes, within vector of effects, of random effects
	RCInds::Vector{Matrix{UInt}}  # for each equation, indexes of equation's set of random-coefficient effects within vector of effects
	Eqs::Vector{eq_t}  # indexes of equations in this level--for upper levels, ones with REs or RCs
	REEqs::Vector{eq_t}  # indexes of equations, within Eqs, with REs (as distinct from RCs)
	ΓEqs::Vector{eq_t}  # indexes of equations in this level that have REs or RCs or depend on them indirectly through Γ
	NEq::eq_t  # number of equations
	NEff::Vector{eq_t}  # number of effects/coefficients by equation, one entry for each eq that has any effects
	X::Vector{Matrix{T}}  # NEq-vector of data matrices for variables with random coefficients
	U::Vector{Matrix{T}}  # draws/observation-vector of N_g x d sets of draws
	XU::Matrix{Matrix{T}}  # draws/observation x d matrix of matrices of X, U products; coefficients on these, elements of T, set contribution to simulated error 
	TotalEffect::Matrix{Matrix{T}}  # matrices of, for each draw set and equation, total simulated effects at this level:: RE + RC*(RC vars)
	Sig::Matrix{T}; T::Matrix{T}; invΓ::Matrix{T}
	D::Matrix{T}  # derivative of vech(Sig) w.r.t lnsigs and atanhrhos
	dSigdParams::Matrix{T}  # derivative of sig, vech(ρ) vector w.r.t. vector of actual sig, ρ parameters, reflecting "exchangeable" and "independent" options
	NSigParams::UInt
  N::UInt  # number of groups at this level
	IDRanges::Matrix{UInt}  # id ranges for each group in data set, as returned by panelsetup()
	IDRangeLengths::Vector{UInt}  # lengths of those ranges
	IDRangesGroup::Matrix{UInt}  # N x 1, id ranges for each group's subgroups in the next level down
  Subscript::Vector{Matrix{UInt}}
	id::Matrix{UInt}  # group id var
	sig::Vector{T}; ρ::Vector{UInt}  # vector of error variances only, and atanhrho's
	covAcross::Cov; covWithin::Vector{Cov}  # cross- and within-eq covariance type:: unstructured, exchangeable, independent; indexed by *included* equations at this level
	FixedSigs::Vector{T}
	FixedRhos::Matrix{T}
	theta::Vector{Matrix{T}}
	Weights::Vector{T}  # weights at this level, one obs per group, renormalized if pweights or aweights
	ToAdapt::Vector{AdaptState}  # by group, state of adaptation attempt for this iteration. 2 = ordinary adaptation needed; 1 = adaptation needed having been reset because of divergence; 0 = converged
	lnNumREDraws::T
	lnLlimits::T
	lnLByDraw::Matrix{T}  # lnLByDraw acculumulates sums of them at next level up, by draw
	lnL::Matrix{T}  # lnL holds latest likelihoods at this level, points to 1e-6 lnf return arg at top level
	QuadW::Vector{T}; QuadX::Vector{T}  # quadrature weights
	QuadMean::Vector{Matrix{T}}; QuadSD::Vector{Matrix{T}}  # by group, estimated RE/RC mean and variance, for adaptive quadrature
	lnnormaldenQuadX::Vector{T}
	QuadXAdapt::Dict{Vector{Int64}, Vector{Matrix{Float64}}}  # asarray("real", l), one set of adaptive shifters per multi-level draw combination; first index is always 1, to prevent vector length 0
	AdaptivePhaseThisIter::Bool; AdaptiveShift::T
  Rho::Matrix{T}
  RCk::Vector{eq_t}  # number of X vars in each random coefficient
end

mutable struct cmp_model{T<::AbstractFloat}
	d::eq_t; L::eq_t; _todo::eq_t; SigXform::Bool; NumCuts::eq_t; MaxCuts::eq_t
	trunceqs::Vector{Bool}; intregeqs::Vector{Bool}
	reverse::Bool  # interpretation of ranking in roprobit
	NonbaseCases::Vector{Bool}
	NumRoprobitGroups::eq_t; RoprobitGroupInds::Matrix{eq_t}; MprobitGroupInds::Matrix{eq_t}
	indVars::Vector{String}; LtVars::Vector{String}; UtVars::Vector{String}; yLVars::Vector{String}
	QuadTol::T; QuadIter::iter_t
	HasΓ::Bool
	ΓId::Matrix{Bool}
	ΓIndByEq::Vector{Vector{eq_t}}  # d x 1 vector of pointers to rowvectors indicating which columns of Γ, for the given row, are real model parameters
	ΓInd::Matrix{eq_t}  # same information, in a 2-col matrix, each row the coordinates in Γ of a real parameter
	Eqs::Matrix{Bool}; NumEff::Matrix{eq_t}
	WillAdapt::Bool

	Adapted::Bool; AdaptivePhaseThisEst::Bool; AdaptNextTime::Bool

	REs::RE{T}; base::RE{T}
	subviews::subview{T}
	y::Vector{Matrix{T}}; Lt::Vector{Matrix{T}}; Ut::Vector{Matrix{T}}; yL::Vector{Matrix{T}}
	Theta::Matrix{T}  # individual theta's in one matrix
	NSimEff::eq_t
	NumREDraws::Vector{draw_t}
	Γ::Matrix{T}
  Ω::Matrix{T}
	dSig_dT::Matrix{T}  # derivative of vech(Sig) w.r.t vech(cholesky(Sig))
	WeightProduct::Vector{T}  # obs-level product of weights at all levels, for weighting scores
	d_cens::eq_t
	vNumCuts::Vector{eq_t}
	cuts::Matrix{T}
	G::Vector{eq_t}  # number of Γ params in each eq
	dΩdΓ::Matrix{Matrix{T}}
	Lastb::Vector{T}
	LastlnLLastIter::T; LastlnLThisIter::T; LastIter::iter_t
	Idd::Diagonal{Bool, Vector{Bool}}
	vKd::Vector{UInt16}; vIKI::Vector{UInt16}; vLd::Vector{UInt16}
	indicators::Matrix{ind_t}
	S0::Matrix{T}  # empty, pre-allocated matrix to build score matrix S
	Scores::scores{T}  # column indices in S corresponding to different parameter groups
	ThisDraw::Vector{draw_t}
	h::T  # if computing 2nd derivatives most recent h used
	X::Vector{Matrix{T}}  # NEq-vector of data matrices--needed only in gfX() estimation, to expand scores to one per regressor
  sTScores::Vector{Matrix{T}}; sΓScores::Vector{Matrix{T}}

	function cmp_model{T}(d, L, _todo, SigXform, NumCuts, MaxCuts, trunceqs, intregeqs, reverse, NonbaseCases,                            RoprobitGroupInds, MprobitGroupInds, indVars, LtVars, UtVars, yLVars, QuadTol, QuadIter, ΓI                              , ΓInd, Eqs, NumEff, WillAdapt) where T<::AbstractFloat
		HasΓ = nrows(ΓInd) > 0
		ΓIndByEq = HasΓ ? [ΓInd[ΓInd[:,2],1] for i ∈ 1:d] : [T[]]
	  new(                d, L, _todo, SigXform, NumCuts, MaxCuts, trunceqs, intregeqs, reverse, NonbaseCases, length(RoprobitGroupInds), RoprobitGroupInds, MprobitGroupInds, indVars, LtVars, UtVars, yLVars, QuadTol, QuadIter, ΓI^(d-1), HasΓ, ΓIndByEq, ΓInd, Eqs, NumEff, WillAdapt, false, false, false)
	end
end

function setNumREDraws(M::cmp_model, t::Vector)
	M.NumREDraws = [1; t]
	nothing
end

function setcol(X::Matrix, c::Vector{<::Integer}, v::Matrix)
  if ncols(X)==length(c)
    X = v
  else
    X[:,c] .= v
	end
	X
end

# assumes S already allocated
scoreAccum(S::Matrix, r, v::Vector, X::Matrix) = (S .= isone(r) ? v .* X : S + v .* X)

Xdotv(X::Matrix, v::Vector) = iszero(length(v)) ? X : X .* v

# insert row vector into a matrix at specified row
insert(X::Matrix, row, newrow::Vector) = @views [X[1:row-1,:]; newrow; X[row:end]]

# like Mata panelsetup() but can group on multiple columns, like sort(). But doesn't take minobs, maxobs arguments.
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

# fast(?) computation of a .+ quadrowsum(lnnormalden(X))
function quadrowsum_lnnormalden!(dest::Vector{T}, X::Matrix{T}, a::T) where T
	@inbounds Threads.@threads for i ∈ eachindex(axes(X,1))
		S = zero(T)
		@inbounds for j ∈ eachindex(axes(X,2))
			S += X[i,j] ^ 2
		end
		dest[i] = (a - T(log2π) * ncols(X)) - S / 2
	end
end
function quadrowsum_lnnormalden(X::Matrix{T}, a::T) where T
	dest = Vector{T}(undef, nrows(X))
	quadrowsum_lnnormalden!(dest,X,a)
end

# paste columns B into matrix A at starting index i, then advance index; for efficiency, overwrite A = B when possible
function PasteAndAdvance!(A::VecOrMat{T}, i<:Integer, B::VecOrMat{T}) where T 
	if ncols(B)>0
		t = i + ncols(B)
		if ncols(A) == ncols(B)
      A = B
    else
      A[:,i:t-1] .= B
		end
		i = t
	end
	A
end

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

# prepare matrix to transform scores w.r.t. elements of Sigma to ones w.r.t. lnsig's and ρ's
function dSigdsigrhos(SigXform::Bool, sig::Vector{T}, Sig::Matrix{T}, ρ::Vector{T}, Rho::Matrix{T}) where T 
	_d = length(sig); _d2 = _d + length(ρ)
	D = I(_d2)
	t = zeros(T, _d, _d)
	@inbounds for k ∈ 1:_d  # derivatives of Sigma w.r.t. lnsig's
		t2 = SigXform ? Sig[k,:] : _d>1 ? Rho[k,:].*sig : sig
		t[k,:] .= t2
		t[:,k] .+= t2
		vech!(view(D,:,k), t)
		t[k,:] .= zero(T)
		t[:,k] .= zero(T)
	end
	@inbounds for j ∈ 1:_d
		for i ∈ j+1:_d
			t[i,j] = sig[i] * sig[j]
			vech!(view(D,:,k), t)
			k += 1
			i[i,j] = zero(T)
		end
	end
	SigXform && (D[:,_d+1:end] ./= cosh.(ρ).^2)  # Datanh=cosh^2
	D
end

# make splat and pipe operators work together
import Base.|>
|>(args...) = args[end](args[1:end-1]...)  # https://stackoverflow.com/posts/58635140/revisions

# Given ranking potentially with ties, return matrix of all un-tied rankings consistent with it, one per row
function PermuteTies(v::Vector{T}) where T 
	p = sortperm(v)
	(panelsetup(v[p], [1]) .|> permutations)... |> Iterators.product .|> collect |> vec .|> x.vcat(x...) |> x.p[x]
end

# given indexes for variables, and dimension of variance matrix, return corresponding indexes in vectorized variance matrix
# e.g., (1,3) . ((1,1), (3,1), (3,3)) . (1, 3, 6)
vSigInds(inds::Vector{<:Integer}, d) = vech(invvech(1:Int(d*(d+1)÷2),d)[inds,inds])

function Lmatrix(d::Int)
	d2 = d*(d+1)÷2
	sparse(1:d2, [i*d+j for i ∈ 0:d-1 for j ∈ i+1:d], ones(Int, d2))
end

function Dmatrix(d::Int)
	d2 = d^2
	sparse(1:d2, [(c=min(i,j); max(i,j)+(2d-c-1)*c÷2 + 1) for i ∈ 0:d-1 for j ∈ 0:d-1], ones(Int, d2))
end

# Given transformation matrix for errors, return transformation matrix for vech(covar)
QE2QSig(QE::Matrix) = Lmatrix(ncols(QE))*(QE ⊗ QE)'Dmatrix(nrows(QE))

# integral of bivariate normal from -infinity to E1, F2 to E2, done to maximize precision as in normal2()
function binormal2(E1::Vector, E2::Matrix, F2::Matrix, ρ) 
	sign = @. E2 + F2 < 0
	@. sign = sign + sign - 1
	abs(binormalGenz(E1, sign.*E2, ρ, sign) - binormalGenz(E1, sign.*F2, ρ, sign))
end

# Based on Genz 2004 Fortran code, https:#web.archive.org/web/20180922125509/http:#www.math.wsu.edu/faculty/genz/software/fort77/tvpack.f
# Alan Genz, "Numerical computation of rectangular bivariate and trivariate normal and t probabilities," Statistics and Computing, August 2004, Volume 14, Issue 3, pp 251-60.
#
#    A function for computing bivariate normal probabilities.
#    This function is based on the method described by 
#        Drezner, Z and G.O. Wesolowsky, (1989), On the computation of the bivariate normal integral, Journal of Statist. Comput. Simul. 35, pp. 101-107,
#    with major modifications for double precision, and for |r| close to 1.
#
# Calculates the probability that X < x1 and Y < x2.
#
# Parameters
#   x1  integration limit
#   x2  integration limit
#   r   correlation coefficient
#   m   optional column vector of +/-1 multipliers for r

const sqrtτ = √(2π)
const sqrtτd12 = √(2π)/12
const inv2tau = 1/4π
function binormalGenz(x1::AbstractVector{T}, x2::AbstractVector{T}, r::T, m::AbstractVector{T}=Vector{T}(undef,0)) where T
	isnan(r) && return fill(nrows(x1),NaN)
	iszero(r) && return cdf.(Normal.(x1)).*cdf.(Normal.(x2))

	absr = abs(r)
	if absr < 0.925
		# Gauss Legendre Points and Weights
		if absr < 0.3
			_X = [-0.9324695142031522e+00, -0.6612093864662647e+00, -0.2386191860831970e+00]
			W =  [0.1713244923791705e+00,  0.3607615730481384e+00,  0.4679139345726904e+00 ]
		elseif absr < 0.75
			_X = [-0.9815606342467191e+00, -0.9041172563704750e+00, -0.7699026741943050e+00, -0.5873179542866171e+00, -0.3678314989981802e+00, -0.1252334085114692e+00]
			W =  [0.4717533638651177e-01,  0.1069393259953183e+00,  0.1600783285433464e+00,  0.2031674267230659e+00,  0.2334925365383547e+00,  0.2491470458134029e+00	]
		else 
			_X = [-0.9931285991850949e+00, -0.9639719272779138e+00, -0.9122344282513259e+00, -0.8391169718222188e+00, -0.7463319064601508e+00,
			    -0.6360536807265150e+00, -0.5108670019508271e+00, -0.3737060887154196e+00, -0.2277858511416451e+00, -0.7652652113349733e-01]

			W = [0.1761400713915212e-01,  0.4060142980038694e-01,  0.6267204833410906e-01,  0.8327674157670475e-01,  0.1019301198172404e+00,
			     0.1181945319615184e+00,  0.1316886384491766e+00,  0.1420961093183821e+00,  0.1491729864726037e+00,  0.1527533871307259e+00]
		end
		_X = [1 .- _X  1 .+ _X]
		W = [W W]

		HK = x1.*x2; length(m)>0 && (HK .*= m)
		HS = x1.*x1 .+ x2.*x2
		asinr = asin(r) 
		sn = sin.((asinr / 2) .* _X); sn2 = 2sn
		asinr = asinr * inv2tau
		return normal(x1) .* normal(x2) + quadrowsum(W .* exp((HK * sn2 .- HS) ./ (2. .- sn2 .* sn))) .* (nrows(m) ? m * asinr : asinr)
	end

	negx2 = -x2
	px2 = r<0 ? x2 : negx2

	if nrows(m)>0
		px2 .*= m
		normalx1    = normal( x1)
		normalx2    = normal( x2)
		normalnegx1 = normal(-x1)
		normalnegx2 = normal(negx2)
	end
	HK = x1 .* px2 ./ 2
	if absr < 1
		_X = -0.9931285991850949e+00, -0.9639719272779138e+00, -0.9122344282513259e+00, -0.8391169718222188e+00, -0.7463319064601508e+00,
		     -0.6360536807265150e+00, -0.5108670019508271e+00, -0.3737060887154196e+00, -0.2277858511416451e+00, -0.7652652113349733e-01
		W =   0.1761400713915212e-01,  0.4060142980038694e-01,  0.6267204833410906e-01,  0.8327674157670475e-01,  0.1019301198172404e+00,
		      0.1181945319615184e+00,  0.1316886384491766e+00,  0.1420961093183821e+00,  0.1491729864726037e+00,  0.1527533871307259e+00
		_X = [1 .- _X 1 .+ _X]
		W = [W W]

		as = (1-r)*(1+r)
		a = √as
		B = abs.(x1 .+ px2); BS = B .^ 2
		C = 2 .+ HK
		_D = 6 .+ HK
		asinr = HK - BS/2as
		retval = a * exp(asinr) .* (1 .- C.*(BS.-as).*(1/12 .- _D.*BS/480) + C.*_D.*(as^2 / 160)) -
		              exp(HK) .* normal(B/-a) .* B .* (sqrtτ .- C.*BS.*(sqrtτd12.-_D.*BS*#=(tau)/480=#0.0052221422388145835)) 

		a /= 2
		xs = (a .* _X) .^ 2
		rs = .√(1 .- xs)
		asinr = HK .- BS * 1 ./(2xs)
		retval = (retval + quadrowsum((a*W) .* (exp(asinr) .* ( exp(HK*((1 .- rs)./(1 .+ rs)))./rs - (1 .+ C*(xs*.25).*(1 .+ _D*(xs*.125))) ))))/-6.2831853071795862
		if nrows(m) > 0
			r<0 && return (m.<0).*(retval + rowmin((normalx1,normalx2))) - (m.>0).*(retval + (x1.>=negx2).*((x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))  # slow but max precision
			return        (m.>0).*(retval + rowmin((normalx1,normalx2))) - (m.<0).*((x1.>=negx2).*(retval + (x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))
		end
		r<0 && return (x1.>=negx2).*((x1.>x2).*(normal(x2)-normal(-x1))+(x1.<=x2).*(normal(x1)-normal(negx2))) - retval  # slow but max precision
		return retval + normal(rowmin((x1,x2)))
	end
	if nrows(m) > 0
		r<0 && return (m.<0).*(rowmin((normalx1,normalx2))) - (m.>0).*((x1.>=negx2).*((x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))  # slow but max precision
		return        (m.>0).*(rowmin((normalx1,normalx2))) - (m.<0).*((x1.>=negx2).*((x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))
	end
	r<0 && return (x1.>=negx2).*((x1.>x2).*(normal(x2)-normal(-x1))+(x1.<=x2).*(normal(x1)-normal(negx2)))  # slow but max precision
	return (normal(rowmin((x1,x2))))
end


# KPN nodes
const KPNnodes=[
[0.000000000000000000e+00],
[0.000000000000000000e+00,                                                  1.732050807568877193e+00],
[0.000000000000000000e+00,                                                  1.732050807568877193e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                                                                                                      4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,                                                  4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
module MP
# export :.

using LinearAlgebra, Kronecker, Combinatorics, SparseArrays
using Distributions 

@inline nrows(X::AbstractArray) = size(X,1)
@inline ncols(X::AbstractArray) = size(X,2)

const ind_t = UInt8
const eq_t = UInt8  # equation numbers
const iter_t = UInt32
const draw_t = UInt

const cmp_cont::ind_t = 1
const cmp_left::ind_t = 2
const cmp_right::ind_t = 3
const cmp_probit::ind_t = 4
const cmp_oprobit::ind_t = 5
const cmp_mprobit::ind_t = 6
const cmp_int::ind_t = 7
const cmp_probity1::ind_t = 8
const cmp_frac::ind_t = 10
const mprobit_ind_base::ind_t = 20
const roprobit_ind_base::ind_t = 40

mutable struct mprobit_group 
	d::eq_t; out::eq_t  # dimension - 1; eq of chosen alternative
	in::Vector{eq_t}; res::Vector{eq_t}  # eqs of remaining alternatives; indices in ECens to hold relative differences
end

mutable struct scores{T<::AbstractFloat}
	ThetaScores::Vector{eq_t}; CutScores::Vector{eq_t}  # in nonhierarchical models, vectors specifying relevant cols of master score matrix, S
	TScores::Vector{Matrix{T}}; SigScores::Vector{Matrix{T}}; ΓScores::Vector{Matrix{T}}  # SigScores only used at top level, to refer to cols of S. In hierarchical models, TScores[L] holds base Sig scores
end

mutable struct scorescol{T<::AbstractFloat}
	M::Vector{scores{T}}
end

mutable struct subview{T<::AbstractFloat}  # info associated with subsets of data defined by given combinations of indicator values
  EUncens::Matrix{T}
  ECens::Matrix{T}; F::Matrix{T}; Et::Matrix{T}; Ft::Matrix{T}
  Fi::Vector{T}  # temporary var used in lf1(); store here in case setcol(pX, Fi::.) leads to pX=&Fi and Fi should be preserved
	theta::Vector{Matrix{T}}; y::Vector{Matrix{T}}; Lt::Vector{Matrix{T}}; Ut::Vector{Matrix{T}}; yL::Vector{Matrix{T}};
	dΩdΓ::Matrix{Matrix{T}};
	Scores::Vector{scorescol{T}}  # one col for each level, one col for each draw
	Yi::Matrix{UInt16}
	subsample::Vector{Bool}; SubsampleInds::Vector{UInt64}
	d_uncens::eq_t; d_cens::eq_t; d2_cens::eq_t; d_two_cens::eq_t; d_oprobit::eq_t; d_trunc::eq_t; d_frac::eq_t; NFracCombs::UInt; N::UInt
	NumCuts::UInt  # number of cuts in ordered probit eqs relevant for *these* observations
	vNumCuts::Vector{UInt16}  # number of cuts per eq for the eq for *these* observations
	dSig_dLTSig::Matrix{T}  # derivative of Sig w.r.t. its lower triangle
	N_perm::UInt
	CensLTInds::Vector{UInt32}  # indexes of lower triangle of a vectorized square matrix of dimension d_cens
	WeightProduct::Vector{T}
	TheseInds::Vector{ind_t}  # user-provided indicator values
	uncens::Vector{eq_t}; two_cens::Vector{eq_t}; oprobit::Vector{eq_t}; cens::Vector{eq_t}; cens_nonrobase::Vector{eq_t}; trunc::Vector{eq_t}; frac::Vector{eq_t}; censnonfrac::Vector{eq_t}
	cens_uncens::Vector{eq_t}  # one_cens, oprobit, uncens
	SigIndsUncens::Vector{Int16}  # Indexes, within the vectorized upper triangle of Sig, entries for the eqs uncens at these obs
	SigIndsTrunc::Vector{Int16}  # Ditto for trunc obs
	SigIndsCensUncens::Vector{Int16}  # Permutes vectorized upper triangle of Sig to order corresponding to cens eqs first
	CutInds::Vector{Int16}  # Indexes, within full list of oprobit cuts, of those relevant for the equations in these observations
  NotBaseEq::Vector{eq_t}  # indicators of which eqs are not mprobit or roprobit base eqs
	QSig::Matrix{T}  # correction factor for trial cov matrix reflecting scores of passed "error" (XB,-XB,Y-XB, or XB-Y) w.r.t XB, and relative differencing
	Sig::Matrix{T}  # Sig, reflecting that correction
	Ω::Matrix{T}  # invΓ * Sig * invΓ' in Γ models. Slips into place of "Sig".
	QE::Matrix{T}  # correction factors for dlnL/dE
	QEinvΓ::Matrix{T}; invΓQSigD::Matrix{T}
	dCensNonrobase::eq_t
	dphi_dE::Matrix{T}; dPhi_dE::Matrix{T}; dPhi_dSig::Matrix{T}; dPhi_dcuts::Matrix{T}; dPhi_dF::Matrix{T}; dPhi_dpF::Matrix{T}; dPhi_dEt::Matrix{T}; dphi_dSig::Matrix{T}; dPhi_dSigt::Matrix{T}; dPhi_dpE_dSig::Matrix{T}; _dPhi_dpE_dSig::Matrix{T}; _dPhi_dpF_dSig::Matrix{T}; dPhi_dpF_dSig::Matrix{T}; EDE
	dPhi_dpE::Vector{Matrix{T}}; dPhi_dpSig::Vector{Matrix{T}}
	XU::Vector{Vector{Matrix{T}}}
	id::Vector{Matrix{UInt}}  # for each level, colvector of observation indexes that explodes group-level data to obs-level data, within this view
	roprobit_QE::Vector{Vector{T}}  # for each roprobit permutation, matrix that effects roprobit differencing of ECens columns
	roprobit_Q_Sig::Vector{Vector{T}}  # ditto for vech() of Sigma of censored E columns
	mprobit::Vector{mprobit_group}
	halfDmatrix::Matrix{T}
	FracCombs::Matrix{T}
	frac_QE::Vector{Matrix{T}}; frac_QSig::Vector{Matrix{T}}; yProd::Vector{Matrix{T}}  # all products of frac prob y's
	
	next::subview{T}
end

@enum Cov::Int8 unstructured=0 exchangeable=1 independent=2
@enum AdaptState::Int8 converged=0 reset=1 ordinary=2  # 0 = converged; 1 = adaptation needed having been reset because of divergence; 2 = ordinary adaptation needed

mutable struct RE{T<::AbstractFloat}  # info associated with given level of model. Top level also holds various globals as an alternative to storing them as separate externals, references to which are slow
	R::draw_t  # number of draws. (*REs)[l].R = NumREDraws[l+1]
	d::eq_t; d2::eq_t  # number of RCs and REs, corresponding triangular number
  JN1pQuadX::Vector{Matrix{T}}
	HasRC::Bool
	REInds::Vector{eq_t}  # indexes, within vector of effects, of random effects
	RCInds::Vector{Matrix{UInt}}  # for each equation, indexes of equation's set of random-coefficient effects within vector of effects
	Eqs::Vector{eq_t}  # indexes of equations in this level--for upper levels, ones with REs or RCs
	REEqs::Vector{eq_t}  # indexes of equations, within Eqs, with REs (as distinct from RCs)
	ΓEqs::Vector{eq_t}  # indexes of equations in this level that have REs or RCs or depend on them indirectly through Γ
	NEq::eq_t  # number of equations
	NEff::Vector{eq_t}  # number of effects/coefficients by equation, one entry for each eq that has any effects
	X::Vector{Matrix{T}}  # NEq-vector of data matrices for variables with random coefficients
	U::Vector{Matrix{T}}  # draws/observation-vector of N_g x d sets of draws
	XU::Matrix{Matrix{T}}  # draws/observation x d matrix of matrices of X, U products; coefficients on these, elements of T, set contribution to simulated error 
	TotalEffect::Matrix{Matrix{T}}  # matrices of, for each draw set and equation, total simulated effects at this level:: RE + RC*(RC vars)
	Sig::Matrix{T}; T::Matrix{T}; invΓ::Matrix{T}
	D::Matrix{T}  # derivative of vech(Sig) w.r.t lnsigs and atanhrhos
	dSigdParams::Matrix{T}  # derivative of sig, vech(ρ) vector w.r.t. vector of actual sig, ρ parameters, reflecting "exchangeable" and "independent" options
	NSigParams::UInt
  N::UInt  # number of groups at this level
	IDRanges::Matrix{UInt}  # id ranges for each group in data set, as returned by panelsetup()
	IDRangeLengths::Vector{UInt}  # lengths of those ranges
	IDRangesGroup::Matrix{UInt}  # N x 1, id ranges for each group's subgroups in the next level down
  Subscript::Vector{Matrix{UInt}}
	id::Matrix{UInt}  # group id var
	sig::Vector{T}; ρ::Vector{UInt}  # vector of error variances only, and atanhrho's
	covAcross::Cov; covWithin::Vector{Cov}  # cross- and within-eq covariance type:: unstructured, exchangeable, independent; indexed by *included* equations at this level
	FixedSigs::Vector{T}
	FixedRhos::Matrix{T}
	theta::Vector{Matrix{T}}
	Weights::Vector{T}  # weights at this level, one obs per group, renormalized if pweights or aweights
	ToAdapt::Vector{AdaptState}  # by group, state of adaptation attempt for this iteration. 2 = ordinary adaptation needed; 1 = adaptation needed having been reset because of divergence; 0 = converged
	lnNumREDraws::T
	lnLlimits::T
	lnLByDraw::Matrix{T}  # lnLByDraw acculumulates sums of them at next level up, by draw
	lnL::Matrix{T}  # lnL holds latest likelihoods at this level, points to 1e-6 lnf return arg at top level
	QuadW::Vector{T}; QuadX::Vector{T}  # quadrature weights
	QuadMean::Vector{Matrix{T}}; QuadSD::Vector{Matrix{T}}  # by group, estimated RE/RC mean and variance, for adaptive quadrature
	lnnormaldenQuadX::Vector{T}
	QuadXAdapt::Dict{Vector{Int64}, Vector{Matrix{Float64}}}  # asarray("real", l), one set of adaptive shifters per multi-level draw combination; first index is always 1, to prevent vector length 0
	AdaptivePhaseThisIter::Bool; AdaptiveShift::T
  Rho::Matrix{T}
  RCk::Vector{eq_t}  # number of X vars in each random coefficient
end

mutable struct cmp_model{T<::AbstractFloat}
	d::eq_t; L::eq_t; _todo::eq_t; SigXform::Bool; NumCuts::eq_t; MaxCuts::eq_t
	trunceqs::Vector{Bool}; intregeqs::Vector{Bool}
	reverse::Bool  # interpretation of ranking in roprobit
	NonbaseCases::Vector{Bool}
	NumRoprobitGroups::eq_t; RoprobitGroupInds::Matrix{eq_t}; MprobitGroupInds::Matrix{eq_t}
	indVars::Vector{String}; LtVars::Vector{String}; UtVars::Vector{String}; yLVars::Vector{String}
	QuadTol::T; QuadIter::iter_t
	HasΓ::Bool
	ΓId::Matrix{Bool}
	ΓIndByEq::Vector{Vector{eq_t}}  # d x 1 vector of pointers to rowvectors indicating which columns of Γ, for the given row, are real model parameters
	ΓInd::Matrix{eq_t}  # same information, in a 2-col matrix, each row the coordinates in Γ of a real parameter
	Eqs::Matrix{Bool}; NumEff::Matrix{eq_t}
	WillAdapt::Bool

	Adapted::Bool; AdaptivePhaseThisEst::Bool; AdaptNextTime::Bool

	REs::RE{T}; base::RE{T}
	subviews::subview{T}
	y::Vector{Matrix{T}}; Lt::Vector{Matrix{T}}; Ut::Vector{Matrix{T}}; yL::Vector{Matrix{T}}
	Theta::Matrix{T}  # individual theta's in one matrix
	NSimEff::eq_t
	NumREDraws::Vector{draw_t}
	Γ::Matrix{T}
  Ω::Matrix{T}
	dSig_dT::Matrix{T}  # derivative of vech(Sig) w.r.t vech(cholesky(Sig))
	WeightProduct::Vector{T}  # obs-level product of weights at all levels, for weighting scores
	d_cens::eq_t
	vNumCuts::Vector{eq_t}
	cuts::Matrix{T}
	G::Vector{eq_t}  # number of Γ params in each eq
	dΩdΓ::Matrix{Matrix{T}}
	Lastb::Vector{T}
	LastlnLLastIter::T; LastlnLThisIter::T; LastIter::iter_t
	Idd::Diagonal{Bool, Vector{Bool}}
	vKd::Vector{UInt16}; vIKI::Vector{UInt16}; vLd::Vector{UInt16}
	indicators::Matrix{ind_t}
	S0::Matrix{T}  # empty, pre-allocated matrix to build score matrix S
	Scores::scores{T}  # column indices in S corresponding to different parameter groups
	ThisDraw::Vector{draw_t}
	h::T  # if computing 2nd derivatives most recent h used
	X::Vector{Matrix{T}}  # NEq-vector of data matrices--needed only in gfX() estimation, to expand scores to one per regressor
  sTScores::Vector{Matrix{T}}; sΓScores::Vector{Matrix{T}}

	function cmp_model{T}(d, L, _todo, SigXform, NumCuts, MaxCuts, trunceqs, intregeqs, reverse, NonbaseCases,                            RoprobitGroupInds, MprobitGroupInds, indVars, LtVars, UtVars, yLVars, QuadTol, QuadIter, ΓI                              , ΓInd, Eqs, NumEff, WillAdapt) where T<::AbstractFloat
		HasΓ = nrows(ΓInd) > 0
		ΓIndByEq = HasΓ ? [ΓInd[ΓInd[:,2],1] for i ∈ 1:d] : [T[]]
	  new(                d, L, _todo, SigXform, NumCuts, MaxCuts, trunceqs, intregeqs, reverse, NonbaseCases, length(RoprobitGroupInds), RoprobitGroupInds, MprobitGroupInds, indVars, LtVars, UtVars, yLVars, QuadTol, QuadIter, ΓI^(d-1), HasΓ, ΓIndByEq, ΓInd, Eqs, NumEff, WillAdapt, false, false, false)
	end
end

function setNumREDraws(M::cmp_model, t::Vector)
	M.NumREDraws = [1; t]
	nothing
end

function setcol(X::Matrix, c::Vector{<::Integer}, v::Matrix)
  if ncols(X)==length(c)
    X = v
  else
    X[:,c] .= v
	end
	X
end

# assumes S already allocated
scoreAccum(S::Matrix, r, v::Vector, X::Matrix) = (S .= isone(r) ? v .* X : S + v .* X)

Xdotv(X::Matrix, v::Vector) = iszero(length(v)) ? X : X .* v

# insert row vector into a matrix at specified row
insert(X::Matrix, row, newrow::Vector) = @views [X[1:row-1,:]; newrow; X[row:end]]

# like Mata panelsetup() but can group on multiple columns, like sort(). But doesn't take minobs, maxobs arguments.
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

# fast(?) computation of a .+ quadrowsum(lnnormalden(X))
const logτ = log(2π)
function quadrowsum_lnnormalden!(dest::Vector{T}, X::Matrix{T}, a::T) where T
	@inbounds Threads.@threads for i ∈ eachindex(axes(X,1))
		S = zero(T)
		@inbounds for j ∈ eachindex(axes(X,2))
			S += X[i,j] ^ 2
		end
		dest[i] = (a - T(log2π) * ncols(X)) - S / 2
	end
end
function quadrowsum_lnnormalden(X::Matrix{T}, a::T) where T
	dest = Vector{T}(undef, nrows(X))
	quadrowsum_lnnormalden!(dest,X,a)
end

# paste columns B into matrix A at starting index i, then advance index; for efficiency, overwrite A = B when possible
function PasteAndAdvance!(A::VecOrMat{T}, i<:Integer, B::VecOrMat{T}) where T 
	if ncols(B)>0
		t = i + ncols(B)
		if ncols(A) == ncols(B)
      A = B
    else
      A[:,i:t-1] .= B
		end
		i = t
	end
	A
end

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

# prepare matrix to transform scores w.r.t. elements of Sigma to ones w.r.t. lnsig's and ρ's
function dSigdsigrhos(SigXform::Bool, sig::Vector{T}, Sig::Matrix{T}, ρ::Vector{T}, Rho::Matrix{T}) where T 
	_d = length(sig); _d2 = _d + length(ρ)
	D = I(_d2)
	t = zeros(T, _d, _d)
	@inbounds for k ∈ 1:_d  # derivatives of Sigma w.r.t. lnsig's
		t2 = SigXform ? Sig[k,:] : _d>1 ? Rho[k,:].*sig : sig
		t[k,:] .= t2
		t[:,k] .+= t2
		vech!(view(D,:,k), t)
		t[k,:] .= zero(T)
		t[:,k] .= zero(T)
	end
	@inbounds for j ∈ 1:_d
		for i ∈ j+1:_d
			t[i,j] = sig[i] * sig[j]
			vech!(view(D,:,k), t)
			k += 1
			i[i,j] = zero(T)
		end
	end
	SigXform && (D[:,_d+1:end] ./= cosh.(ρ).^2)  # Datanh=cosh^2
	D
end

# make splat and pipe operators work together
import Base.|>
|>(args...) = args[end](args[1:end-1]...)  # https://stackoverflow.com/posts/58635140/revisions

# Given ranking potentially with ties, return matrix of all un-tied rankings consistent with it, one per row
function PermuteTies(v::Vector{T}) where T 
	p = sortperm(v)
	(panelsetup(v[p], [1]) .|> permutations)... |> Iterators.product .|> collect |> vec .|> x.vcat(x...) |> x.p[x]
end

# given indexes for variables, and dimension of variance matrix, return corresponding indexes in vectorized variance matrix
# e.g., (1,3) . ((1,1), (3,1), (3,3)) . (1, 3, 6)
vSigInds(inds::Vector{<:Integer}, d) = vech(invvech(1:Int(d*(d+1)÷2),d)[inds,inds])

function Lmatrix(d::Int)
	d2 = d*(d+1)÷2
	sparse(1:d2, [i*d+j for i ∈ 0:d-1 for j ∈ i+1:d], ones(Int, d2))
end

function Dmatrix(d::Int)
	d2 = d^2
	sparse(1:d2, [(c=min(i,j); max(i,j)+(2d-c-1)*c÷2 + 1) for i ∈ 0:d-1 for j ∈ 0:d-1], ones(Int, d2))
end

# Given transformation matrix for errors, return transformation matrix for vech(covar)
QE2QSig(QE::Matrix) = Lmatrix(ncols(QE))*(QE ⊗ QE)'Dmatrix(nrows(QE))

# integral of bivariate normal from -infinity to E1, F2 to E2, done to maximize precision as in normal2()
function binormal2(E1::Vector, E2::Matrix, F2::Matrix, ρ) 
	sign = @. E2 + F2 < 0
	@. sign = sign + sign - 1
	abs(binormalGenz(E1, sign.*E2, ρ, sign) - binormalGenz(E1, sign.*F2, ρ, sign))
end

# Based on Genz 2004 Fortran code, https:#web.archive.org/web/20180922125509/http:#www.math.wsu.edu/faculty/genz/software/fort77/tvpack.f
# Alan Genz, "Numerical computation of rectangular bivariate and trivariate normal and t probabilities," Statistics and Computing, August 2004, Volume 14, Issue 3, pp 251-60.
#
#    A function for computing bivariate normal probabilities.
#    This function is based on the method described by 
#        Drezner, Z and G.O. Wesolowsky, (1989), On the computation of the bivariate normal integral, Journal of Statist. Comput. Simul. 35, pp. 101-107,
#    with major modifications for double precision, and for |r| close to 1.
#
# Calculates the probability that X < x1 and Y < x2.
#
# Parameters
#   x1  integration limit
#   x2  integration limit
#   r   correlation coefficient
#   m   optional column vector of +/-1 multipliers for r

const sqrtτ = √(2π)
const sqrtτd12 = √(2π)/12
const inv2tau = 1/4π
function binormalGenz(x1::AbstractVector{T}, x2::AbstractVector{T}, r::T, m::AbstractVector{T}=Vector{T}(undef,0)) where T
	isnan(r) && return fill(nrows(x1),NaN)
	iszero(r) && return cdf.(Normal.(x1)).*cdf.(Normal.(x2))

	absr = abs(r)
	if absr < 0.925
		# Gauss Legendre Points and Weights
		if absr < 0.3
			_X = [-0.9324695142031522e+00, -0.6612093864662647e+00, -0.2386191860831970e+00]
			W =  [0.1713244923791705e+00,  0.3607615730481384e+00,  0.4679139345726904e+00 ]
		elseif absr < 0.75
			_X = [-0.9815606342467191e+00, -0.9041172563704750e+00, -0.7699026741943050e+00, -0.5873179542866171e+00, -0.3678314989981802e+00, -0.1252334085114692e+00]
			W =  [0.4717533638651177e-01,  0.1069393259953183e+00,  0.1600783285433464e+00,  0.2031674267230659e+00,  0.2334925365383547e+00,  0.2491470458134029e+00	]
		else 
			_X = [-0.9931285991850949e+00, -0.9639719272779138e+00, -0.9122344282513259e+00, -0.8391169718222188e+00, -0.7463319064601508e+00,
			    -0.6360536807265150e+00, -0.5108670019508271e+00, -0.3737060887154196e+00, -0.2277858511416451e+00, -0.7652652113349733e-01]

			W = [0.1761400713915212e-01,  0.4060142980038694e-01,  0.6267204833410906e-01,  0.8327674157670475e-01,  0.1019301198172404e+00,
			     0.1181945319615184e+00,  0.1316886384491766e+00,  0.1420961093183821e+00,  0.1491729864726037e+00,  0.1527533871307259e+00]
		end
		_X = [1 .- _X  1 .+ _X]
		W = [W W]

		HK = x1.*x2; nrows(m)>0 && (HK .*= m)
		HS = x1.*x1 .+ x2.*x2
		asinr = asin(r) 
		sn = sin.((asinr / 2) .* _X); sn2 = 2 * sn
		asinr = asinr * inv2tau
		return normal(x1) .* normal(x2) + quadrowsum(W .* exp((HK * sn2 .- HS) ./ (2. .- sn2 .* sn))) .* (nrows(m) ? m * asinr : asinr)
	end

	negx2 = -x2
	px2 = r<0 ? x2 : negx2

	if nrows(m)>0
		px2 .*= m
		normalx1    = normal( x1)
		normalx2    = normal( x2)
		normalnegx1 = normal(-x1)
		normalnegx2 = normal(negx2)
	end
	HK = x1 .* px2 ./ 2
	if absr < 1
		_X = -0.9931285991850949e+00, -0.9639719272779138e+00, -0.9122344282513259e+00, -0.8391169718222188e+00, -0.7463319064601508e+00,
		     -0.6360536807265150e+00, -0.5108670019508271e+00, -0.3737060887154196e+00, -0.2277858511416451e+00, -0.7652652113349733e-01
		W =   0.1761400713915212e-01,  0.4060142980038694e-01,  0.6267204833410906e-01,  0.8327674157670475e-01,  0.1019301198172404e+00,
		      0.1181945319615184e+00,  0.1316886384491766e+00,  0.1420961093183821e+00,  0.1491729864726037e+00,  0.1527533871307259e+00
		_X = [1 .- _X 1 .+ _X]
		W = [W W]

		as = (1-r)*(1+r)
		a = √as
		B = abs.(x1 .+ px2); BS = B .^ 2
		C = 2 .+ HK
		_D = 6 .+ HK
		asinr = HK - BS/2as
		retval = a * exp(asinr) .* (1 .- C.*(BS.-as).*(1/12 .- _D.*BS/480) + C.*_D.*(as^2 / 160)) -
		              exp(HK) .* normal(B/-a) .* B .* (sqrtτ .- C.*BS.*(sqrtτd12.-_D.*BS*#=(tau)/480=#0.0052221422388145835)) 

		a /= 2
		xs = (a .* _X) .^ 2
		rs = .√(1 .- xs)
		asinr = HK .- BS * 1 ./(2xs)
		retval = (retval + quadrowsum((a*W) .* (exp(asinr) .* ( exp(HK*((1 .- rs)./(1 .+ rs)))./rs - (1 .+ C*(xs*.25).*(1 .+ _D*(xs*.125))) ))))/-6.2831853071795862
		if nrows(m) > 0
			r<0 && return (m.<0).*(retval + rowmin((normalx1,normalx2))) - (m.>0).*(retval + (x1.>=negx2).*((x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))  # slow but max precision
			return        (m.>0).*(retval + rowmin((normalx1,normalx2))) - (m.<0).*((x1.>=negx2).*(retval + (x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))
		end
		r<0 && return (x1.>=negx2).*((x1.>x2).*(normal(x2)-normal(-x1))+(x1.<=x2).*(normal(x1)-normal(negx2))) - retval  # slow but max precision
		return retval + normal(rowmin((x1,x2)))
	end
	if nrows(m) > 0
		r<0 && return (m.<0).*(rowmin((normalx1,normalx2))) - (m.>0).*((x1.>=negx2).*((x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))  # slow but max precision
		return        (m.>0).*(rowmin((normalx1,normalx2))) - (m.<0).*((x1.>=negx2).*((x1.>x2).*(normalnegx1-normalx2)+(x1.<=x2).*(normalnegx2-normalx1)))
	end
	r<0 && return (x1.>=negx2).*((x1.>x2).*(normal(x2)-normal(-x1))+(x1.<=x2).*(normal(x1)-normal(negx2)))  # slow but max precision
	return (normal(rowmin((x1,x2))))
end


# KPN nodes
const KPNnodes=[
[0.000000000000000000e+00],
[0.000000000000000000e+00,                                                  1.732050807568877193e+00],
[0.000000000000000000e+00,                                                  1.732050807568877193e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                                                                                                      4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.732050807568877193e+00,                                                                           2.861279576057058183e+00,                                                  4.184956017672732287e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,                                                  4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,                         7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,                         2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,                         4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,                         5.187016039913656229e+00,                         6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,                         5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00],
[0.000000000000000000e+00,2.489922975799606086e-01,7.410953499945408529e-01,1.230423634027306035e+00,1.732050807568877193e+00,2.233626061676941887e+00,2.596083115049202306e+00,2.861279576057058183e+00,3.205333794499194422e+00,3.635318519037278318e+00,4.184956017672732287e+00,4.736433085952296729e+00,5.187016039913656229e+00,5.698177768488109862e+00,6.363394494336369611e+00,7.122106700804616608e+00,7.980771798590560628e+00,9.016939789890303203e+00]]
const KPNweights=[
[1.000000000000000000e+00],
[6.666666666666666297e-01,1.666666666666666574e-01],
[6.666666666666667407e-01,1.666666666666666574e-01],
[4.587448682574918890e-01,1.313786069831356096e-01,1.385532747297492373e-01,6.956841583691398666e-04],
[2.539682539682540652e-01,2.700743295779377551e-01,9.485094850948512513e-02,7.996325470893529339e-03,9.426945755651747010e-05],
[2.539682539682542872e-01,2.700743295779377551e-01,9.485094850948506962e-02,7.996325470893529339e-03,9.426945755651755141e-05],
[2.539682539682541762e-01,2.700743295779378106e-01,9.485094850948501410e-02,7.996325470893531073e-03,9.426945755651759207e-05],
[2.539682539682541762e-01,2.700743295779378106e-01,9.485094850948504186e-02,7.996325470893527604e-03,9.426945755651737523e-05],
[2.669222303350530234e-01,2.545612320417122154e-01,1.419265482644936453e-02,8.868100215202801007e-02,1.965677093877749200e-03,7.033480237827907482e-03,1.056378361541694140e-04,-8.204920754150921686e-07,2.113649950542425693e-08],
[3.034671998542062266e-01,2.083249916496087706e-01,6.115173012524771634e-02,6.409605468680761031e-02,1.808523425479846222e-02,-6.337224793373757124e-03,2.884880436506755911e-03,6.012336945984799651e-05,6.094808731468984020e-07,8.629684602229863183e-10],
[3.034671998542062266e-01,2.083249916496087151e-01,6.115173012524770940e-02,6.409605468680754092e-02,1.808523425479845875e-02,-6.337224793373754522e-03,2.884880436506755477e-03,6.012336945984792197e-05,6.094808731468982961e-07,8.629684602229883863e-10],
[3.034671998542062266e-01,2.083249916496087151e-01,6.115173012524771634e-02,6.409605468680762419e-02,1.808523425479846569e-02,-6.337224793373754522e-03,2.884880436506755911e-03,6.012336945984784065e-05,6.094808731468982961e-07,8.629684602229896270e-10],
[3.034671998542060045e-01,2.083249916496088261e-01,6.115173012524773022e-02,6.409605468680763807e-02,1.808523425479845875e-02,-6.337224793373757992e-03,2.884880436506755477e-03,6.012336945984786776e-05,6.094808731468982961e-07,8.629684602229875591e-10],
[3.034671998542061711e-01,2.083249916496087428e-01,6.115173012524770246e-02,6.409605468680759643e-02,1.808523425479845875e-02,-6.337224793373756257e-03,2.884880436506755477e-03,6.012336945984793552e-05,6.094808731468985079e-07,8.629684602229832164e-10],
[3.034671998542061155e-01,2.083249916496087428e-01,6.115173012524772328e-02,6.409605468680765195e-02,1.808523425479845875e-02,-6.337224793373759726e-03,2.884880436506756345e-03,6.012336945984809137e-05,6.094808731468985079e-07,8.629684602229898338e-10],
[2.589000532415156597e-01,2.812810154003316659e-02,1.996886351173454976e-01,6.541739283609256106e-02,6.171853256586717906e-02,1.760847558131800172e-03,1.659249269893601011e-02,-5.561006306835815718e-03,2.729843046733400162e-03,1.504420539091421892e-05,5.947496116393162150e-05,6.143584323261791332e-07,7.929826786486933825e-10,5.115805310550420830e-12,-1.484083574029886795e-13,1.261846428081511810e-15],
[1.391102223633803869e-01,1.038768712557428392e-01,1.760759874157145910e-01,7.744360274629948082e-02,5.467755614346304222e-02,7.353011020495507644e-03,1.152924706539878996e-02,-2.771218900778924313e-03,2.120225955959632522e-03,8.323604529576674469e-05,5.569115898108147934e-05,6.908626117911373775e-07,-1.348601734854293015e-08,1.554219599278265799e-09,-1.934130500088095548e-11,2.664062516623165063e-13,-9.931391328682246513e-16],
[5.148945080692137691e-04,1.917601158880443413e-01,1.480708311552158540e-01,9.236472671698635339e-02,4.527368546515039144e-02,1.567347375185115105e-02,3.155446269187551275e-03,2.311345240352207127e-03,8.189539275022673492e-04,2.752421411678513123e-04,3.572934819897533216e-05,2.734220680118788814e-06,2.467642134579814009e-07,2.139419447956106222e-08,4.601176034865591675e-10,3.097222357606299486e-12,5.450041265063812808e-15,1.054132658233401363e-18],
[5.148945080692137691e-04,1.917601158880443690e-01,1.480708311552158540e-01,9.236472671698635339e-02,4.527368546515052328e-02,1.567347375185115105e-02,3.155446269187560382e-03,2.311345240352204958e-03,8.189539275022666986e-04,2.752421411678513123e-04,3.572934819897544736e-05,2.734220680118788390e-06,2.467642134579814009e-07,2.139419447956105560e-08,4.601176034865607702e-10,3.097222357606301101e-12,5.450041265063766265e-15,1.054132658233795794e-18],
[5.148945080692555109e-04,1.917601158880443968e-01,1.480708311552158540e-01,9.236472671698629788e-02,4.527368546515053715e-02,1.567347375185115452e-02,3.155446269187557346e-03,2.311345240352207994e-03,8.189539275022672407e-04,2.752421411678513665e-04,3.572934819897535249e-05,2.734220680118788814e-06,2.467642134579812421e-07,2.139419447956105560e-08,4.601176034865614423e-10,3.097222357606296255e-12,5.450041265063836474e-15,1.054132658233540223e-18],
[5.148945080691374413e-04,1.917601158880442858e-01,1.480708311552159373e-01,9.236472671698631176e-02,4.527368546515039144e-02,1.567347375185115105e-02,3.155446269187556479e-03,2.311345240352208862e-03,8.189539275022666986e-04,2.752421411678514208e-04,3.572934819897528473e-05,2.734220680118788814e-06,2.467642134579811891e-07,2.139419447956105891e-08,4.601176034865659401e-10,3.097222357606295043e-12,5.450041265063869606e-15,1.054132658233204148e-18],
[5.148945080690336832e-04,1.917601158880444800e-01,1.480708311552157430e-01,9.236472671698642278e-02,4.527368546515051634e-02,1.567347375185116146e-02,3.155446269187554310e-03,2.311345240352206259e-03,8.189539275022671323e-04,2.752421411678516376e-04,3.572934819897531861e-05,2.734220680118790508e-06,2.467642134579815068e-07,2.139419447956108207e-08,4.601176034865600464e-10,3.097222357606304333e-12,5.450041265063759165e-15,1.054132658233992624e-18],
[5.148945080691375497e-04,1.917601158880444245e-01,1.480708311552157708e-01,9.236472671698638115e-02,4.527368546515046777e-02,1.567347375185115452e-02,3.155446269187556045e-03,2.311345240352204525e-03,8.189539275022657229e-04,2.752421411678515834e-04,3.572934819897529828e-05,2.734220680118789237e-06,2.467642134579812950e-07,2.139419447956107215e-08,4.601176034865610287e-10,3.097222357606296255e-12,5.450041265063820696e-15,1.054132658233836816e-18],
[5.148945080691443802e-04,1.917601158880444245e-01,1.480708311552157708e-01,9.236472671698633952e-02,4.527368546515050940e-02,1.567347375185115452e-02,3.155446269187558647e-03,2.311345240352205826e-03,8.189539275022655060e-04,2.752421411678514208e-04,3.572934819897538637e-05,2.734220680118788390e-06,2.467642134579808185e-07,2.139419447956105891e-08,4.601176034865638204e-10,3.097222357606294235e-12,5.450041265063838051e-15,1.054132658233694105e-18],
[5.148945080691998914e-04,1.917601158880443690e-01,1.480708311552157985e-01,9.236472671698639503e-02,4.527368546515042613e-02,1.567347375185115799e-02,3.155446269187553877e-03,2.311345240352205392e-03,8.189539275022668071e-04,2.752421411678514208e-04,3.572934819897529151e-05,2.734220680118788390e-06,2.467642134579810832e-07,2.139419447956105560e-08,4.601176034865590124e-10,3.097222357606297466e-12,5.450041265063841207e-15,1.054132658233752653e-18]]

const KPNd = lengths.(KPNnodes)


# SpGr(dim, k): function for generating nodes & weights for nested sparse grids integration with Gaussian weights
# dim  : dimension of the integration problem
# k    : Accuracy level. The rule will be exact for polynomial up to total order 2k-1
# Returns 1x2 vector of pointers to matrices: nodes and weights
# correspond to Heiss and Winschel GQN & KPN types
# Adapted with permission from Florian Heiss & Viktor Winschel, https:#web.archive.org/web/20181007012445/http:#sparse-grids.de/stata/build_nwspgr.do.
# Sources: Florian Heiss and Viktor Winschel, "Likelihood approximation by numerical integration on sparse grids", Journal of Econometrics 144(1): 62-80.
#          A. Genz and B. D. Keister (1996): "Fully symmetric interpolatory rules for multiple integrals over infinite regions with Gaussian weight." Journal of Computational and Applied Mathematics 71, 299-309.
function SpGr(dim, k) 
	if dim ≤ 2  # "sparse" grids only sparser for dim > 2
		nodes, weights = gausshermite(k) .* (√2, 1/√π)  # for normal pdf, not exp(-x^2)
		return isone(dim) ? (            nodes                   , weights          ) :
		                    (ones(k,1) ⊗ nodes, nodes ⊗ ones(k,1), weights ⊗ weights)  # Kronecker square of non-nested nodes
	end
	
	nodes = weights = Matrix{Float64}(undef,0,dim)

	for q ∈ max(0,k-dim):k-1
		r = nrows(weights)
		bq = (2*mod(k-q, 2)-1) * binomial(dim-1, dim+q-k)
		is = SpGrGetSeq(dim, dim+q)  # matrix of all rowvectors in N^D_D+qend
		Rq = mapreduce(c.KPNd[c], .*, eachcol(is))
		nodes   = [nodes   ; Matrix{Float64}(undef,sum(Rq), dim)]
		weights = [weights ; Matrix{Float64}(undef,sum(Rq), dim)]

		# inner loop collecting product rules
		for (j,midx) ∈ enumerate(eachrow(is))
			newnw = SpGrKronProd(KPNnodes[midx], KPNweights[midx])
			nodes[  r+1:r+Rq[j], :] .= newnw[1]
			weights[r+1:r+Rq[j]   ] .= newnw[2] .* bq 
			r += Rq[j]
		end
		
		# combine identical nodes, summing weights
		if nrows(nodes) > 1
			sortvec = sortperm(collect(eachrow(nodes)))
			nodes   = nodes[sortvec,:]
			weights = weights[sortvec]
			keep = [any.(eachrow(nodes[1:end-1,:] .≠ nodes[2:end,:])) ; true]
			weights = cumsum(weights)[keep]
			weights[2:end] .-= @view weights[1:end-1]
			nodes = nodes[keep]
		end
	end

	# 2. expand rules to other orthants
	for (j,keep) ∈ enumerate(eachcol(nodes))
		if any(keep)
			t = nodes[keep]
			t[:,j] .= .-t[:,j]
			nodes   = [nodes   ; t]
			weights = [weights ; weights[keep]]
		end
	end

	nodes, weights
end

# SpGrGetSeq(): generate all d-length sequences of positive integers summing to norm, one sequence per row
SpGrGetSeq(d, norm) = d==1 ? norm : [[norm-d+1 fill(1,1,d-1)]; vcat([[fill(i, binomial(norm-i-1,d-2), 1) SpGrGetSeq(d-1, norm-i)] for i ∈ 1:norm-d]...)]


# SpGrKronProd(): generate tensor product quadrature rule 
# Input: 
#     n1d : vector of pointers to 1D nodes 
#     n1d : vector of pointers to 1D weights 
# Output:
#     out  = pair of pointers to nodes and weights
function SpGrKronProd(n1d::Vector{Vector{T}}, w1d::Vector{Vector{T}}) where T
	nodes, weights = n1d[1], w1d[1]
	
	for j ∈ 2:length(n1d)
			# Create tensor product by repeating existing nodes and combining with new nodes
			nodes = [repeat(nodes, outer=length(n1d[j])) repeat(n1d[j], inner=length(nodes))]
			# Multiply weights for tensor product
			weights = w1d[j] ⊗ weights
	end
	
	nodes, weights
end

# vectorize binormal(). Accepts general covariance matrix, not just ρ parameter. Optionally computes scores.
function vecbinormal(p::Matrix, Sig::Matrix, todo::Integer, dPhi_dX::Matrix, dPhi_dSig::Matrix) 
	Xhat = X ./ (sqrtSigDiag = .√(SigDiag = diag(Sig)'))
	ρ = Sig[1,2] / (sqrtSigDiag[1] * sqrtSigDiag[2])
	Φ = binormalGenz(@view Xhat[:,1], @view Xhat[:,2], ρ)

	if todo > 0
		ϕ = pdf.(Normal(), Xhat)
		X_ = Xhat * ([1 -ρ ; -ρ 1] / √(1 - ρ * ρ))  # each X_ with the other partialled out, then renormalized to s.d. 1
		X_1, X_2 = @view X_[:,1], @view X_[:,2]
		_dPhi_dSig .= ϕ[:,1] .* pdf.(Normal(), X_2) ./ √det(Sig)
		dPhi_dX .= ϕ .* cdf.(Normal(), @view X_[:,[2,1]]) ./ sqrtSigDiag
		dPhi_dSigDiag = (X .* dPhi_dX .+ Sig[1,2] .* _dPhi_dSig) ./ (-2 .* SigDiag) 
		dPhi_dSig .= [dPhi_dSigDiag[:,1] _dPhi_dSig dPhi_dSigDiag[:,2]]
	end
	Φ
end

# Define editmissing function to match Mata behavior
function editmissing(x::Union{Vector{Float64}, Vector{Union{Float64, Missing}}}, value::Float64)
    return replace(x, missing => value)
end

function vecbinormal2(E1::Vector{Float64}, E2::Vector{Float64}, F2::Vector{Float64}, 
                      Sig::Matrix{Float64}, infsign::Bool, flip::Bool, todo::Bool,
                      dPhi_dE1::Vector{Float64}, dPhi_dE2::Vector{Float64}, 
                      dPhi_dF2::Vector{Float64}, dPhi_dSig::Matrix{Float64})
    
    # Set indices based on flip parameter
    if flip
        i1, i2 = 2, 1
    else
        i1, i2 = 1, 2
    end
    
    # Extract and compute diagonal elements
    SigDiag = [Sig[1,1], Sig[2,2]][[i1, i2]]
    sqrtSigDiag = sqrt.(SigDiag)
    
    # Normalize inputs
    E1hat = E1 ./ sqrtSigDiag[1]
    E2hat = E2 ./ sqrtSigDiag[2]
    F2hat = F2 ./ sqrtSigDiag[2]
    ρ = Sig[1,2] / (sqrtSigDiag[1] * sqrtSigDiag[2])
    
    # Compute binormal difference based on infsign
    if infsign
        Φ = binormal2.(editmissing(E1hat, 1e6), 
                       editmissing(E2hat, 1e6), 
                       editmissing(F2hat, -1e6), ρ)
    else
        Φ = binormal2.(editmissing(E1hat, -1e6), 
                       editmissing(E2hat, 1e6), 
                       editmissing(F2hat, -1e6), ρ)
    end
    
    if todo
        # Compute probability density functions
        phiE1 = editmissing(normalden.(E1hat), 0.0)
        phiE2 = editmissing(normalden.(E2hat), 0.0)
        phiF2 = editmissing(normalden.(F2hat), 0.0)
        
        t = sqrt(1 - ρ^2)
        E1hat = E1hat ./ t
        E2hat = E2hat ./ t
        F2hat = F2hat ./ t
        
        # Compute conditional expectations
        E1E2hat1 = E1hat .- ρ .* E2hat
        E1E2hat2 = E2hat .- ρ .* E1hat
        E1F2hat1 = E1hat .- ρ .* F2hat
        E1F2hat2 = F2hat .- ρ .* E1hat
        
        # Compute derivatives
        t_det = sqrt(det(Sig))
        dPhi_dSigE = phiE1 .* editmissing(normalden.(E1E2hat2), 0.0) ./ t_det
        dPhi_dSigF = phiE1 .* editmissing(normalden.(E1F2hat2), 0.0) ./ t_det
        
        dPhi_dXE = hcat(phiE1, phiE2) .* 
                   hcat(editmissing(normalcdf.(E1E2hat2), 1.0),
                        editmissing(normalcdf.(E1E2hat1), Float64(infsign))) ./ sqrtSigDiag'
        
        dPhi_dXF = hcat(phiE1, phiF2) .* 
                   hcat(editmissing(normalcdf.(E1F2hat2), 0.0),
                        editmissing(normalcdf.(E1F2hat1), Float64(infsign))) ./ sqrtSigDiag'
        
        t_denom = -2 .* SigDiag
        dPhi_dSigDiagE = (editmissing(hcat(E1, E2), 0.0) .* dPhi_dXE .+ 
                          (Sig[1,2] .* dPhi_dSigE)) ./ t_denom'
        dPhi_dSigDiagF = (editmissing(hcat(E1, F2), 0.0) .* dPhi_dXF .+ 
                          (Sig[1,2] .* dPhi_dSigF)) ./ t_denom'
        
        # Modify passed-in parameters using .=
        dPhi_dSig .= hcat(dPhi_dSigDiagE[:, i1] .- dPhi_dSigDiagF[:, i1],
                          dPhi_dSigE .- dPhi_dSigF,
                          dPhi_dSigDiagE[:, i2] .- dPhi_dSigDiagF[:, i2])
        
        dPhi_dE1 .= dPhi_dXE[:, 1] .- dPhi_dXF[:, 1]
        dPhi_dE2 .= dPhi_dXE[:, 2]
        dPhi_dF2 .= -dPhi_dXF[:, 2]
    end
    
    return Φ
end

# Helper functions that would need to be defined:
# normalden: probability density function of standard normal distribution
normalden(x) = exp(-x^2/2) / sqrt(2π)

# normalcdf: cumulative distribution function of standard normal distribution
normalcdf(x) = cdf(Normal(), x)


# neg_half_E_Dinvsym_E() -- compute -0.5 * inner product of given errors weighting by derivative of inverse of a symmetric matrix 
# Passed +/- E times the inverse of X. Returns a matrix with one column for each of the N(N+1)/2 independent entries in X.
function neg_half_E_Dinvsym_E(E_invX::Matrix{Float64}, EDE::Matrix{Float64})
    N = size(E_invX, 2)
    if N > 0
        l = size(EDE, 2)
        E_invX_j = E_invX[:, N]
        EDE[:, l] .= E_invX_j .* E_invX_j .* 0.5
        l -= 1
        
        for j = N-1:-1:1
            E_invX_j = E_invX[:, j]
            # In Julia, we use views and broadcasting for the assignment
            EDE[:, (l-N+j+1):l] .= view(E_invX, :, (j+1):N) .* E_invX_j
            l = l - N + j
            EDE[:, l] .= E_invX_j .* E_invX_j .* 0.5
            l -= 1
        end
    end
    return EDE
end

# Compute product of derivative of Φ w.r.t. partialled-out errors (provided) and derivative of partialled-out errors w.r.t. 
# original covariance matrix. Used as part of an application of the chain rule to transform the initial scores for Φ
# w.r.t. the partialled-out errors and covariance matrix into scores w.r.t. the un-partialled ones.
# Returns a score matrix with one row for each observation and one column for each element of the lower triangle of
# Var[in | out], ordered by the lists in parameters "in" and "out". E.g. if in=(1,3) and out=(2), then the column 
# order corresponds to (1,1),(1,3),(1,2),(3,3),(3,2),(2,2)
function dPhi_dpE_dSig(E_out::Matrix{Float64}, beta::Matrix{Float64}, invSig_out::Matrix{Float64}, 
                       Sig_out_in::Matrix{Float64}, dPhi_dpE::Matrix{Float64}, lin::Int, lout::Int, 
                       scores::Matrix{Float64}, J_d_uncens_d_cens_0::Matrix{Float64})
    
    l = 1
    
    for j = 1:lin
        # scores w.r.t. sig_ij where both i,j are in are 0, so skip those columns in score matrix
        l = l + lin - j + 1
        # scores w.r.t. sig_ij where i out and j in
        for i = 1:lout
            neg_dbeta_dSig = J_d_uncens_d_cens_0
            neg_dbeta_dSig[:, j] .= -invSig_out[:, i]
            scores[:, l] .= sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2)
            l += 1
        end
    end
    
    # scores w.r.t. sig_ij where both i,j out
    for j = 1:lout
        beta_j = beta[j, :]
        invSig_out_j = invSig_out[:, j]
        neg_dbeta_dSig = invSig_out_j * (invSig_out_j' * Sig_out_in)
        scores[:, l] .= sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2)
        l += 1
        for i = (j+1):lout
            neg_dbeta_dSig = invSig_out[:, i] * beta_j' + invSig_out_j * beta[i, :]'
            scores[:, l] .= sum(dPhi_dpE .* (E_out * neg_dbeta_dSig), dims=2)
            l += 1
        end
    end
    
    return scores
end

# (log) likelihood and scores for cumulative multivariate normal for a vector of observations of upper bounds and optional lower bounds
# i.e., computes multivariate normal cdf over L_1<=x_1<=U_1, L_2<=x_2<=U_2, :., where some L_i's can be negative infinity
# Argument -bounded- indicates which dimensions have lower bounds as well as upper bounds.
# If argument log>0, returns Φ, not log Φ
# returns scores if requested in dPhi_dE, dPhi_dF, dPhi_dSig. dPhi_dF must already be allocated
function vecmultinormal(E::Matrix{Float64}, F::Matrix{Float64}, Sig::Matrix{Float64}, d::Int, 
                       bounded::Vector{Float64}, todo::Bool, dPhi_dE::Matrix{Float64}, 
                       dPhi_dF::Matrix{Float64}, dPhi_dSig::Matrix{Float64}, N_perm::Int)
    
    # Initialize variables
    local dPhi_dE1, dPhi_dE2, dPhi_dF1, dPhi_dF2, _dPhi_dF2, _dPhi_dE1, _dPhi_dF1, _dPhi_dSig, dM
    local Φ::Vector{Float64}
    
    if d == 1
        sqrtSig = sqrt(Sig[1,1])
        if length(bounded) > 0
            # Compute log difference of CDFs
            Φ = exp.(logdiffcdf.(Normal(), view(F, :, 1) ./ sqrtSig, view(E, :, 1) ./ sqrtSig))
            if todo
                if N_perm == 1
                    dPhi_dE .= editmissing(normalden.(E, 0.0, sqrtSig), 0.0) ./ Φ
                    dPhi_dF .= -editmissing(normalden.(F, 0.0, sqrtSig), 0.0) ./ Φ
                end
                dPhi_dSig .= (sum(dPhi_dE .* E, dims=2) + sum(dPhi_dF .* F, dims=2)) ./ (-2 * Sig[1,1])
            end
        else
            Φ = cdf.(Normal(), E ./ sqrtSig)
            if todo
                if N_perm == 1
                    dPhi_dE .= editmissing(normalden.(E, 0.0, sqrtSig), 0.0) ./ Φ
                end
                dPhi_dSig .= dPhi_dE .* E ./ (-2 * Sig[1,1])
            end
        end
        if N_perm == 1
            return log.(Φ)
        end
        return Φ
    end
    
    if d == 2
        if length(bounded) > 0
            if bounded[1] == 1
                E1 = E[:, 1]
                F1 = F[:, 1]
                Φ = vecbinormal2(E[:, 2], E1, F1, Sig, true, true, todo, dPhi_dE2, dPhi_dE1, dPhi_dF1, dPhi_dSig)
                
                if bounded == [1]
                    if todo
                        dPhi_dE .= hcat(dPhi_dE1, dPhi_dE2)
                        dPhi_dF .= hcat(dPhi_dF1, zeros(size(E, 1), 1))
                    end
                else  # rectangular region integration
                    Φ_temp = vecbinormal2(F[:, 2], E1, F1, Sig, false, true, todo, _dPhi_dF2, _dPhi_dE1, _dPhi_dF1, _dPhi_dSig)
                    Φ = Φ .- Φ_temp
                    if todo
                        dPhi_dE .= hcat(dPhi_dE1 .- _dPhi_dE1, dPhi_dE2)
                        dPhi_dF .= hcat(dPhi_dF1 .- _dPhi_dF1, -_dPhi_dF2)
                        dPhi_dSig .= dPhi_dSig .- _dPhi_dSig
                    end
                end
            else
                Φ = vecbinormal2(E[:, 1], E[:, 2], F[:, 2], Sig, true, false, todo, dPhi_dE1, dPhi_dE2, dPhi_dF2, dPhi_dSig)
                if todo
                    dPhi_dE .= hcat(dPhi_dE1, dPhi_dE2)
                    dPhi_dF .= hcat(zeros(size(E, 1), 1), dPhi_dF2)
                end
            end
        else
            Φ = vecbinormal(E, Sig, :, todo, dPhi_dE, dPhi_dSig)
        end
    else
        if length(bounded) > 0
            if @isdefined(ghk2DrawSet) && ghk2DrawSet !== nothing
                if todo
                    Φ = _ghk2_2d(ghk2DrawSet, F, E, Sig, ghkAnti, GHKStart, dPhi_dF, dPhi_dE, dPhi_dSig)
                else
                    Φ = _ghk2_2(ghk2DrawSet, F, E, Sig, ghkAnti, GHKStart)
                end
            else
                Φ = _mvnormalcv(F, E, zeros(1, size(E, 2)), vech(Sig)')
                if todo
                    _mvnormalcvderiv(F, E, zeros(1, size(E, 2)), vech(Sig)', dPhi_dF, dPhi_dE, dM, dPhi_dSig)
                end
            end
        else
            if @isdefined(ghk2DrawSet) && ghk2DrawSet !== nothing
                if todo
                    Φ = _ghk2_d(ghk2DrawSet, E, Sig, ghkAnti, GHKStart, dPhi_dE, dPhi_dSig)
                else
                    Φ = _ghk2(ghk2DrawSet, E, Sig, ghkAnti, GHKStart)
                end
            else
                Φ = _mvnormalcv(fill(-Inf, 1, size(E, 2)), E, zeros(1, size(E, 2)), vech(Sig)')
                if todo
                    _mvnormalcvderiv(fill(quantile(Normal(), Φ * 1e-20), 1, size(E, 2)), E, 
                                   zeros(1, size(E, 2)), vech(Sig)', dPhi_dF, dPhi_dE, dM, dPhi_dSig)
                end
            end
        end
    end
    
    if N_perm == 1
        if todo
            dPhi_dE .= dPhi_dE ./ Φ
            dPhi_dSig .= dPhi_dSig ./ Φ
            if length(bounded) > 0
                dPhi_dF .= dPhi_dF ./ Φ
            end
        end
        return log.(Φ)
    end
    
    Φ
end

# Helper functions
_mvnormalcv(a, b, c, d) = mvnormalcv(a, b, c, d)
_mvnormalcvderiv(a, b, c, d, e, f, g, h) = mvnormalcvderiv(a, b, c, d, e, f, g, h)


function logdiffcdf(dist, a, b)
    # Compute log(CDF(a) - CDF(b)) in a numerically stable way
    return log.(cdf(dist, a) - cdf(dist, b))
end


real colvector _mvnormalcv(a,b,c,d) return (mvnormalcv(a,b,c,d))
void _mvnormalcvderiv(a,b,c,d,e,f,g,h) return (mvnormalcvderiv(a,b,c,d,e,f,g,h))

# compute the log likelihood associated with a given error data matrix, for "continuous" variables
# Sig is the assumed covariance for the full error set and inds marks the observed variables assumed to have a joint normal distribution,
# i.e., the ones not censored
# dphi_dE should already be allocated
function lnLContinuous(v::SubView, todo::Bool)
    # Compute C as inverse of upper triangular Cholesky factor
    C = inv(cholesky(v.Ω[v.uncens, v.uncens]).U)
    
    # Compute log density
    ϕ = quadrowsum_lnnormalden(v.EUncens * C', sum(log.(diag(C))))
    
    if todo
        # Compute inverse covariance matrix
        invSig = C' * C
        
        # Compute derivatives
        t = v.EUncens * (-invSig)
        v.dphi_dE[:, v.uncens] .= t
        
        # Compute derivative w.r.t. Sigma
        v.dphi_dSig[:, v.SigIndsUncens] .= neg_half_E_Dinvsym_E(t, v.EDE) .- 
                                           reshape(invSig, :, 1)' .* v.halfDmatrix
    end
    
    ϕ
end

# log likelihood and scores for likelihood over total range of truncation--denominator of L
function lnLTrunc(v::SubView, todo::Bool)
    # Declare local variables
    local dPhi_dEt::Matrix{Float64}, dPhi_dFt::Matrix{Float64}, dPhi_dSigt::Matrix{Float64}
    local Φ::Vector{Float64}
    
    # Initialize matrices if todo is true
    if todo
        n_rows = size(v.Et, 1)
        n_trunc = length(v.trunc)
        dPhi_dEt = Matrix{Float64}(undef, n_rows, n_trunc)
        dPhi_dFt = Matrix{Float64}(undef, n_rows, n_trunc)
        # Size of dPhi_dSigt depends on the number of unique elements in the covariance matrix
        n_sig_inds = length(v.SigIndsTrunc)
        dPhi_dSigt = Matrix{Float64}(undef, n_rows, n_sig_inds)
    else
        # Create empty matrices to pass to function
        dPhi_dEt = Matrix{Float64}(undef, 0, 0)
        dPhi_dFt = Matrix{Float64}(undef, 0, 0)
        dPhi_dSigt = Matrix{Float64}(undef, 0, 0)
    end
    
    # Call vecmultinormal function
    Φ = vecmultinormal(v.Et, v.Ft, v.Ω[v.trunc, v.trunc], v.d_trunc, v.one2d_trunc, 
                       todo, dPhi_dEt, dPhi_dFt, dPhi_dSigt, 1)
    
    if todo
        # Combine derivatives and assign to output
        v.dPhi_dEt[:, v.trunc] .= dPhi_dEt .+ dPhi_dFt
        v.dPhi_dSigt[:, v.SigIndsTrunc] .= dPhi_dSigt
    end
    
    return Φ
end


# log likelihood and scores for cumulative normal
# returns scores in v.dPhi_dE.M, v.dPhi_dSig.M if requested
function lnLCensored(v::SubView, todo::Bool)
    # Local variable declarations
    local t, pSig, roprobit_pSig, fracprobit_pSig, beta, invSig_uncens, Sig_uncens_cens
    local S_dPhi_dpE, S_dPhi_dpF, S_dPhi_dpSig, SS_dPhi_dpE, SS_dPhi_dpF, SS_dPhi_dpSig
    local dPhi_dpE, dPhi_dpF, dPhi_dpSig
    local ThisNumCuts, d_cens, d_two_cens, N_perm, ThisPerm, ThisFracComb
    local i, j, S_Phi, SS_Phi, Φ
    local uncens, cens, oprobit
    local pE, roprobit_pE, fracprobit_pE, F, roprobit_pQE, pdPhi_dpF
    
    # Extract frequently used values
    uncens = v.uncens
    oprobit = v.oprobit
    cens = v.cens
    d_cens = v.d_cens
    d_two_cens = v.d_two_cens
    N_perm = v.N_perm
    ThisNumCuts = v.NumCuts
    
    # Initialize matrices based on todo flag
    if todo
        n_rows = size(v.ECens, 1)
        dPhi_dpE = Matrix{Float64}(undef, n_rows, d_cens)
        dPhi_dpF = Matrix{Float64}(undef, n_rows, d_cens)
        dPhi_dpSig = Matrix{Float64}(undef, n_rows, size(pSig, 1) * (size(pSig, 1) + 1) ÷ 2)
    end
    
    pdPhi_dpF = v.NumRoprobitGroups > 0 ? dPhi_dpF : v.dPhi_dpF
    
    if v.d_uncens > 0  # Partial continuous variables out of the censored ones
        invSig_uncens = inv(cholesky(v.Ω[uncens, uncens]))
        Sig_uncens_cens = v.Ω[uncens, cens]
        beta = invSig_uncens * Sig_uncens_cens
        
        t = v.EUncens * beta
        pE = v.ECens .- t  # partial out errors from upper bounds
        roprobit_pE = fracprobit_pE = pE
        F = d_two_cens > 0 ? v.F .- t : zeros(0, 0)  # partial out errors from lower bounds
        pSig = v.Ω[cens, cens] - Sig_uncens_cens' * beta  # corresponding covariance
        roprobit_pSig = fracprobit_pSig = pSig
    else
        pE = v.ECens
        roprobit_pE = fracprobit_pE = pE
        F = d_two_cens > 0 ? v.F : zeros(0, 0)
        pSig = v.Ω[cens, cens]
        roprobit_pSig = fracprobit_pSig = pSig
    end
    
    for ThisFracComb = v.NFracCombs:-1:1
        if ThisFracComb < v.NFracCombs
            fracprobit_pE = pE .* diag(v.frac_QE[ThisFracComb].M)'
            roprobit_pE = fracprobit_pE
            fracprobit_pSig = v.frac_QE[ThisFracComb].M' * pSig * v.frac_QE[ThisFracComb].M
            roprobit_pSig = fracprobit_pSig
        end
        
        for ThisPerm = N_perm:-1:1
            if v.NumRoprobitGroups > 0
                roprobit_pQE = v.roprobit_QE[ThisPerm]
                roprobit_pE = fracprobit_pE * roprobit_pQE
                roprobit_pSig = roprobit_pQE' * fracprobit_pSig * roprobit_pQE
            end
            
            Φ = vecmultinormal(roprobit_pE, F, roprobit_pSig, v.dCensNonrobase, v.two_cens,
                              todo, dPhi_dpE, v.dPhi_dpF, dPhi_dpSig, N_perm)
            
            if todo && v.NumRoprobitGroups > 0
                dPhi_dpE = dPhi_dpE * roprobit_pQE'
                dPhi_dpSig = dPhi_dpSig * v.roprobit_Q_Sig[ThisPerm]
                if d_two_cens > 0
                    dPhi_dpF = zeros(v.N, d_cens)
                    dPhi_dpF[:, v.cens_nonrobase] = v.dPhi_dpF
                end
            end
            
            if N_perm > 1
                if ThisPerm == N_perm
                    S_Phi = Φ
                    if todo
                        S_dPhi_dpE = dPhi_dpE
                        S_dPhi_dpSig = dPhi_dpSig
                        if d_two_cens > 0
                            S_dPhi_dpF = dPhi_dpF
                        end
                    end
                else
                    S_Phi = S_Phi .+ Φ
                    if todo
                        S_dPhi_dpE = S_dPhi_dpE .+ dPhi_dpE
                        S_dPhi_dpSig = S_dPhi_dpSig .+ dPhi_dpSig
                        if d_two_cens > 0
                            S_dPhi_dpF = S_dPhi_dpF .+ dPhi_dpF
                        end
                    end
                end
            end
        end
        
        if N_perm > 1
            Φ = log.(S_Phi)
            if todo
                dPhi_dpE = S_dPhi_dpE ./ S_Phi
                dPhi_dpSig = S_dPhi_dpSig ./ S_Phi
                if d_two_cens > 0
                    dPhi_dpF = S_dPhi_dpF ./ S_Phi
                end
            end
        end
        
        if v.d_frac > 0
            if ThisFracComb == v.NFracCombs
                SS_Phi = Φ .* v.yProd[ThisFracComb].M
                if todo
                    SS_dPhi_dpE = dPhi_dpE .* v.yProd[ThisFracComb].M
                    SS_dPhi_dpSig = dPhi_dpSig .* v.yProd[ThisFracComb].M
                    if d_two_cens > 0
                        SS_dPhi_dpF = pdPhi_dpF .* v.yProd[ThisFracComb].M
                    end
                end
            elseif ThisFracComb == 1
                Φ = SS_Phi .+ Φ .* v.yProd[ThisFracComb].M
                if todo
                    dPhi_dpE = SS_dPhi_dpE .+ (dPhi_dpE .* v.yProd[ThisFracComb].M) * v.frac_QE[ThisFracComb].M
                    dPhi_dpSig = SS_dPhi_dpSig .+ (dPhi_dpSig .* v.yProd[ThisFracComb].M) * v.frac_QSig[ThisFracComb].M
                    if d_two_cens > 0
                        pdPhi_dpF = SS_dPhi_dpF .+ (pdPhi_dpF .* v.yProd[ThisFracComb].M) * v.frac_QE[ThisFracComb].M
                    end
                end
            else
                SS_Phi = SS_Phi .+ Φ .* v.yProd[ThisFracComb].M
                if todo
                    SS_dPhi_dpE = SS_dPhi_dpE .+ (dPhi_dpE .* v.yProd[ThisFracComb].M) * v.frac_QE[ThisFracComb].M
                    SS_dPhi_dpSig = SS_dPhi_dpSig .+ (dPhi_dpSig .* v.yProd[ThisFracComb].M) * v.frac_QSig[ThisFracComb].M
                    if d_two_cens > 0
                        SS_dPhi_dpF = SS_dPhi_dpF .+ (pdPhi_dpF .* v.yProd[ThisFracComb].M) * v.frac_QE[ThisFracComb].M
                    end
                end
            end
        end
    end
    
    if todo
        local dpE_dE, dpSig_dSig
        local lcut, lcat
        local pYi_lcat, pYi_lcatm1
        
        # Translate scores w.r.t. partialled errors and variance to ones w.r.t. unpartialled ones
        if v.d_uncens > 0
            t = hcat(I(size(beta, 2)), -beta')
            dpE_dE = v.J_d_cens_d_0
            dpE_dE[:, v.cens_uncens] = t
            v.dPhi_dE .= dPhi_dpE * dpE_dE
            
            dpSig_dSig = v.J_d2_cens_d2_0
            dpSig_dSig[:, v.SigIndsCensUncens] = kron(t, t)[v.CensLTInds, :] * v.dSig_dLTSig
            
            v.dPhi_dpE_dSig[:, v.SigIndsCensUncens] .= 
                dPhi_dpE_dSig(v.EUncens, beta, invSig_uncens, Sig_uncens_cens, dPhi_dpE, 
                             d_cens, v.d_uncens, v._dPhi_dpE_dSig, v.J_d_uncens_d_cens_0)
            v.dPhi_dSig .= dPhi_dpSig * dpSig_dSig .+ v.dPhi_dpE_dSig
        else
            v.dPhi_dE[:, v.cens_uncens] .= dPhi_dpE
            v.dPhi_dSig[:, v.SigIndsCensUncens] .= dPhi_dpSig
        end
        
        if d_two_cens > 0
            if v.d_uncens > 0
                v.dPhi_dF .= pdPhi_dpF * dpE_dE
                v.dPhi_dpF_dSig[:, v.SigIndsCensUncens] .= 
                    dPhi_dpE_dSig(v.EUncens, beta, invSig_uncens, Sig_uncens_cens, v.dPhi_dpF,
                                 d_cens, v.d_uncens, v._dPhi_dpF_dSig, v.J_d_uncens_d_cens_0)
                v.dPhi_dSig .= v.dPhi_dSig .+ v.dPhi_dpF_dSig
            else
                v.dPhi_dF[:, v.cens_uncens] .= pdPhi_dpF
            end
            
            if ThisNumCuts > 0
                lcat = (lcut = ThisNumCuts) + (i = v.d_oprobit) + 1
                for _ = 1:i  # for each oprobit eq
                    pYi_lcat = v.Yi[:, lcat]
                    lcat -= 1
                    for j = 1:v.vNumCuts[i]
                        pYi_lcatm1 = v.Yi[:, lcat]
                        lcat -= 1
                        v.dPhi_dcuts[:, v.CutInds[lcut]] .= 
                            v.dPhi_dE[:, oprobit[i]] .* pYi_lcatm1 .+ v.dPhi_dF[:, oprobit[i]] .* pYi_lcat
                        lcut -= 1
                        pYi_lcat = pYi_lcatm1
                    end
                    i -= 1
                end
            end
            v.dPhi_dE .= v.dPhi_dE .+ v.dPhi_dF
        end
    end
    
    return Φ
end

# translate draws or nodes at a given level, possibly adaptively shifted, into total effects of random effects and coefficients
ffunction BuildTotalEffects(l::Int, REs::Vector{RE}, NumREDraws::Vector{Int}, HasΓ::Bool)
    RE = REs[l]
    
    for r = NumREDraws[l+1]:-1:1
        if RE.HasRC
            pUT = copy(RE.J_N_NEq_0)
            if size(RE.REInds, 2) > 0
                # Set columns for REs
                pUT[:, RE.REEqs] = RE.U[r].M * RE.T[:, RE.REInds]
            end
        else
            pUT = RE.U[r].M * RE.T  # REs
        end
        
        # RCs
        for eq = RE.NEq:-1:1
            if RE.RCk[eq] > 0
                # RCs * X
                pUT[:, eq] = view(pUT, :, eq) + 
                    sum((RE.U[r].M * RE.T[:, RE.RCInds[eq].M]) .* RE.X[eq].M, dims=2)
            end
        end
        
        if HasΓ
            for eq = length(RE.ΓEqs):-1:1
                RE.TotalEffect[r, RE.ΓEqs[eq]].M = pUT * RE.invΓ[:, eq]
            end
        else
            for eq = length(RE.ΓEqs):-1:1
                RE.TotalEffect[r, RE.ΓEqs[eq]].M = view(pUT, :, eq)
            end
        end
    end
end

function BuildXU(l::Int, REs::Vector{RE}, subviews::Union{SubView, Nothing}, base::Base)
    RE = REs[l]
    
    if RE.HasRC
        for r = RE.R:-1:1  # pre-compute X-U products in order most convenient for computing scores w.r.t upper-level T's
            k = 0
            e = 0
            for eq1 = 1:RE.NEq
                for c = 1:(RE.RCk[eq1] + (eq1 in RE.REEqs ? 1 : 0))
                    e += 1
                    Ue = view(RE.U[r].M, :, e)
                    k += 1
                    
                    if c <= RE.RCk[eq1]
                        RE.XU[r, k] = Ue .* view(RE.X[eq1].M, :, c:RE.RCk[eq1])
                    else
                        RE.XU[r, k] = base.J_N_0_0
                    end
                    
                    if eq1 in RE.REEqs
                        RE.XU[r, k] = hcat(RE.XU[r, k], Ue)
                    end
                    
                    for eq2 = (eq1+1):RE.NEq
                        k += 1
                        if RE.RCk[eq2] > 0
                            RE.XU[r, k] = Ue .* RE.X[eq2].M
                        else
                            RE.XU[r, k] = base.J_N_0_0
                        end
                        
                        if eq2 in RE.REEqs
                            RE.XU[r, k] = hcat(RE.XU[r, k], Ue)
                        end
                    end
                end
            end
        end
    else
        # simpler form works when just REs
        for r = RE.R:-1:1
            for j = RE.d:-1:1
                RE.XU[r, j] = view(RE.U[r].M, :, j)
            end
        end
    end
    
    # Process subviews
    v = subviews
    while v !== nothing
        for r = RE.R:-1:1
            for j = size(v.XU[l].M, 2):-1:1
                if RE.pXU[r, j]
                    v.XU[l].M[r, j].M = RE.XU[r, j][v.SubsampleInds, :]
                end
            end
        end
        v = v.next
    end
end

# Helper structure definitions (adjust based on your actual implementation)
struct MatrixWrapper
    M::Matrix{Float64}
end


# main evaluator routine
void cmp_lf1(transmorphic M, real scalar todo, real rowvector b, real colvector lnf, real matrix S, real matrix H) 
  pragma unused H
  pointer(class cmp_model scalar) scalar mod
  mod = moptimize_init_userinfo(M, 1)
  mod.lf1(M, todo, b, lnf, S)
end


void lf1(transmorphic M, real scalar todo, real rowvector b, real colvector lnf, real matrix S) where T
	real matrix t, L_g, invΓ, C, dΩ_dSig, L_gv, L_gvr, sThetaScores, sCutScores
	real scalar e, c, i, j, k, l, m, _l, r, tEq, EUncensEq, ECensEq, FCensEq, NewIter, eq, eq1, eq2, _eq, c1, c2, cut, lnsigWithin, lnsigAccross, atanhrhoAccross, atanhrhoWithin, Iter
	real colvector shift, lnLmin, lnLmax, lnL, out
	pointer(struct subview scalar) scalar v
	pointer(real matrix) scalar pdlnL_dtheta, pdlnL_dSig, pThisQuadXAdapt_j, pt
	pointer(struct scores scalar) scalar Scores
	pointer (struct RE scalar) scalar RE
	pointer(pointer (real matrix) colvector) scalar pThisQuadXAdapt
	pragma unset out; pragma unset sThetaScores; pragma unset sCutScores  # allocate these instead of relying on scoreAccum() to do it on first call

	lnf = .

	for (i=1; i<=d; i++) 
		REs.theta[i].M = moptimize_util_xb(M, b, i)
		if (nrows(REs.theta[i].M)==1) REs.theta[i].M = J(base.N, 1, REs.theta[i].M)
	end

	for (j=1; j<=nrows(ΓInd); j++)
		Γ[|ΓInd[j,]|] = -moptimize_util_xb(M, b, i++)

	for (eq1=1; eq1<=d; eq1++)
		if (vNumCuts[eq1])
			for (cut=2; cut<=vNumCuts[eq1]+1; cut++) 
				cuts[cut, eq1] = moptimize_util_xb(M, b, i++)
				if (trunceqs[eq1])
					if (any(indicators[,eq1] :& ((Lt[eq1].M .< . :& cuts[cut, eq1] .< Lt[eq1].M) :| cuts[cut, eq1].>Ut[eq1].M)))
						return
			end

  for (l=1; l<=L; l++)  # loop over hierarchy levels
		RE = &((*REs)[l])
		RE.sig = RE.ρ = J(1, 0, 0)
		if (RE.covAcross==0)  # exchangeable across?
			lnsigWithin = lnsigAccross = moptimize_util_xb(M, b, i++)

		for (eq1=1; eq1<=RE.NEq; eq1++) 
			if (RE.covWithin[RE.Eqs[eq1]]==0 & RE.covAcross)  # exchangeable within but not across?
				lnsigWithin = lnsigAccross

			for (c1=1; c1<=RE.NEff[eq1]; c1++)
				if (RE.FixedSigs[RE.Eqs[eq1]] == .) 
					if (RE.covWithin[RE.Eqs[eq1]] & RE.covAcross)  # exchangeable neither within nor accross?
						lnsigWithin = moptimize_util_xb(M, b, i++)
				  !SigXform && iszero(lnsigWithin) && return
					RE.sig = RE.sig, (SigXform ? exp(lnsigWithin) : lnsigWithin)
				else
					RE.sig = RE.sig, RE.FixedSigs[RE.Eqs[eq1]]
		end

		if (RE.covAcross==0 & RE.d > 1)  # exchangeable across?
			atanhrhoAccross = moptimize_util_xb(M, b, i++)
		for (eq1=1; eq1<=RE.NEq; eq1++) 
			if (RE.covWithin[RE.Eqs[eq1]] == 2)  # independent?
				atanhrhoWithin = 0
			else if (RE.covWithin[RE.Eqs[eq1]]==0 & RE.NEff[eq1] > 1)  # exchangeable within?
				atanhrhoWithin = moptimize_util_xb(M, b, i++)
			for (c1=1; c1<=RE.NEff[eq1]; c1++) 
				for (c2=c1+1; c2<=RE.NEff[eq1]; c2++) 
					if (RE.covWithin[RE.Eqs[eq1]] == 1)  # unstructured?
						atanhrhoWithin = moptimize_util_xb(M, b, i++)
					RE.ρ = RE.ρ, atanhrhoWithin
				end
				for (eq2=eq1+1; eq2<=RE.NEq; eq2++)
					for (c2=1; c2<=RE.NEff[eq2]; c2++)
						if (RE.FixedRhos[RE.Eqs[eq2],RE.Eqs[eq1]] == .) 
							if (RE.covAcross == 1)  # unstructured?
								atanhrhoAccross = moptimize_util_xb(M, b, i++)
							RE.ρ = RE.ρ, atanhrhoAccross
						else
							RE.ρ = RE.ρ, RE.FixedRhos[RE.Eqs[eq2],RE.Eqs[eq1]]
			end
		end
	end

	if (HasΓ) 
		invΓ = luinv(Γ)
		if (invΓ[1,1] == .) return
		for (eq1=d; eq1; eq1--)  # XXX is this faster than manually multipling invΓ by the individual theta columns and summing?
			Theta[:,eq1] = editmissing(REs.theta[eq1].M, 0)  # only time missing values would appear and be used is when multiplied by invΓ with 0's in corresponding entries
		for (eq1=d; eq1; eq1--)
			REs.theta[eq1].M = Theta * invΓ[,eq1]
	end

	if (WillAdapt)
		if (NewIter = (Iter = moptimize_result_iterations(M)) != LastIter) 
			LastIter = Iter
			if (Adapted==0)
				if (AdaptNextTime) 
					M.Adapted = M.AdaptivePhaseThisEst = true
					printf("\nresendPerforming Naylor-Smith adaptive quadrature.\n")
				else 
					isdefined(M, :Lastb) && AdaptNextTime = mreldif(b, Lastb) < .1  # criterion to begin adaptive phase
					M.Lastb = b
				end
		end

  for (l=1; l<=L; l++) 
		RE = &((*REs)[l])
		if (RE.d == 1)
			RE.Sig = (RE.T = RE.sig) * RE.sig
		else 
			k = 0
			for (j=1; j<=RE.d; j++)
				for (i=j+1; i<=RE.d; i++)
					if (SigXform)
						if (RE.ρ[++k]>100)
							RE.Rho[i,j] = 1
						else if (RE.ρ[k]<-100)
							RE.Rho[i,j] = -1
						else
  						RE.Rho[i,j] = tanh(RE.ρ[k])
					else
						RE.Rho[i,j] = RE.ρ[++k]
			_makesymmetric(RE.Rho)
			RE.T = cholesky(RE.Rho)' .* RE.sig
			if (RE.T[1,1] == .) return
			RE.Sig = quadcross(RE.sig,RE.sig) .* RE.Rho
		end

		if (todo)
			RE.D = dSigdsigrhos(SigXform, RE.sig, RE.Sig, RE.ρ, RE.Rho) * RE.dSigdParams

		if (HasΓ)
			RE.invΓ = invΓ[RE.Eqs,RE.ΓEqs]

		if (l < L) 
			BuildTotalEffects(l)
			for (eq1=ncols(RE.ΓEqs); eq1; eq1--)  # compute effect of first draws
				_eq = RE.ΓEqs[eq1]
				(*REs)[l+1].theta[_eq].M = nrows(RE.TotalEffect[1,_eq].M) ? RE.theta[_eq].M .+ RE.TotalEffect[1,_eq].M : RE.theta[_eq].M
			end
			for (eq1=d; eq1; eq1--)  # by default lower errors = upper ones, for eqs with no random effects/coefs at this level
				if (anyof(RE.ΓEqs,eq1)==0)
					(*REs)[l+1].theta[eq1].M = RE.theta[eq1].M
			if (todo)
				RE.D = ghk2_dTdV(RE.T') * RE.D
			if (AdaptivePhaseThisEst & NewIter)
				RE.ToAdapt = RE.JN12
			RE.AdaptivePhaseThisIter = 0
		end
	end

	if (HasΓ) 
		if (todo) 
			dΩ_dSig = (Lmatrix(ncols(invΓ))*(invΓ'#invΓ')*Dmatrix(nrows(invΓ)))  # QE2QSig(invΓ)
			t = colshape(invΓ, 1)
			t = (colshape(base.Sig,1)'#Idd)[vLd,vIKI] * (Idd#t + t#Idd)[,vKd]
			for (m=d; m; m--)
				for (c=1; c<=G[m]; c++)
					dΩdΓ[m,c].M = t * invΓ[m,]'#invΓ[,(*ΓIndByEq[m])[c]]
		end
		Ω = quadcross(invΓ, base.Sig) * invΓ
	end

	for (v = subviews; v; v = v.next) 
		v.Ω = quadcross(v.QE, *Ω) * v.QE
		if (todo)
			if (HasΓ) 
				for (m=d; m; m--) 
					if (G[m] & v.TheseInds[m])
						for (c=1; c<=G[m]; c++)
							v.dΩdΓ[m,c].M = v.QSig * dΩdΓ[m,c].M
				end
				v.QEinvΓ    = quadcross(v.QE, invΓ')
				v.invΓQSigD = quadcross(v.QSig, dΩ_dSig) * base.D
			else 
				v.QEinvΓ    = v.QE'
				v.invΓQSigD = quadcross(v.QSig, base.D)
			end
	end

	base.plnL = &(lnf = zeros(T,N))
	if (todo) S = S0

	do  # for each draw combination
		for (v = subviews; v; v = v.next) 
			tEq = EUncensEq = ECensEq = FCensEq = 0
			for (i=1; i<=d; i++)
				if (v.TheseInds[i]==cmp_mprobit)  # handle mprobit eqs below
					++ECensEq
					++FCensEq
				else 
					if v.NotBaseEq[i]
						v.theta[i].M = base.theta[i].M[v.SubsampleInds]

						if (v.TheseInds[i] & v.TheseInds[i]<.) 
							if (v.TheseInds[i]==cmp_cont)
								v.EUncens[:,++EUncensEq] = v.y[i].M - v.theta[i].M
							else 
								++ECensEq
								if (v.TheseInds[i]==cmp_left | v.TheseInds[i]==cmp_int)
									setcol(v.ECens, ECensEq, v.y[i].M - v.theta[i].M)
								else if (v.TheseInds[i]==cmp_right)
									setcol(v.ECens, ECensEq, v.theta[i].M - v.y[i].M)
								else if (v.TheseInds[i]==cmp_probit)
									setcol(v.ECens, ECensEq, -v.theta[i].M)
								else if (v.TheseInds[i]==cmp_probity1 | v.TheseInds[i]==cmp_frac)
									setcol(v.ECens, ECensEq, v.theta[i].M)
								else if (v.TheseInds[i]==cmp_oprobit) 
									if trunceqs[i]
										t = v.y[i].M .> v.vNumCuts[i]  # bit of inefficiency in truncated oprobit case
										setcol(v.ECens, ECensEq, (t .* v.Ut[i].M + (1.-t) .* cuts[v.y[i].M.+1, i]) - v.theta[i].M)
									else
										setcol(v.ECens, ECensEq, cuts[v.y[i].M.+1, i] - v.theta[i].M)
								else  # roprobit
									setcol(v.ECens, ECensEq, -v.theta[i].M)
								if (v.F)
									if (NonbaseCases[ECensEq]) 
										++FCensEq
                    v.Fi = J(0,0,0)
										if (v.TheseInds[i]==cmp_int)
											v.Fi = v.yL[i].M - v.theta[i].M
										else if (v.TheseInds[i]==cmp_oprobit)
											if (trunceqs[i]) 
												t = v.y[i].M
												v.Fi = (t .* v.Lt[i].M + (1.-t) .* cuts[v.y[i].M, i]) - v.theta[i].M
											else
												v.Fi = cuts[ v.y[i].M, i] - v.theta[i].M
										else if (trunceqs[i])
											if (v.TheseInds[i]==cmp_left)
												v.Fi = v.Lt[i].M - v.theta[i].M
											else if (v.TheseInds[i]==cmp_right)
												v.Fi = v.theta[i].M - v.Ut[i].M
											else if (v.TheseInds[i]==cmp_probit)
												v.Fi = v.Lt[i].M - v.theta[i].M
											else if (v.TheseInds[i]==cmp_probity1)
												v.Fi = v.theta[i].M - v.Ut[i].M
                    if (nrows(v.Fi)) setcol(v.F, FCensEq, v.Fi)
									end
							end

							if (trunceqs[i]) 
								++tEq
								if (v.TheseInds[i]==cmp_left) 
									setcol(v.Et, tEq, v.Ut[i].M - v.theta[i].M)
									setcol(v.Ft, tEq, v.Fi)
								else if (v.TheseInds[i]==cmp_right) 
									setcol(v.Et, tEq, v.theta[i].M - v.Lt[i].M)
									setcol(v.Ft, tEq, v.Fi)
								else if (v.TheseInds[i]==cmp_probit) 
									setcol(v.Et, tEq, v.Ut[i].M - v.theta[i].M)
									setcol(v.Ft, tEq, v.Fi)
								else if (v.TheseInds[i]==cmp_probity1) 
									setcol(v.Et, tEq, v.theta[i].M - v.Lt[i].M)
									setcol(v.Ft, tEq, v.Fi)
								else if (anyof((cmp_cont,cmp_oprobit,cmp_int), v.TheseInds[i])) 
									setcol(v.Et, tEq, v.Ut[i].M - v.theta[i].M)
									setcol(v.Ft, tEq, v.Lt[i].M - v.theta[i].M)
								end
							end
						end
					end
				end

      for (j=nrows(MprobitGroupInds); j; j--)  # relative-difference mprobit errors
				if (v.mprobit[j].d > 0) 
					out = base.theta[v.mprobit[j].out].M[v.SubsampleInds]
					for (i=v.mprobit[j].d; i; i--)
						setcol(v.ECens, (v.mprobit[j].res)[i], out - base.theta[(v.mprobit[j].in)[i]].M[v.SubsampleInds])
				end

			if (v.ECens) _editmissing(*v.ECens,  1.701e+38)  # maxfloat()--just a big number
			if (v.F    ) _editmissing(*v.F    , -1.701e+38)
			if (v.Et   ) _editmissing(*v.Et   ,  1.701e+38)
			if (v.Ft   ) _editmissing(*v.Ft   , -1.701e+38)

			if (v.d_cens) 
				lnL = lnLCensored(v, todo)
				if (v.d_uncens)
					lnL = lnL + lnLContinuous(v, todo)
			else
				lnL = lnLContinuous(v, todo)

			if (v.d_trunc)
				lnL = lnL - lnLTrunc(v, todo)

      (*(base.plnL))[v.SubsampleInds] = lnL

			if (todo) 
				if (v.d_cens)
					if (v.d_uncens) 
						pdlnL_dtheta = &(v.dphi_dE + v.dPhi_dE) 
						pdlnL_dSig =  &(v.dphi_dSig + v.dPhi_dSig)
					else 
						pdlnL_dtheta = &(v.dPhi_dE)
						pdlnL_dSig   =  &(v.dPhi_dSig)
					end
				else 
					pdlnL_dtheta = &(v.dphi_dE)
					pdlnL_dSig   = &(v.dphi_dSig)
				end
				if (v.d_trunc) 
					pdlnL_dtheta = &(*pdlnL_dtheta - v.dPhi_dEt)
					pdlnL_dSig   = &(*pdlnL_dSig   - v.dPhi_dSigt)
				end

				pdlnL_dtheta = &(*pdlnL_dtheta * v.QEinvΓ)

				if (L == 1) 
					                   S[v.SubsampleInds, Scores.ThetaScores  ] = *pdlnL_dtheta
					if (NumCuts)  S[v.SubsampleInds, Scores.  CutScores  ] = v.dPhi_dcuts
					if (ncols(base.D)) S[v.SubsampleInds, Scores.  SigScores.M] = *pdlnL_dSig * v.invΓQSigD
					for (i=m=1; m<=d; m++)
						for (c=1; c<=G[m]; c++)
   						S[v.SubsampleInds, Scores.ΓScores[i++].M] .= v.TheseInds[m] ? 
								(v.NotBaseEq[(*ΓIndByEq[m])[c]] ? *pdlnL_dSig*v.dΩdΓ[m,c].M + (*pdlnL_dtheta)[:,m].*v.theta[(*ΓIndByEq[m])[c]].M :
                                                            *pdlnL_dSig*v.dΩdΓ[m,c].M                                                                        ) :
                0
				else 
					_editmissing(*pdlnL_dtheta, 0)
					_editmissing(v.dPhi_dcuts, 0)
					_editmissing(*pdlnL_dSig, 0)

					Scores = &(v.Scores[L].M[ThisDraw[L]])
					                   Scores.ThetaScores  = *pdlnL_dtheta
					if (NumCuts)  Scores.CutScores    = v.dPhi_dcuts
					if (ncols(base.D)) Scores.TScores[L].M = *pdlnL_dSig
					for (i=m=1; m<=d; m++)
						if (v.TheseInds[m])
							for (c=1; c<=G[m]; c++)
								Scores.ΓScores[i++].M  = view(*pdlnL_dtheta,:,m) .* v.theta[(*ΓIndByEq[m])[c]].M
						else
							i = i + G[m]

					for (l=1; l<L; l++) 
						RE = &((*REs)[l])
						  # dlnL/dSigparams = dlnL/dE^ * dE^/dE * dE/dT * dT/dΩ * dΩ/dSig * dSig/dSigparams = dlnL/dE * QE * X*Uend * dT_dSig * dΩ_dSig * D. Last 3 terms draw-invariant, so saved for end
						for (e=k=eq1=1; eq1<=RE.NEq; eq1++)
							if (RE.HasRC)
								for (c=1; c<=ncols(RE.RCInds[eq1].M)+anyof(RE.REEqs, eq1); c++)
									for (eq2=eq1; eq2<=RE.NEq; eq2++)
										PasteAndAdvance!(Scores.TScores[l].M, k, 
											(v.XU[l].M[ThisDraw[l+1], e++].M) .* view(Scores.ThetaScores,:,RE.Eqs[eq2]))
							else
								PasteAndAdvance!(Scores.TScores[l].M, k, (v.XU[l].M[ThisDraw[l+1], eq1].M) .* view(Scores.ThetaScores,:,RE.Eqs[|eq1 \ .|]))
					end
				end
			end
		end

		for (l=L-1; l; l--)  # If L=1, sets l=0 as needed to terminate do loop. Usually this loop runs once.
			RE = REs[l]

			RE.lnLByDraw[:, ThisDraw[l+1]] =  panelsum(*((*REs)[l+1].plnL), (*REs)[l+1].Weights, RE.IDRangesGroup)
			if (ThisDraw[l+1] < RE.R)
				ThisDraw[l+1] = ThisDraw[l+1] + 1
			else 
				if (Adapted)
					RE.lnLByDraw = RE.lnLByDraw + RE.AdaptiveShift  # even if active adaptation done, add adaptive log(det(C)*normalden(QuadXAdapt)/normalden(QuadX))

  # for each group, make weights proportional to L (not lnL) for the group/obs at next-lower level
				t = RE.lnLlimits .- rowminmax(RE.lnLByDraw)  # In summing groups' Ls, shift just enough to prevent underflow in exp(), but if necessary even less to avoid overflow
				lnLmin = t[,1]; lnLmax = t[,2]
				t = lnLmin.*(lnLmin.>0) - lnLmax; shift = t .* (t .< 0) + lnLmax  # parallelizes better than rowminmax()
				L_g = editmissing(exp(RE.lnLByDraw.+shift), 0)  # un-log likelihood for each group & draw; lnL=. => L=0
			  L_g = L_g .* RE.QuadW
				RE.plnL = &quadrowsum(L_g)  # in non-quadrature case, sum rather than average of likelihoods across draws
				if (todo | (AdaptivePhaseThisEst & WillAdapt))
					L_g = editmissing(L_g ./ *(RE.plnL), 0)  # normalize L_g's as weights for obs-level scores or for use in Naylor-Smith adaptation

				if (AdaptivePhaseThisEst & NewIter) 
					pThisQuadXAdapt = &asarray(RE.QuadXAdapt, ThisDraw[|.\l|])
					if (nrows(*pThisQuadXAdapt)==0)  # initialize if needed
						asarray(RE.QuadXAdapt, ThisDraw[|.\l|], RE.JN1pQuadX)
						pThisQuadXAdapt = &asarray(RE.QuadXAdapt, ThisDraw[|.\l|])
					end
          
          if (RE.d == 1)  # optimized code for 1-D case
            for (j=RE.N; j; j--)
              if (RE.ToAdapt[j]) 
                RE.QuadMean[j].M = (t = L_g[j,]) * *(pThisQuadXAdapt_j = (*pThisQuadXAdapt)[j])  # weighted sum

                C = *pThisQuadXAdapt_j .- RE.QuadMean[j].M; C = sqrt(t * (C .* C))

                if (C == .)  # diverged? try restarting, but decrement counter to prevent infinite loop
                  RE.ToAdapt[j] = RE.ToAdapt[j] - 1
                  pThisQuadXAdapt_j = (*pThisQuadXAdapt)[j] = &(RE.QuadX)
                  RE.AdaptiveShift[j,] = RE.J1R0
                else 
                  RE.QuadSD[j].M = C
                  if (mreldif(*pThisQuadXAdapt_j, *(pt = &(RE.QuadX * C .+ RE.QuadMean[j].M))) < QuadTol)  # has adaptation converged for this ML search iteration?
                    RE.ToAdapt[j] = 0
                    continue
                  end
                  (*pThisQuadXAdapt)[j] = pt
                  if (pThisQuadXAdapt_j != (&(RE.QuadX))) pThisQuadXAdapt_j = pt
                  RE.AdaptiveShift[j,] = (log(C) - 0.91893853320467267 /*log(2pi)/2*/) .- (.5 * (*pt .* *pt)' + RE.lnnormaldenQuadX)
                end

                for (r=RE.R; r; r--)
                  RE.U[r].M[|RE.Subscript[j].M|] = J(RE.IDRangeLengths[j], 1, (*pThisQuadXAdapt_j)[r,])  # faster to explode these here than after multiplying by T in BuildTotalEffects(), BuildXU()
              end
          else 
            for (j=RE.N; j; j--)
              if (RE.ToAdapt[j]) 
                RE.QuadMean[j].M = (t = L_g[j,]) * *(pThisQuadXAdapt_j = (*pThisQuadXAdapt)[j])  # weighted sum

                C = cholesky(crossdev(*pThisQuadXAdapt_j, RE.QuadMean[j].M, t, *pThisQuadXAdapt_j, RE.QuadMean[j].M))

                if (C[1,1] == .)  # diverged? try restarting, but decrement counter to prevent infinite loop
                  RE.ToAdapt[j] = RE.ToAdapt[j] - 1
                  pThisQuadXAdapt_j = (*pThisQuadXAdapt)[j] = &(RE.QuadX)
                  RE.AdaptiveShift[j,] = RE.J1R0
                else 
                  RE.QuadSD[j].M = diag(C)
                  if (mreldif(*pThisQuadXAdapt_j, *(pt = &(RE.QuadX * C' .+ RE.QuadMean[j].M))) < QuadTol)  # has adaptation converged for this ML search iteration?
                    RE.ToAdapt[j] = 0
                    continue
                  end
                  (*pThisQuadXAdapt)[j] = pt
                  if (pThisQuadXAdapt_j != (&(RE.QuadX))) pThisQuadXAdapt_j = pt
                  RE.AdaptiveShift[j,] = quadrowsum_lnnormalden(*pt, quadcolsum(log(RE.QuadSD[j].M),1))' - RE.lnnormaldenQuadX
                end

                for (r=RE.R; r; r--)
                  RE.U[r].M[|RE.Subscript[j].M|] = J(RE.IDRangeLengths[j], 1, (*pThisQuadXAdapt_j)[r,])  # faster to explode these here than after multiplying by T in BuildTotalEffects(), BuildXU()
              end
          end

          if (RE.AdaptivePhaseThisIter = any(RE.ToAdapt) * mod(RE.AdaptivePhaseThisIter-1, QuadIter))  # not converged and haven't hit max number of adaptations?
						BuildTotalEffects(l)
						if (_todo)
							BuildXU(l)
					end
				end
				ThisDraw[l+1] = 1
			end

			if (ThisDraw[l+1] > 1 | RE.AdaptivePhaseThisIter)  # no (more) carrying? propagate draw changes down the tree
				for (_l=l; _l<L; _l++)
					for (eq=ncols(RE.ΓEqs); eq; eq--) 
						_eq = RE.ΓEqs[eq]
						(*REs)[_l+1].theta[_eq].M = ncols((*REs)[_l].TotalEffect[ThisDraw[_l+1], _eq].M)? (*REs)[_l].theta[_eq].M + (*REs)[_l].TotalEffect[ThisDraw[_l+1], _eq].M : (*REs)[_l].theta[_eq].M
					end
				break
 			end

			# finished the group's (adaptive) draws
			if (todo)  # obs-level score for next level up is avg of scores over this level's draws, weighted by group's L for each draw
				for (v = subviews; v; v = v.next) 
					L_gv = L_g[v.id[l].M, RE.one2R]
					for (r=1; r<=NumREDraws[l+1]; r++) 
						L_gvr = L_gv[:, r]

            scoreAccum(sThetaScores, r, L_gvr, v.Scores[l+1].M[r].ThetaScores)
						if (NumCuts)
              scoreAccum(sCutScores, r, L_gvr, v.Scores[l+1].M[r].CutScores)
						for (i=L; i; i--)
							if ((*REs)[i].NSigParams)
                scoreAccum(sTScores[i].M, r, L_gvr, v.Scores[l+1].M[r].TScores[i].M)
						for (i=ncols(v.Scores.M.ΓScores); i; i--)
							if (nrows(v.Scores[l+1].M[r].ΓScores[i].M))
                scoreAccum(sΓScores[i].M, r, L_gvr, v.Scores[l+1].M[r].ΓScores[i].M)
					end
					if (l==1)  # final scores
						S[v.SubsampleInds, Scores.ThetaScores] = *Xdotv(sThetaScores, v.WeightProduct)
						if (NumCuts)
							S[v.SubsampleInds, Scores.CutScores] = *Xdotv(sCutScores, v.WeightProduct)
						if (base.NSigParams)
							S[v.SubsampleInds, Scores.SigScores[L].M] = *Xdotv(sTScores[L].M * v.invΓQSigD, v.WeightProduct)
						for (i=L-1; i; i--)
							if ((*REs)[i].NSigParams)
								S[v.SubsampleInds, Scores.SigScores[i].M] = *Xdotv(sTScores[i].M * (*REs)[i].D, v.WeightProduct)
						for (i=m=1; m<=d; m++)
							for (c=1; c<=G[m]; c++) 
								if (v.TheseInds[m])
									S[v.SubsampleInds, Scores.ΓScores[i].M]  = *Xdotv(sΓScores[i].M + sTScores[L].M * v.dΩdΓ[m,c].M, v.WeightProduct)
								else
									S[v.SubsampleInds, Scores.ΓScores[i].M] .= 0
								i++
							end
					else 
							v.Scores[l].M[ThisDraw[l]].ThetaScores = sThetaScores
						if (NumCuts)
							v.Scores[l].M[ThisDraw[l]].CutScores = sCutScores
						for (i=L; i; i--)
							if ((*REs)[i].NSigParams)
								v.Scores[l].M[ThisDraw[l]].TScores[i].M = sTScores[i].M
						for (i=ncols(v.Scores.M[1].ΓScores); i; i--)
							v.Scores[l].M[ThisDraw[l]].ΓScores[i].M = sΓScores[i].M
					end
				end

			RE.plnL = &(log(*(RE.plnL)) - shift)
		end
	end while (l)  # exit when adding one more draw causes carrying all the way accross the draw counters, back to 1, 1, 1...

	if (L > 1) 
		lnf = quadsum(nrows(REs.Weights)? REs.Weights .* *(REs.plnL) : *(REs.plnL), 1)
		if (AdaptivePhaseThisEst & NewIter) 
			if (AdaptivePhaseThisEst = mreldif(LastlnLThisIter, LastlnLLastIter) >= 1e-6)
				LastlnLLastIter = LastlnLThisIter
			else
				printf("\nresendAdaptive quadrature points fixed.\n")
		end
		if (lnf < .) LastlnLThisIter = lnf
		if (todo == 0)
			lnf = J(base.N, 1, lnf/base.N)
	end
end

void cmp_gf1(transmorphic M, real scalar todo, real rowvector b, real colvector lnf, real matrix S, real matrix H) 
	pointer(class cmp_model scalar) scalar mod
	pragma unused H
	mod = moptimize_init_userinfo(M, 1)
  mod.gf1(M, todo, b, lnf, S)
end

void gf1(transmorphic M, real scalar todo, real rowvector b, real colvector lnf, real matrix S) 
	real matrix subscripts, _S; real scalar i, n, K
  pragma unset _S

	lf1(M, todo, b, lnf, _S)

	if (hasmissing(lnf) == 0) 
		lnf = *(REs.plnL)
		if (todo) 
			K = ncols(b); n = moptimize_init_eq_n(M)  # numbers of eqs (inluding auxiliary parameters); number of parameters
			S = J(nrows(lnf), K, 0)
			if (length(X) == 0) 
				X = Matrix(base.d)
				for (i=base.d;i;i--)
					X[i].M = editmissing(moptimize_util_indepvars(M, i),0)
			end
			for (i=1;i<=base.d;i++) 
				(subscripts = moptimize_util_eq_indices(M,i))[2,1] = .
				S[|subscripts|] =  panelsum(_S[,i] .* X[i].M, WeightProduct, REs.IDRanges)
			end

			if (n > d)  # any aux params?
				subscripts[1,2] = subscripts[2,2] + 1
				subscripts[2,2] = .
				S[|subscripts|] =  panelsum(_S[|.,base.d+1\.,.|], WeightProduct, REs.IDRanges)
			end
		end
	end
end

real scalar cmp_init(transmorphic M) 
	real scalar i, l, ghk_nobs, d_ghk, eq1, eq2, c, m, j, r, k, d_oprobit, d_mprobit, d_roprobit, start, stop, PrimeIndex, Hammersley, NDraws, HasRE, ncols, d2
	real matrix Yi, U
	real colvector remaining, S
	real rowvector mprobit, Primes, t, one2d
	string scalar varnames, LevelName
	pointer(struct subview scalar) scalar v, next
	pointer(struct RE scalar) scalar RE
	pointer(real matrix) rowvector QuadData
	pragma unset Yi

	REs = &RE(L, 1)
	base = &((*REs)[L])
	base.d = d
	one2d = 1:d; d2 = d*(d+1) ÷ 2

	Γ = I(d)  # really will hold I - Γ
	cuts = J(MaxCuts+2, d, 1.701e+38)  # maxfloat()
	cuts[1,] = J(1, d, -1.701e+38)  # minfloat()
	y = Lt = Ut = yL = Matrix(d)

	if (HasΓ) 
		dΩdΓ = Matrix(d,d)
		Idd = I(d*d)
		vLd = rowsum(Lmatrix(d) .*(1:d*d))  # X[vLd,] = L*X, but faster
		vKd = colsum(Kmatrix(d,d) .* (1:d*d))
		vIKI = colsum((I(d) ⊗ Kmatrix(d,d) ⊗ I(d)) .* (1:d^4))
	else
    Ω = &(base.Sig)

	ThisDraw = J(1,L,1)

	for (l=L; l; l--)
		RE = &((*REs)[l])
		RE.NEq = ncols(RE.Eqs = selectindex(Eqs[,l]'))
		RE.NEff = NumEff[l,RE.Eqs]
		RE.ΓEqs = HasΓ ? selectindex((ΓId * Eqs[,l])') : RE.Eqs
		RE.one2d = 1:( RE.d = rowsum(RE.NEff) )
		RE.theta = Matrix(d)
    RE.Rho = I(RE.d)
		RE.d2 = RE.d * (RE.d + 1) ÷ 2
		RE.covAcross = cross( st_global("cmp_cov"+strofreal(l)) .== ("exchangeable"\"unstructured"\"independent"), 0:2 ) 
		for (i=d; i; i--)
			RE.covWithin = cross( st_global("cmp_cov"+strofreal(l)+"_"+strofreal(i)) .== ("exchangeable"\"unstructured"\"independent"), 0:2 ) \ RE.covWithin
		RE.FixedSigs = st_matrix("cmp_fixed_sigs"+strofreal(l))
		RE.FixedRhos = st_matrix("cmp_fixed_rhos"+strofreal(l))
	end

	st_view(indicators, ., indVars)
	base.N = length(indicators)
	Theta = J(base.N,d,0)

	for (i=d; i; i--) 
		y[i].M = moptimize_util_depvar(M, i)
		if (trunceqs[i]) 
			st_view(Lt[i].M, ., LtVars[i])
			st_view(Ut[i].M, ., UtVars[i])
		end

		if (intregeqs[i])
			st_view(yL[i].M,  ., yLVars[i])
	end

	for (l=L-1; l; l--)
		st_view((*REs)[l].id,  ., "_cmp_id" + strofreal(l))

	for (l=L; l; l--) 
		RE = &((*REs)[l])
		if (_todo | HasΓ) 
			# build dSigdParams, derivative of sig, vech(ρ) vector w.r.t. vector of actual sig, ρ parameters, reflecting "exchangeable" and "independent" options
			real scalar accross, within, c1, c2
			t = J(0, 1, 0); i = 0  # index of entries in full sig, vech(ρ) vector
			if (RE.covAcross==0)  # exchangeable across?
				accross = ++i
			for (eq1=1; eq1<=RE.NEq; eq1++) 
				if (RE.covWithin[RE.Eqs[eq1]]==0)  # exchangeable within?
					if (RE.covAcross)  # exchangeable across?
						within = ++i
					else
						within = accross
				for (c1=1; c1<=RE.NEff[eq1]; c1++)
					if  (RE.FixedSigs[RE.Eqs[eq1]] == .)
						if (RE.covWithin[RE.Eqs[eq1]] & RE.covAcross)  # exchangeable neither within nor across?
							t = t \ ++i
						else
							t = t \ within
					else
						t = t \ .  # entry of sig vector corresponds to no parameter in model, being fixed
			end
			if (RE.covAcross==0 & RE.d>1)  # exchangeable across?
				accross  = ++i
			for (eq1=1; eq1<=RE.NEq; eq1++) 
				if (RE.covWithin[RE.Eqs[eq1]]==0 & RE.NEff[eq1]>1)  # exchangeable within?
					within = ++i
				for (c1=1; c1<=RE.NEff[eq1]; c1++) 
					for (c2=c1+1; c2<=RE.NEff[eq1]; c2++) 
						if (RE.covWithin[RE.Eqs[eq1]]==1)  # unstructured
							within = ++i
						t = t \ (RE.covWithin[RE.Eqs[eq1]]==2 ? . : within)  # independent?
					end
					for (eq2=eq1+1; eq2<=RE.NEq; eq2++)
						for (c2=1; c2<=RE.NEff[eq2]; c2++) 
							if (RE.FixedRhos[RE.Eqs[eq2],RE.Eqs[eq1]]==.) 
								if (RE.covAcross == 1)  # unstructured
									accross = ++i
								t = t \ accross
							else
								t = t \ .
					end
				end
			end
			RE.dSigdParams = (RE.NSigParams = i) ? designmatrix(editmissing(t,i+1))[|.,.\.,i|] : J(RE.d2, 0, 0)
		end
	end

	Primes = 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109
	if (L>1 & REType != "random" & length(Primes) < sum(NumEff) - 1 - (ghkType=="hammersley" | REType=="hammersley")) 
		errprintf("Number of unobserved variables to simulate too high for Halton-based simulation. Try cmd:retype(random)end.\n")
		return(1001) 
	end
	PrimeIndex = 1

	if (_todo) 
		Scores = scores()
		G = J(d, 1, 0); Scores.ΓScores = Matrix(d*d)  # more than needed
		ncols = d + 1
		if (HasΓ)
			for (c=m=1; m<=d; m++)
				for (i=1; i<=(G[m]=nrows(*ΓIndByEq[m])); i++)
											Scores.ΓScores[c++].M = ncols++
		             Scores.ThetaScores   = one2d
		if (NumCuts) Scores.CutScores     = ncols:ncols+NumCuts-1
		ncols = ncols + NumCuts
		Scores.SigScores = Matrix(L)
		for (l=1; l<=L; l++)
			if (t = ncols((*REs)[l].dSigdParams)) 
											Scores.SigScores[l].M  = ncols:ncols+t-1
				ncols = ncols + t
			end
		S0 = J(base.N, ncols-1, 0)
	end

	for (l=L-1; l; l--) 
		RE = &((*REs)[l])

		RE.N = RE.id[base.N]
    RE.R = NumREDraws[l+1]
		RE.REInds = selectindex(tokens(st_global("cmp_rc"+strofreal(l))) .== "_cons")
		RE.X = RE.RCInds = Matrix(RE.NEq)

		RE.HasRC = 0
    RE.RCk = J(RE.NEq, 1, 0)
		for (start=j=1; j<=RE.NEq; j++) 
			if (HasRE = st_global("cmp_re"+strofreal(l)+"_"+strofreal(RE.Eqs[j])) != "")
				RE.REEqs = RE.REEqs, j
			if (strlen(varnames = st_global("cmp_rc"+strofreal(l)+"_"+strofreal(RE.Eqs[j])))) 
				RE.HasRC = 1
				RE.X[j].M = editmissing(st_data(., varnames, st_global("ML_samp")), 0)  # missing values in X can occur if there's a random coefficient on a var used in one eq and not another, with a distinct sample
				RE.RCk[j] = ncols(RE.X[j].M)
        stop = start + RE.RCk[j]
				RE.RCInds[j].M = start:stop-1
				start = stop + HasRE
			end
		end
		if (RE.HasRC) RE.J_N_NEq_0 = J(base.N, RE.NEq, 0)

		RE.IDRanges = panelsetup(RE.id, 1)
		RE.IDRangesGroup = l==L-1 ? RE.IDRanges : panelsetup(RE.id[(*REs)[l+1].IDRanges[,1]], 1)
    RE.IDRangeLengths = RE.IDRanges[,2] - RE.IDRanges[,1] .+ 1

    RE.Subscript = Matrix(RE.N)
    for (j=RE.N;j;j--)
      RE.Subscript[j].M = RE.IDRanges[j,]', (.\.)

		LevelName = L>2 ? " for level " +strofreal(l) : ""

		if (RE.d <= 2)
			printf("resendRandom effects/coefficients%s modeled with Gauss-Hermite quadrature with %f integration points.\n", LevelName, RE.R)
		else 
			printf("resendRandom effects/coefficients%s modeled with sparse-grid quadrature.\n", LevelName)
			printf("Precision equivalent to that of one-dimensional quadrature with %f integration points.\n", RE.R)
		end

		QuadData = SpGr(RE.d, RE.R)
		RE.R = NDraws = NumREDraws[l+1] = nrows(*QuadData[1])
		if (WillAdapt==0) printf("Number of integration points = %f.\n\n", NDraws)
# inefficiently duplicates draws over groups then parcels them out below
		U = J(RE.N, 1, QuadData[1])
		RE.QuadX = QuadData[1]
		RE.QuadW = QuadData[2]'
		if (WillAdapt) 
			RE.QuadMean = RE.QuadSD = Matrix(RE.N)
			RE.QuadXAdapt = asarray_create("real", l)
			for (j=RE.N; j; j--)
				RE.QuadSD[j].M = J(RE.d, 1, .)
			RE.AdaptiveShift = J(RE.N, NDraws, 0)
			RE.lnnormaldenQuadX = quadrowsum_lnnormalden(RE.QuadX,0)'
			LastlnLThisIter=0; LastlnLLastIter=1
      RE.JN12 = J(RE.N, 1, 2)
      RE.J1R0 = J(1, RE.R, 0)
      RE.JN1pQuadX = J(RE.N, 1, &(RE.QuadX))
		end

		RE.U = Matrix(RE.R)
		RE.TotalEffect = Matrix(RE.R, d)
 		RE.XU         = J(RE.R, sum((RE.NEq:1) .* RE.NEff), NULL)

		S = ((1:RE.N) * NDraws)[RE.id]
		for (r=NDraws; r; r--) 
			RE.U[r].M = U[S, RE.one2d]
			S .-= 1
		end

		RE.lnLlimits = log(smallestdouble()) + 1, log(maxdouble()) - (RE.lnNumREDraws = log(RE.R)) - 1

		RE.lnLByDraw = J(RE.N, RE.R, 0)
	end

	if (L > 1)
		for (l=L; l; l--)
			if (st_global("parse_wexp"+strofreal(l)) != "") 
				RE = &((*REs)[l])
				RE.Weights = st_data(., st_global("cmp_weight"+strofreal(l)), st_global("ML_samp"))  # can't be a view because panelsum() doesn't accept weights in views
				if (l < L) RE.Weights = RE.Weights[RE.IDRanges[,1]]  # get one instance of each group's weight
				if (anyof(("pweight", "aweight"), st_global("parse_wtype"+strofreal(l))))  # normalize pweights, aweights to sum to  # of groups
					if (l == 1)
						REs.Weights = RE.Weights * nrows(RE.Weights) / quadsum(RE.Weights)  # fast way to divide by mean
					else
						for (j=(*REs)[l-1].N; j; j--) 
							S = (*REs)[l-1].IDRangesGroup[j,]', (.\.)
              t = RE.Weights[|S|]
							RE.Weights[|S|] = t * (nrows(t) / quadsum(t))  # fast way to divide by mean
						end
				t = l==L ? RE.Weights : RE.Weights[RE.id]
				WeightProduct = nrows(WeightProduct) ? WeightProduct.* t : t
			end

	ghk_nobs = 0; v = NULL
	remaining = 1:base.N
	d_cens = d_ghk = 0
	while (t = max(remaining))  # build linked list of subviews onto data, each a set of nrows with same indicator combination
		next = v; (v = &(subview())).next = next  # add new subview to linked list
		remaining = remaining .* !(v.subsample = rowmin(indicators .== (v.TheseInds = indicators[t,])))
		v.SubsampleInds = selectindex(v.subsample)
		v.theta = Matrix(d)
		v.QE = diag(2*(v.TheseInds.==cmp_right :| v.TheseInds.==cmp_probity1 :| v.TheseInds.==cmp_frac) .- 1)
		v.N = colsum(v.subsample)
		v.d_uncens = ncols(v.uncens = selectindex(v.TheseInds.==cmp_cont))
		v.halfDmatrix = 0.5 * Dmatrix(v.d_uncens)
		v.d_oprobit = d_oprobit = ncols(v.oprobit = selectindex(v.TheseInds.==cmp_oprobit))
		v.d_trunc = ncols(v.trunc = selectindex(trunceqs))
		v.d_cens = ncols(v.cens = selectindex(v.TheseInds.>cmp_cont :& v.TheseInds.<. :& (v.TheseInds.<mprobit_ind_base :| v.TheseInds.>=roprobit_ind_base)))
		v.censnonfrac           = selectindex(v.TheseInds.>cmp_cont :& v.TheseInds.<. :& (v.TheseInds.<mprobit_ind_base :| v.TheseInds.>=roprobit_ind_base) :& v.TheseInds:!=cmp_frac)
		v.d_frac = ncols(v.frac = ncols(v.cens) ? selectindex(v.TheseInds[v.cens].==cmp_frac) : J(1,0,0))
		d_cens = max((d_cens, v.d_cens))
    d_ghk = max((d_ghk, v.d_trunc, v.d_cens))
		v.dCensNonrobase = ncols(v.cens_nonrobase = selectindex(NonbaseCases :& (v.TheseInds.>cmp_cont :& v.TheseInds.<. :& (v.TheseInds.<mprobit_ind_base :| v.TheseInds.>=roprobit_ind_base))))

		if (v.d_cens)
			v.d_two_cens = ncols(v.two_cens = selectindex((v.TheseInds.==cmp_oprobit :| v.TheseInds.==cmp_int :| (v.TheseInds.==cmp_left :| v.TheseInds.==cmp_right :| v.TheseInds.==cmp_probit :| v.TheseInds.==cmp_probity1) :& trunceqs)[v.cens]))  #indexes *within* list of censored eqs of doubly censored ones
		else
			v.d_two_cens = 0
		
		v.y = v.Lt = v.Ut = v.yL = Matrix(d)
		for (i=d; i; i--) 
			v.y[i].M = y[i].M[v.SubsampleInds]
			if (trunceqs[i]) 
				v.Lt[i].M = Lt[i].M[v.SubsampleInds]
				v.Ut[i].M = Ut[i].M[v.SubsampleInds]
			end
			if (intregeqs[i])
				v.yL[i].M = yL[i].M[v.SubsampleInds]
		end

		if (v.d_cens > 2) 
			v.GHKStart = ghk_nobs + 1
			ghk_nobs = ghk_nobs + v.N
		end
		if (v.d_uncens) v.EUncens =  J(v.N, v.d_uncens, 0)
		if (v.d_cens)   v.ECens  = &J(v.N, v.d_cens  , 0)
		if (NumCuts | sum(intregeqs) | sum(trunceqs)) 
			v.F = &J(v.N, v.dCensNonrobase, .)
			if (sum(trunceqs)) 
				v.Et = &J(v.N, v.d_trunc, .)
				v.Ft = &J(v.N, v.d_trunc, .)
      end
		end

		if (v.d_frac) 
			v.FracCombs = 2*mod(floor(J(1,v.d_frac,0:2^v.d_frac-1)./2:^(v.d_frac-1:0)),2).-1  # matrix whose nrows count from 0 to 2^v.d_frac-1 in +/-1 binary, one digit/column
			v.yProd = v.frac_QE = v.frac_QSig = Matrix(v.NFracCombs = nrows(v.FracCombs))
			for (i=v.NFracCombs; i; i--) 
				if (i < v.NFracCombs) 
				 (v.frac_QE[i].M = I(v.d_cens))[v.frac,v.frac] = diag(v.FracCombs[i,])
				 v.frac_QSig[i].M = QE2QSig(v.frac_QE[i].M)
				end

				v.yProd[i].M = J(v.N, 1, 1)
				for (j=v.d_frac; j; j--)  # make all the combinations of products of frac prob y's and 1-y's
					v.yProd[i].M = v.yProd[i].M .* (v.FracCombs[i,j]==1 ? y[v.cens[v.frac[j]]].M[v.SubsampleInds] : 1.-y[v.cens[v.frac[j]]].M[v.SubsampleInds])
			end
		else
			v.NFracCombs = 1

		v.dPhi_dpE = v.dPhi_dpSig = Matrix(2^v.d_frac)

		if (d_oprobit) 
			l = 1
			if (v.oprobit[1]>1) l = l + colsum(vNumCuts[1:v.oprobit[1]-1])
			v.CutInds = l : l+vNumCuts[v.oprobit[1]]-1
			for (k=2; k<=d_oprobit; k++) 
				l = l + colsum(vNumCuts[v.oprobit[k-1]:v.oprobit[k]-1])
				v.CutInds = v.CutInds, l : l+vNumCuts[v.oprobit[k]]-1
			end
			v.vNumCuts = vNumCuts[v.oprobit]

			v.NumCuts = ncols(v.CutInds)
		else
			v.NumCuts = 0

		v.mprobit = mprobit_group(nrows(MprobitGroupInds))
		for (k=nrows(MprobitGroupInds); k; k--) 
			start = MprobitGroupInds[k, 1]; stop = MprobitGroupInds[k, 2]
			v.mprobit[k].d = d_mprobit = (v.TheseInds[start]<.) * (ncols( mprobit = selectindex(v.TheseInds :& one2d.>=start :& one2d.<=stop) ) - 1)
			if (d_mprobit > 0) 
				v.mprobit[k].out = v.TheseInds[start] - mprobit_ind_base  # eq of chosen alternative
				v.mprobit[k].res = selectindex((v.TheseInds :& one2d.>start  :& one2d.<=stop)[v.cens])  # index in v.ECens for relative differencing results
				v.mprobit[k].in =  selectindex( v.TheseInds :& one2d.>=start :& one2d.<=stop :& one2d:!=v.mprobit[k].out)  # eqs of rejected alternatives
				(v.QE)[mprobit,mprobit] = J(d_mprobit+1, 1, 0), insert(-I(d_mprobit), v.mprobit[k].out-start+1-sum(!v.TheseInds[|start\v.mprobit[k].out|]), J(1, d_mprobit, 1))
			end
		end

		v.N_perm = 1
		if (NumRoprobitGroups) 
			pointer (real rowvector) colvector roprobit
			real rowvector this_roprobit
			pointer (real matrix) colvector perms
			pointer (real matrix) scalar ThesePerms
			real scalar ThisPerm
			
			perms = roprobit = J(NumRoprobitGroups, 1, NULL)
			v.d2_cens = v.d_cens * (v.d_cens + 1) ÷ 2

			for (k=NumRoprobitGroups; k; k--)
				if (ncols(this_roprobit=*(roprobit[k] = &selectindex(v.TheseInds :& one2d.>=RoprobitGroupInds[k,1] :& one2d.<=RoprobitGroupInds[k,2]))))
					v.N_perm = v.N_perm * (nrows(*(perms[k] = &PermuteTies(reverse ? v.TheseInds[this_roprobit] : -v.TheseInds[this_roprobit]))))
			
			v.roprobit_QE = v.roprobit_Q_Sig = J(i=v.N_perm, 1, NULL)
			for (; i; i--)  # combinations of perms across multiple roprobit groups
				j = i - 1
				t = I(d)
				for (k = NumRoprobitGroups; k; k--) 
					if (d_roprobit = ncols(this_roprobit = *roprobit[k])) 
						ThisPerm = mod(j, nrows(*(ThesePerms=perms[k]))) + 1
						t[this_roprobit, this_roprobit] = 
							J(d_roprobit, 1, 0), (I(d_roprobit)[,(*ThesePerms)[|ThisPerm, 2 \ ThisPerm, .           |]] - 
																		I(d_roprobit)[,(*ThesePerms)[|ThisPerm, 1 \ ThisPerm, d_roprobit-1|]] )
						j = (j - ThisPerm + 1) / nrows(*ThesePerms)
					end
				(v.roprobit_Q_Sig)[i] = &QE2QSig(*((v.roprobit_QE)[i] = &t[v.cens, v.cens_nonrobase]))
			end
		end

    v.NotBaseEq = v.TheseInds .< mprobit_ind_base :| v.TheseInds .>= roprobit_ind_base

		if (v.d_trunc) 
			v.one2d_trunc = 1:v.d_trunc
			v.SigIndsTrunc = vSigInds(v.trunc, d)

			if (v.d_trunc > 2) 
				v.GHKStartTrunc = ghk_nobs + 1
				ghk_nobs = ghk_nobs + v.N
			end
		end

		if _todo 
			v.XU = Vector{Matrix{T}}(L-1)
			for l ∈ 1:L-1
				v.XU[l].M = Matrix{Matrix{T}}(undef, size((*REs)[l].XU)...)
			end
		end

		if (_todo)  # pre-compute stuff for scores
			v.Scores = scorescol(L)
      sTScores=Matrix(L); sΓScores=Matrix(sum(G))
			for (l=L; l; l--) 
				v.Scores[l].M = scores(NumREDraws[l])
				for (r=NumREDraws[l]; r; r--) 
					v.Scores[l].M[r].ΓScores = sΓScores
					v.Scores[l].M[r].TScores = sTScores  # last entry holds scores of base-level Sig parameters not T
				end
			end
			v.Scores.M.SigScores = Matrix(L)
			v.id =  Matrix(L-1)
			for (l=L-1; l; l--)
				v.id[l].M = (*REs)[l].id[v.SubsampleInds,]

			if (nrows(WeightProduct)) v.WeightProduct = WeightProduct[v.SubsampleInds,]
				
			for (l=L-1; l; l--)
				for (r=NumREDraws[L]; r; r--)
					v.Scores[L].M[r].TScores[l].M = J(v.N, (*REs)[l].d2, 0)

			v.dΩdΓ = Matrix{Matrix{T}}(undef,d,d)
			
			v.SigIndsUncens = vSigInds(v.uncens, d)
			v.cens_uncens = v.cens, v.uncens
			v.J_d_uncens_d_cens_0 = J(v.d_uncens, v.d_cens, 0)
			v.J_d_cens_d_0 = J(v.d_cens, d, 0)
			v.J_d2_cens_d2_0 = J(v.d_cens*(v.d_cens+1) ÷ 2, d2, 0)				

			if (v.d_uncens) 
				v.dphi_dE = J(v.N, d, 0)
				v.dphi_dSig = J(v.N, d2, 0)
				v.EDE = J(v.N, v.d_uncens*(v.d_uncens+1)*.5, 0)
			else
				v.dPhi_dE = J(v.N, d, 0)

			if (v.d_two_cens | v.d_trunc) 
				v.dPhi_dpF = J(v.N, v.dCensNonrobase, 0)
				if (v.d_uncens==0)
					v.dPhi_dF = J(v.N, d, 0)
				if (v.d_trunc) 
					v.dPhi_dEt = J(v.N, d,  0)
					v.dPhi_dSigt = J(v.N, d2, 0)
				end
			end

			if (v.d_cens & v.d_uncens==0)
				v.dPhi_dSig = J(v.N, d2, 0)
			if (v.d_cens & v.d_uncens) 
				v.dPhi_dpE_dSig = J(v.N, d2, 0)
				v._dPhi_dpE_dSig = J(v.N, (v.d_cens+v.d_uncens)*(v.d_cens+v.d_uncens+1)÷2, 0)
			end
			if (v.d_two_cens & v.d_uncens) 
				v.dPhi_dpF_dSig = J(v.N, d2, 0)
				v._dPhi_dpF_dSig = J(v.N, (v.d_cens+v.d_uncens)*(v.d_cens+v.d_uncens+1)÷2, 0)
			end
			if (NumCuts)
				v.dPhi_dcuts = J(v.N, NumCuts, 0)
			
			if (v.d_cens)
				v.CensLTInds = vech(colshape(1:v.d_cens*v.d_cens, v.d_cens)')

			if (d_oprobit) 
				varnames = ""
				for (k=1; k<=d_oprobit; k++) 
					stata("unab yis: _cmp_y" + strofreal(v.oprobit[k]) + "_*")
					varnames = varnames + " " + st_local("yis")
				end
				st_view(Yi, ., tokens(varnames))
				st_select(v.Yi, Yi, v.subsample)
			end

			v.QSig = QE2QSig(v.QE)'
			v.SigIndsCensUncens = vSigInds(v.cens_uncens, d)
			v.dSig_dLTSig = Dmatrix(v.d_cens + v.d_uncens)
		end
	end
	subviews = v

	if (_todo)
		for (l=L-1;l;l--)
			BuildXU(l)
  return(0)
end


void SaveSomeResults() 
	pointer (struct RE scalar) scalar RE; real scalar L, l, j, k_aux_nongamma; real matrix means, ses; string matrix colstripe, _colstripe

	st_matrix("e(MprobitGroupEqs)", MprobitGroupInds)
	st_matrix("e(ROprobitGroupEqs)", RoprobitGroupInds)

	if ((L =st_numscalar("e(L)")) == 1)
		st_matrix("e(Sigma)", REs.Sig)
	else 
		for (l=L; l; l--) 
			RE = &((*REs)[l])
			st_matrix("e(Sigma"+(l<L ?strofreal(l):"")+")", RE.Sig)
			if (l<L & (AdaptivePhaseThisEst | Adapted))  # means and ses don't exist if iter() option stopped search before adaptive phase
				ses = means = J(RE.N, RE.d, 0)
				for (j=RE.N; j; j--) 
					means[j,] = RE.QuadMean[j].M
					ses  [j,] = RE.QuadSD[j].M'
				end
				st_matrix("e(REmeans"+strofreal(l)+")", means * RE.T)
				st_matrix("e(RESEs"  +strofreal(l)+")", ses   * RE.T)
				colstripe = tokens(st_global("cmp_rceq"+strofreal(l)))', tokens(st_global("cmp_rc"+strofreal(l)))'
				st_matrixcolstripe("e(REmeans"+strofreal(l)+")", colstripe)
				st_matrixcolstripe("e(RESEs"  +strofreal(l)+")", colstripe)
			end
		end
		if (nrows(WeightProduct))
			st_numscalar("e(N)", sum(WeightProduct))
	end

	if (HasΓ) 
		real scalar eq, d, k, NumCoefs, rows_dbr_db, cols_dbr_db, k_gamma
		real matrix Beta, BetaInd, ΓInd, REInd, dBeta_dB, dBeta_dΓ, dbr_db, dΩ_dSig, V, br, sig, ρ, Rho, invΓ, Ω, NumEff
		real rowvector eb, p
		real colvector keep
		string rowvector eqnames
		pragma unset p
		
		colstripe = J(0, 1, ""); _colstripe = J(0, 2, "")
		V = st_matrix("e(V)")
		BetaInd = st_matrix("cmpBetaInd"); ΓInd = st_matrix("cmpΓInd")
		invΓ = (*REs)[L].invΓ'
		Beta = invΓ * st_matrix(st_local("Beta"))
		d = nrows(Beta); k = ncols(Beta)
		br = colshape(Beta, 1)
		eb = st_matrix("e(b)")
		k_aux_nongamma = st_numscalar("e(k_aux)") - (k_gamma = st_numscalar("e(k_gamma)"))
		dBeta_dB = invΓ  # I(k); dBeta_dΓ = invΓ  # Beta'

		dbr_db = J(nrows(dBeta_dB), 0, 0)
		for (eq=d; eq; eq--)
			dbr_db = dBeta_dΓ[, ΓInd[selectindex(ΓInd[,2].==eq),1] .+ (eq-1)*d], dbr_db        
		for (eq=d; eq; eq--)
			dbr_db = dBeta_dB    [, BetaInd [selectindex(BetaInd [,2].==eq),1] .+ (eq-1)*k], dbr_db        

		keep = selectindex(rowsum(dbr_db:!=0).>0)
		br = br[keep]'
		dbr_db = dbr_db[keep,]
    rows_dbr_db = nrows(dbr_db); cols_dbr_db = ncols(dbr_db)

		eqnames = tokens(st_global("cmp_eq"))
		for (eq=d; eq; eq--)
			colstripe = J(k, 1, eqnames[eq]) \ colstripe
		colstripe = (colstripe, J(d, 1, tokens(st_local("xvarsall"))'))[keep,]

		if (NumCuts) 
			br = br, eb[|ncols(eb)-k_aux_nongamma+1 \ ncols(eb)-k_aux_nongamma+NumCuts|]
			colstripe = colstripe \ st_matrixcolstripe("e(b)")[|ncols(eb)-k_aux_nongamma+1, . \ ncols(eb)-k_aux_nongamma+NumCuts,.|]
			dbr_db = blockdiag(dbr_db, I(NumCuts))
		end		

		NumEff = J(0, d, 0)
		for (l=1; l<=L; l++) 
			RE = &((*REs)[l])
			REInd = st_matrix("cmpREInd"+strofreal(l))
			k = colmax(REInd[,2])
			dΩ_dSig = (invΓ  # I(k))[, (REInd[,1].-1)*k + REInd[,2]]'
			st_matrix("e(Ω"+(l<L ? strofreal(l) : "") + ")", Ω = quadcross(dΩ_dSig, RE.Sig) * dΩ_dSig)
			Rho = corr(Ω); ρ = nrows(Rho)>1 ? vech(Rho[|2,1 \ .,ncols(Rho)-1|])' : J(1,0,0)
			sig = sqrt(diag(Ω))'
			dΩ_dSig = edittozero(pinv(editmissing(dSigdsigrhos(SigXform, sig, Ω, ρ, Rho),0)),10) * QE2QSig(dΩ_dSig) * dSigdsigrhos(SigXform, RE.sig, RE.Sig, RE.ρ, Rho) * RE.dSigdParams
			keep = selectindex((((sig:!=.) .* (sig.>0)), (ρ:!=.)) .* (rowsum(dΩ_dSig:!=0).>0)')'
			br = br, (SigXform ? log(sig), atanh(ρ) : sig, ρ)[keep]
			_colstripe = _colstripe \ ((tokens(st_local("sigparams"+strofreal(l)))' \ tokens(st_local("rhoparams"+strofreal(l)))')[keep] , J(nrows(keep), 1, "_cons"))
			dbr_db = blockdiag(dbr_db, dΩ_dSig[keep,])

			if (RE.NSigParams) 
        keep = colshape(rowsum(dΩ_dSig[|.,.\k*d,.|]:!=0).>0, k)  # get retained sig params by eq
        NumEff = NumEff \ rowsum(keep)'
        for (j=d; j; j--)
          st_global("e(EffNames_reducedform"+strofreal(l)+"_"+strofreal(j)+")", invtokens(tokens(st_local("cmp_rcu"+strofreal(l)))[selectindex(keep[j,])]))
      else
        NumEff = NumEff \ J(1,d,0)

			st_matrix("e(fixed_sigs_reducedform"+strofreal(l)+")", J(1, d, .)) 
			st_matrix("e(fixed_rhos_reducedform"+strofreal(l)+")", J(d, d, .)) 
		end
		st_matrix("e(NumEff_reducedform)", NumEff)
		st_numscalar("e(k_sigrho_reducedform)", nrows(_colstripe))
		colstripe = colstripe \ _colstripe
		
		NumCoefs = nrows(BetaInd) - 1
		BetaInd  = runningsum(colsum( BetaInd[|2,2\.,.|]#J(1,d,1) .== (1:d))')
		ΓInd = runningsum(colsum(ΓInd[|2,2\.,.|]#J(1,d,1) .== (1:d))') .+ NumCoefs
		BetaInd   = (0        \  BetaInd[|.\d-1|]).+1,  BetaInd
		ΓInd  = (NumCoefs \ ΓInd[|.\d-1|]).+1, ΓInd
		for (eq=1; eq<=d; eq++) 
			if (ΓInd[eq,2] >= ΓInd[eq,1]) p = p, ΓInd[eq,1]:ΓInd[eq,2]
			if ( BetaInd[eq,2] >=  BetaInd[eq,1]) p = p,  BetaInd[eq,1]: BetaInd[eq,2]
		end
		if (ncols(p)<ncols(eb))
			p = p, ncols(p)+1 : ncols(eb)

		st_matrix("e(br)", br)
		st_matrix("e(Vr)", dbr_db * V * dbr_db')
		st_matrix("e(_p)", p)
		st_matrixcolstripe("e(br)", colstripe)
		st_matrixcolstripe("e(Vr)", colstripe)
		st_matrixrowstripe("e(Vr)", colstripe)
    st_matrix("e(invΓ)", (*REs)[L].invΓ)
    st_matrix("e(dbr_db)", k_aux_nongamma ? blockdiag(blockdiag(invΓ, J(0, k_gamma, 0)), dbr_db[|rows_dbr_db+1, cols_dbr_db+1 \ .,.|]) : invΓ)
	end
end

end