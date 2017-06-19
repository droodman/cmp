*! cmp 6.8.0 5 March 2015
*! Copyright David Roodman 2007-13. May be distributed free.
cap program drop cmp_clear
program define cmp_clear
	version 10.0
	forvalues l=1/0$parse_L {
		cap mat drop cmp_fixed_sigs`l'
		cap mat drop cmp_fixed_rhos`l'
		cap mat drop cmpREInd`l'
	}
	foreach mat in cmp_mprobit_group_inds cmp_roprobit_group_inds cmp_num_cuts cmpEqs cmp_nonbase_cases cmp_RC_T cmp_trunceqs cmp_intregeqs cmp_NumEff cmpGammaInd cmpBetaInd {
		cap mat drop `mat'
	}
	foreach vars in lnfi y ind id u Ut Lt theta {
		cap drop _cmp_`vars'*
	}
	cap drop _mp_cmp*
	cap drop _cmp_y*_*
	cap drop _cmp_weight*
	macro drop ml_*
	macro drop parse_*
	forvalues eq=1/0$cmp_d {
		macro drop cmp_y`eq'_revar
		macro drop cmp_x`eq'_revar
		macro drop cmp_xo`eq'_revar
		macro drop cmp_xe`eq'_revar
		cap label drop cmp_y`eq'_label
		cap mat drop cmp_cat`eq'
	}
	foreach global in REDraws XVars HasGamma ParamsDisplay Obs1 N SigXform AnyOprobit {
		macro drop cmp`global'
	}
	foreach global in d truncreg* intreg* y* gammaparams* tot_cuts max_cuts eq* x* mprobit_ind_base roprobit_ind_base num_mprobit_groups num_roprobit_groups ///
			reverse rc* re* id* L* Lt* Ut* cov* NSimEff num_scores lf num_coefs k probity1 IntMethod {
		macro drop cmp_`global'
	}
	foreach var in _ghk_p _ghkAnti _ghkDraws _ghkType _ghkScramble _Sig _subviews _first_call _d _NumCuts _NumScores _interactive _X _num_mprobit_groups _Eqs _GammaEqs _Quadrature, _Adaptive ///
			_mprobit_ind_base _mprobit_group_inds _intreg _Weights _NumEff _REType _REAnti _REScramble _Cns _R _X _L _colnames _t _p _Y _trunceqs _HasGamma _IntMethod ///
			_GammaInds __GammaInds _HasRC _NScores _NumREDraws _REs __NumREDraws _intregeqs _nonbase_cases _num_roprobit_groups _reverse _roprobit_group_inds _roprobit_ind_base _trunc _vNumCuts l ///
			__EmpiricalBayesLevel _EmpiricalBayesCoefInd _QuadTol _QuadIter {
		cap mata mata drop `var'
	}
	ml clear
end
