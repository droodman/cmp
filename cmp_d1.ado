*! cmp 6.4.0 5 July 2013
*! Copyright David Roodman 2007-13. May be distributed free.
cap program drop cmp_d1
program define cmp_d1
	version 10.0

	forvalues eq=1/$cmp_num_scores {
		local scargs `scargs' sc`eq'
	}
	args todo b lnfi g negH `scargs'
	tokenize `0'
	macro shift 5
	tempname t cuts rc _g lnsigAccross lnsigWithin atanhrhoAccross atanhrhoWithin
	tempvar theta

	local i 0
	while `i' < $cmp_d {
		cap drop _cmp_theta`++i'
		mleval _cmp_theta`i' = `b', eq(`i')
	}

	if $cmpHasGamma {
		tempname Gamma
		mat `Gamma' = J($cmp_d, $cmp_d, 0)
		local GammaIndRow 1
		forvalues eq1=1/$cmp_d {
			while cmpGammaInd[`GammaIndRow', 1]==`eq1' {
				mleval `theta' = `b', eq(`++i') scalar
				mat `Gamma'[cmpGammaInd[`GammaIndRow++',2], `eq1'] = `theta'
			}
		}
	}

	mat `cuts' = J($cmp_max_cuts+2, $cmp_d, .)
	forvalues eq=1/$cmp_d {
		if cmp_num_cuts[`eq',1] {
			mat `cuts'[1, `eq'] = minfloat()
			forvalues cut=1/`=cmp_num_cuts[`eq',1]' {
				mleval `t' = `b', eq(`++i') scalar
				qui if ${cmp_truncreg`eq'} {
					count if _cmp_ind`eq' & ((${cmp_Lt`eq'}<. & `t'<${cmp_Lt`eq'}) | `t'>${cmp_Ut`eq'})
					if r(N) {
						replace `lnf' = .
						exit
					}
				}
				mat `cuts'[`cut'+1,`eq'] = `t' 
			}
		}
	}

	forvalues l=1/$parse_L {
		tempname sig`l' atanhrho`l'
		local sigs `sigs' `sig`l''
		local atanhrhos `atanhrhos' `atanhrho`l''

		if "${cmp_cov`l'}" == "exchangeable" {
			mleval `lnsigAccross' = `b', eq(`++i') scalar
			scalar `lnsigWithin' = `lnsigAccross'
		}
		forvalues eq=1/$cmp_d {
			if "${cmp_cov`l'_`eq'}"=="exchangeable" {
				if "${cmp_cov`l'}" != "exchangeable" {
					mleval `lnsigWithin' = `b', eq(`++i') scalar
				}
			}
			forvalues c=1/`=cmp_NumEff[`l', `eq']' {
				if  cmp_fixed_sigs`l'[1,`eq'] == . {
					if inlist("${cmp_cov`l'_`eq'}", "independent", "unstructured") & "${cmp_cov`l'}" != "exchangeable" {
						 mleval `lnsigWithin' = `b', eq(`++i') scalar
					}
				    mat `sig`l'' = nullmat(`sig`l''), exp(`lnsigWithin')
				}
				else mat `sig`l'' = nullmat(`sig`l''), cmp_fixed_sigs`l'[1,`eq']
			}
		}

		if "${cmp_cov`l'}" == "exchangeable" & $cmp_d > 1 {
			mleval `atanhrhoAccross' = `b', eq(`++i') scalar
		}
		forvalues eq1=1/$cmp_d {
			if "${cmp_cov`l'_`eq1'}"=="independent" {
				scalar `atanhrhoWithin' = 0
			}
			else if "${cmp_cov`l'_`eq1'}"=="exchangeable" & cmp_NumEff[`l', `eq1'] > 1 {
				mleval `atanhrhoWithin' = `b', eq(`++i') scalar
			}
			forvalues c1=1/`=cmp_NumEff[`l', `eq1']' {
				forvalues c2=`=`c1'+1'/`=cmp_NumEff[`l', `eq1']' {
					if "${cmp_cov`l'_`eq1'}" == "unstructured" {
						mleval `atanhrhoWithin' = `b', eq(`++i') scalar
					}
					mat `atanhrho`l'' = nullmat(`atanhrho`l''), `atanhrhoWithin'
				}
				forvalues eq2=`=`eq1'+1'/$cmp_d {
					forvalues c2=1/`=cmp_NumEff[`l', `eq2']' {
						if cmp_fixed_rhos`l'[`eq2',`eq1'] == . {
							if "${cmp_cov`l'}" == "unstructured" {
								 mleval `atanhrhoAccross' = `b', eq(`++i') scalar
							}
							mat `atanhrho`l'' = nullmat(`atanhrho`l''), `atanhrhoAccross'
						}
						else mat `atanhrho`l'' = nullmat(`atanhrho`l''), cmp_fixed_rhos`l'[`eq2',`eq1']
					}
				}
			}
		}
	}

	mata (void) cmp_lnL(`todo', "", "`*'")
	if $parse_L==1 mlsum `lnfi' = _cmp_lnfi

	if `todo' {
		tempname _g
		local GammaIndRow 1
		local i 1
		forvalues eq=1/$cmp_d {
			while cmpGammaInd[`GammaIndRow', 1]==`eq' {
				mlsum `t' = `sc`i++''
				mat `_g' = nullmat(`_g'), `t'
				local ++GammaIndRow
			}
			mlvecsum `lnfi' `t' = `sc`i'' if _cmp_ind`eq', eq(`i')
			local ++i
			mat `_g' = nullmat(`_g'), `t'
		}
		forvalues eq=`i'/$cmp_num_scores {
			mlsum `t' = `sc`eq''
			mat `_g' = `_g', `t'
		}
		mat `g' = `_g'
	}
end
