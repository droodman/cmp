*! cmp 6.8.0 5 March 2015
*! Copyright David Roodman 2007-15. May be distributed free.
cap program drop cmp_lf1
program define cmp_lf1
	version 10.0
	args todo b lnf
	tokenize `0'
	macro shift 3
	local `=`todo'*$cmp_num_scores+1' // zap extra arg passed to lfx evaluators
	tempname t cuts rc lnsigAccross lnsigWithin atanhrhoAccross atanhrhoWithin
	tempvar theta

	local i 0
	while `i' < $cmp_d {
		cap drop _cmp_theta`++i'
		mleval _cmp_theta`i' = `b', eq(`i')
	}

	if $cmpHasGamma {
		qui recode _cmp_theta* (. = 0) // only time missing values would appear and be used is when multiplied by invGamma with 0's in corresponding entries
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
			mat `cuts'[1,`eq'] = minfloat()
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
				  if $cmpSigXform==0 & `lnsigWithin'==0 {
						replace `lnf' = .
						exit
					}
					mat `sig`l'' = nullmat(`sig`l''), `=cond($cmpSigXform, exp(`lnsigWithin'), `lnsigWithin')'
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

	if $parse_L > 1 {
		tempname lnfi
		scalar `lnfi' = . // create it in case cmp_lnL() doesn't, when it returns "."
		mata cmp_lnL(`todo', "", "`*'")
		qui replace `lnf' = `lnfi'/$cmpN
	}
	else mata cmp_lnL(`todo', "`lnf'", "`*'")
end
