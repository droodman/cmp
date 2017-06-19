*! cmp 6.4.0 5 July 2013
*! Copyright David Roodman 2007-13. May be distributed free.
cap program drop cmp_lf
program define cmp_lf
	version 10.0

	args lnf
	macro shift
	tempname t sig cuts rc lnsigAccross lnsigWithin atanhrhoAccross atanhrhoWithin

	local i 0
	forvalues eq1=1/$cmp_d {
		cap drop _cmp_theta`eq1'
		gen double _cmp_theta`eq1' = ``++i''
	}

	if $cmpHasGamma {
		tempname Gamma
		mat `Gamma' = J($cmp_d, $cmp_d, 0)
		local GammaIndRow 1
		forvalues eq1=1/$cmp_d {
			while cmpGammaInd[`GammaIndRow', 1]==`eq1' {
				mat `Gamma'[cmpGammaInd[`GammaIndRow++',2], `eq1'] = ``++i''[$cmp_n]
			}
		}
	}

	mat `cuts' = J($cmp_d, $cmp_max_cuts+2, .)
	forvalues eq=1/$cmp_d {
		if cmp_num_cuts[`eq',1] {
			mat `cuts'[`eq',1] = minfloat()
			forvalues cut=1/`=cmp_num_cuts[`eq',1]' {
				scalar `t' = ``++i''[$cmp_n]
				qui if ${cmp_truncreg`eq'} {
					count if _cmp_ind`eq' & ((${cmp_Lt`eq'}<. & `t'<${cmp_Lt`eq'}) | `t'>${cmp_Ut`eq'})
					if r(N) {
						replace `lnf' = .
						exit
					}
				}
				mat `cuts'[`eq',`cut'+1] = `t' 
			}
		}
	}

	forvalues l=1/$parse_L {
		tempname sig`l' atanhrho`l'
		local sigs `sigs' `sig`l''
		local atanhrhos `atanhrhos' `atanhrho`l''

		if "${cmp_cov`l'}" == "exchangeable" {
			scalar `lnsigAccross' = ``++i''[$cmp_n]
			scalar `lnsigWithin' = `lnsigAccross'
		}
		forvalues eq=1/$cmp_d {
			if "${cmp_cov`l'_`eq'}"=="exchangeable" {
				if "${cmp_cov`l'}" != "exchangeable" {
					scalar `lnsigWithin' = ``++i''[$cmp_n]
				}
			}
			forvalues c=1/`=cmp_NumEff[`l', `eq']' {
				if  cmp_fixed_sigs`l'[1,`eq'] == . {
					if inlist("${cmp_cov`l'_`eq'}", "independent", "unstructured") & "${cmp_cov`l'}" != "exchangeable" {
						 scalar `lnsigWithin' = ``++i''[$cmp_n]
					}
				    mat `sig`l'' = nullmat(`sig`l''), exp(`lnsigWithin')
				}
				else mat `sig`l'' = nullmat(`sig`l''), cmp_fixed_sigs`l'[1,`eq']
			}
		}

		if "${cmp_cov`l'}" == "exchangeable" & $cmp_d > 1 {
			scalar `atanhrhoAccross' = ``++i''[$cmp_n]
		}
		forvalues eq1=1/$cmp_d {
			if "${cmp_cov`l'_`eq1'}"=="independent" {
				scalar `atanhrhoWithin' = 0
			}
			else if "${cmp_cov`l'_`eq1'}"=="exchangeable" & cmp_NumEff[`l', `eq1'] > 1 {
				scalar `atanhrhoWithin' = ``++i''[$cmp_n]
			}
			forvalues c1=1/`=cmp_NumEff[`l', `eq1']' {
				forvalues c2=`=`c1'+1'/`=cmp_NumEff[`l', `eq1']' {
					if "${cmp_cov`l'_`eq1'}" == "unstructured" {
						scalar `atanhrhoWithin' = ``++i''[$cmp_n]
					}
					mat `atanhrho`l'' = nullmat(`atanhrho`l''), `atanhrhoWithin'
				}
				forvalues eq2=`=`eq1'+1'/$cmp_d {
					forvalues c2=1/`=cmp_NumEff[`l', `eq2']' {
						if cmp_fixed_rhos`l'[`eq2',`eq1'] == . {
							if "${cmp_cov`l'}" == "unstructured" {
								 scalar `atanhrhoAccross' = ``++i''[$cmp_n]
							}
							mat `atanhrho`l'' = nullmat(`atanhrho`l''), `atanhrhoAccross'
						}
						else mat `atanhrho`l'' = nullmat(`atanhrho`l''), cmp_fixed_rhos`l'[`eq2',`eq1']
					}
				}
			}
		}
	}
	mata cmp_lnL(0, "`lnf'", "")
end
