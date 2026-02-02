*! cmp_lf1.ado - ml lf1 evaluator wrapper for Julia backend
*! Part of cmp package
*! This file bridges Stata's ml command with the Julia CMP.lf1! function

program cmp_lf1
    version 15
    args todo b lnfj g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11 g12 g13 g14 g15 g16 g17 g18 g19 g20

    local neq = $cmp_d

    // Use mleval to compute linear predictions (X*b) for each equation
    forvalues eq = 1/`neq' {
        tempvar theta`eq'
        mleval `theta`eq'' = `b', eq(`eq')
    }

    // Transfer linear predictions to Julia
    forvalues eq = 1/`neq' {
        local tvar `theta`eq''
        _jl: _cmp_model.REs[end].theta[`eq'] = st_data("`tvar'")
    }

    // Extract auxiliary parameters (gamma, cuts, sig/rho) using mleval
    local n_aux = $cmp_num_scores - $cmp_d
    if `n_aux' > 0 {
        _jl: _cmp_aux_params = zeros(`n_aux')

        local aux_idx = 1
        local i = `neq' + 1
        forvalues eq = `i'/$cmp_num_scores {
            tempvar auxvar
            mleval `auxvar' = `b', eq(`eq')
            _jl: _cmp_aux_params[`aux_idx'] = st_data("`auxvar'")[1]
            local ++aux_idx
        }
    }

    tempname bmat
    mat `bmat' = `b'

    // Call Julia lf1!
    if `todo' {
        _jl: _cmp_lnfj, _cmp_scores = CMP.lf1!(_cmp_model, 1, vec(sf_get_matrix("`bmat'")))
        _jl: sf_store(".", "`lnfj'", _cmp_lnfj)

        // Write scores
        local total_eq = $cmp_num_scores
        forvalues eq = 1/`total_eq' {
            local gvar : word `eq' of `g1' `g2' `g3' `g4' `g5' `g6' `g7' `g8' `g9' `g10' `g11' `g12' `g13' `g14' `g15' `g16' `g17' `g18' `g19' `g20'
            if "`gvar'" != "" {
                _jl: sf_store(".", "`gvar'", _cmp_scores[:, `eq'])
            }
        }
    }
    else {
        _jl: _cmp_lnfj, _ = CMP.lf1!(_cmp_model, 0, vec(sf_get_matrix("`bmat'")))
        _jl: sf_store(".", "`lnfj'", _cmp_lnfj)
    }
end
