*! cmp_lf1.ado - ml lf1 evaluator wrapper for Julia backend
*! Part of cmp package
*! This file bridges Stata's ml command with the Julia CMP.lf1! function

program cmp_lf1
    version 15
    args todo b lnfj g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11 g12 g13 g14 g15 g16 g17 g18 g19 g20

    local neq = $cmp_d

    // Use mleval to compute linear predictions (X*b) for each equation
    // mleval creates Stata variables containing the linear predictions
    forvalues eq = 1/`neq' {
        tempvar theta`eq'
        mleval `theta`eq'' = `b', eq(`eq')
    }

    // Transfer linear predictions to Julia
    forvalues eq = 1/`neq' {
        _jl: _cmp_model.REs[end].theta[`eq'] = sf_get_var("`theta`eq''")
    }

    // Extract auxiliary parameters (gamma, cuts, sig/rho) using mleval
    // These are scalar equations, so mleval returns a constant
    // Store them in a Julia vector for lf1! to use

    local i = `neq' + 1  // Start after main equations

    // Initialize auxiliary parameter vector in Julia
    local n_aux = $cmp_num_scores - $cmp_d
    if `n_aux' > 0 {
        _jl: _cmp_aux_params = zeros(`n_aux')

        local aux_idx = 1
        forvalues eq = `i'/$cmp_num_scores {
            tempvar aux`eq'
            mleval `aux`eq'' = `b', eq(`eq')
            // For scalar equations, all obs have same value; take first
            _jl: _cmp_aux_params[`aux_idx'] = sf_get_var("`aux`eq''")[1]
            local ++aux_idx
        }
    }

    tempname bmat
    mat `bmat' = `b'

    // Pass parameter vector to Julia and compute likelihood
    // _cmp_model is the global Julia model object initialized in cmp.ado
    // CMP.lf1! returns (lnfj_vec, scores_matrix) where:
    //   - lnfj_vec is observation-level log-likelihood
    //   - scores_matrix has one column per equation (only if todo==1)

    if `todo' {
        // Compute likelihood and scores
        _jl: _cmp_lnfj, _cmp_scores = CMP.lf1!(_cmp_model, 1, sf_get_matrix("`bmat'"))

        // Write observation-level log-likelihood to Stata variable
        _jl: sf_store(".", "`lnfj'", _cmp_lnfj)

        // Write equation-level scores to Stata variables
        // For lf1, scores are per-equation (including auxiliary equations)
        local total_eq = $cmp_num_scores
        forvalues eq = 1/`total_eq' {
            local gvar : word `eq' of `g1' `g2' `g3' `g4' `g5' `g6' `g7' `g8' `g9' `g10' `g11' `g12' `g13' `g14' `g15' `g16' `g17' `g18' `g19' `g20'
            if "`gvar'" != "" {
                _jl: sf_store(".", "`gvar'", _cmp_scores[:, `eq'])
            }
        }
    }
    else {
        // Compute likelihood only
        _jl: _cmp_lnfj = CMP.lf1!(_cmp_model, 0, sf_get_matrix("`bmat'"))

        // Write observation-level log-likelihood to Stata variable
        _jl: sf_store(".", "`lnfj'", _cmp_lnfj)
    }
end
