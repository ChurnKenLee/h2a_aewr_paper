import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import itertools
    import polars as pl
    import polars.selectors as cs
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    import lineax as lx
    from jax.flatten_util import ravel_pytree
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt

    return (
        cs,
        itertools,
        jax,
        jnp,
        lx,
        mo,
        np,
        optax,
        pickle,
        pl,
        pyprojroot,
        ravel_pytree,
    )


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    code_path = root_path / 'code'
    return binary_path, code_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load and preprocess data for JAX
    """)
    return


@app.cell
def _(binary_path, pl):
    # BEA farm employment in 2011 as baseline for calculating shares
    bea = pl.read_parquet(binary_path / 'bea_farm_nonfarm_emp.parquet')
    bea = bea.filter(
        pl.col('year')==2011
    ).rename({
        'bea_farm_emp':'bea_farm_emp_2011',
        'county_fips':'county_ansi'
    }).drop([
        'bea_nonfarm_emp', 'year'
    ])
    return (bea,)


@app.cell
def _(binary_path, pl):
    h2a = pl.read_parquet(
        binary_path / 'h2a_aggregated.parquet'
    ).with_columns(
        (pl.col('state_fips_code') + pl.col('county_fips_code')).alias('county_ansi'),
        pl.col('year').cast(pl.Int32).alias('year')
    ).rename({
        'nbr_workers_certified_start_year':'h2a_certified'
    }).select([
        'year', 'county_ansi', 'h2a_certified',
    ]).filter(
        pl.col('year') >= 2008,
        pl.col('year') <= 2011
    )
    return (h2a,)


@app.cell
def _(binary_path, cs, pl):
    # Grab climate variables we actually want to use
    climate = pl.read_parquet(binary_path / "county_h2a_prediction_climate_gdd_annual.parquet")
    climate = climate.select([
        'year', 'fips',
        'tmin_ann', 'tmax_ann', 'tavg_ann', cs.contains('days_D'),
        'prcp_ann', 'prcp_gs', 'prcp_spring', 'n_wet_days', 'max_cdd_gs', cs.contains('days_P')
    ]).filter(
        pl.col('year') >=2008,
        pl.col('year') <= 2011
    ).rename({
        'fips':'county_ansi'
    })
    return (climate,)


@app.cell
def _(binary_path, pl):
    soil = pl.read_parquet(binary_path / "county_h2a_prediction_gnatsgo.parquet")
    return (soil,)


@app.cell
def _(climate):
    cont_cols = climate.columns.copy()
    cont_cols.remove('year')
    cont_cols.remove('county_ansi')
    cont_cols = cont_cols + ['slope_r', 'resdept_r']
    cat_cols = [
        'taxorder', 'drainagecl', 'nirrcapcl'
    ]
    return cat_cols, cont_cols


@app.cell
def _(itertools, jnp, mo, np, pl):
    @mo.persistent_cache
    def prep_jax_composition_arrays(h2a, bea, soil, climate, cont_cols, cat_cols, max_order):
        """
        Merge, standardize continuous variables, map categorical variables
        """
        merged = soil.join(
            climate, on='county_ansi', how='inner'
        )
        merged = merged.join(
            h2a, on=['county_ansi', 'year'], how='left'
        ).with_columns(
            pl.col('h2a_certified').fill_null(0).alias('h2a_certified')
        )
        merged = merged.join(
            bea, on='county_ansi', how='inner'
        )

        # Drop missing or zero employment counties entirely)
        merged = merged.filter(
            pl.col('bea_farm_emp_2011').is_not_null() & (pl.col('bea_farm_emp_2011') > 0)
        )

        # Optional: cap target rate at a reasonable ceiling like 2.0 to prevent 
        # extreme BEA data artifacts from destroying gradient magnitudes
        merged = merged.with_columns(
            (pl.col('h2a_certified') / pl.col('bea_farm_emp_2011')).clip(0.0, 2.0).alias('h2a_rate')
        )

        # For continuous variables, fill missing with the column mean
        # For categorical variables, create new missing category for missing
        merged = merged.with_columns(
            pl.col(c).fill_null(pl.col(c).mean()) for c in cont_cols
        ).with_columns(
            pl.col(c).fill_null('MISSING') for c in cat_cols
        )

        # Generate IDs for each county-year
        merged = merged.with_columns([
            pl.struct(["county_ansi", "year"]).rank("dense").alias("group_id") - 1
        ])

        # Calculate acreage fraction AND Patch Exposure (frac of total BEA farm emp allocated that patch)
        merged = merged.with_columns(
            (pl.col('total_acres') / pl.col('total_acres').sum().over('group_id')).alias('acreage_frac')
        ).with_columns(
            (pl.col('acreage_frac') * pl.col('bea_farm_emp_2011')).alias('patch_exposure')
        )

        # Extract arrays
        patch_exposure = jnp.array(merged["patch_exposure"].to_numpy(), dtype=jnp.float32)
        group_ids = jnp.array(merged["group_id"].to_numpy(), dtype=jnp.int32)
        num_groups = merged["group_id"].max() + 1

        # Extract Unique target COUNTS per County-Year (Not Rates)
        unique_targets = merged.group_by("group_id").agg(pl.first("h2a_certified")).sort("group_id")
        y_count_county_year = jnp.array(unique_targets["h2a_certified"].to_numpy(), dtype=jnp.float32)

        # Standardize continuous variables (27 total)
        X_cont = merged.select(cont_cols).to_numpy()
        X_cont = (X_cont - X_cont.mean(axis=0)) / (X_cont.std(axis=0) + 1e-8)
        X_cont = jnp.array(X_cont, dtype=jnp.float32)

        # Build categorical interaction matrices
        feature_ids = {}
        feature_sizes = {}
        feature_names = {}

        for order in range(1, max_order + 1):
            cols = list(itertools.combinations(cat_cols, order))
            order_id_matrix =[]
            order_names =[]
            current_offset = 0
            for combo in cols:
                combined_str = merged.select(pl.concat_str(pl.col(combo), separator="|")).to_series()
                uniques, integer_ids = np.unique(combined_str, return_inverse=True)

                # Create readable labels
                combo_label = "|".join(combo)
                readable_names = [f"{combo_label}: {val}" for val in uniques]
                order_names.extend(readable_names)

                order_id_matrix.append(integer_ids + current_offset)
                current_offset += len(uniques)

            feature_ids[order] = jnp.array(np.column_stack(order_id_matrix))
            feature_sizes[order] = current_offset
            feature_names[order] = order_names

        return feature_ids, feature_sizes, feature_names, X_cont, patch_exposure, group_ids, y_count_county_year, num_groups, merged

    return (prep_jax_composition_arrays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPML objective function
    """)
    return


@app.cell
def _(feature_sizes_tup, jax, jnp):
    def initialize_params(feature_sizes: dict, num_continuous: int) -> dict:
        """
        Strategy C: 1 Intercept + 27 Slopes per interaction combo.
        Give categorical IDs 1 slope + N-categorical slopes from interacting with each continuous variables.
        This sets random initial slope guess.
        """
        # Nest the parameters so the root keys are both strings ('bias' and 'weights')
        params = {
            'bias': jnp.array(0.0, dtype=jnp.float32),
            'weights': {}
        }

        key = jax.random.PRNGKey(42)
        for order, size in feature_sizes_tup:
            key, subkey = jax.random.split(key)
            shape = (size, 1 + num_continuous)
            # Store the integer-keyed interaction weights inside the 'weights' dictionary
            params['weights'][order] = jax.random.normal(subkey, shape, dtype=jnp.float32) * 0.01

        return params

    return (initialize_params,)


@app.cell
def _(jnp):
    def compute_patch_log_worker(params: dict, feature_ids: dict, X_cont: jnp.ndarray) -> jnp.ndarray:
        """
        Computes log(worker) for EACH specific soil group in a county.
        """
        log_mu = params['bias']

        for order in feature_ids.keys():
            gathered_weights = params['weights'][order][feature_ids[order]]
            intercepts = gathered_weights[:, :, 0]       
            slopes = gathered_weights[:, :, 1:]          

            log_mu += jnp.sum(intercepts, axis=1)
            interaction_effect = jnp.sum(slopes * X_cont[:, None, :], axis=(1, 2))
            log_mu += interaction_effect

        return jnp.clip(log_mu, a_max=15.0) # Clipped slightly lower for per-acre rates to prevent overflow

    return (compute_patch_log_worker,)


@app.cell
def _(compute_patch_log_worker, jax, jnp):
    def ppml_objective_compositional(params: dict, feature_ids: dict, X_cont: jnp.ndarray, 
                                     patch_exposure: jnp.ndarray, group_ids: jnp.ndarray, 
                                     y_count_county_year: jnp.ndarray, num_groups: int,
                                     l1_rates: dict, l2_rates: dict) -> float:
        """
        The PPML objective function.
        It predicts the workers for every patch, sums them to the county level, 
        and THEN calculates the Poisson Error.
        """
        # 1. Get the log(rate) for all patches
        log_worker_per_patch = compute_patch_log_worker(params, feature_ids, X_cont)

        # 2. Patch Exposure * exp(log_worker_per_patch)) = Predicted workers for that patch
        workers_in_patch = patch_exposure * jnp.exp(log_worker_per_patch)

        # 3. Sum up the workers to the county level
        mu_count_cy = jax.ops.segment_sum(
            data=workers_in_patch, 
            segment_ids=group_ids, 
            num_segments=num_groups
        )

        # 4. POISSON NLL targeting the COUNTS
        nll = jnp.mean(mu_count_cy - y_count_county_year * jnp.log(mu_count_cy + 1e-7))

        # 5. Elastic Net Penalty
        # We impose massive penalties on interactions terms if parent terms are dropped, to impose heredity
        total_penalty = 0.0
        tau = 0.01 
        for k in feature_ids.keys():
            w_matrix = params['weights'][k]
            sp = tau * (jax.nn.softplus(w_matrix/tau) + jax.nn.softplus(-w_matrix/tau) - 2.0*jnp.log(2.0))

            if k == 1:
                multiplier = 1.0 
            else:
                parent_magnitude = jax.lax.stop_gradient(jnp.mean(params['weights'][k-1]**2))
                multiplier = jnp.maximum(1.0, 1e-3 / (parent_magnitude + 1e-8))

            total_penalty += l1_rates[k] * jnp.sum(multiplier * sp) + l2_rates[k] * jnp.sum(w_matrix**2)

        return nll + total_penalty

    return (ppml_objective_compositional,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # JAX custom autodiff
    """)
    return


@app.cell
def _(
    initialize_params,
    jax,
    jnp,
    lx,
    optax,
    ppml_objective_compositional,
    ravel_pytree,
):
    # This is the inner loop that is actually solving the PPML optimization problem
    def train_model_inner(feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates, max_iter=1000, tol=1e-5):
        num_continuous = X_cont.shape[1]
        params = initialize_params(feature_sizes_tup, num_continuous)

        def loss_fn(p):
            return ppml_objective_compositional(p, feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, l1_rates, l2_rates)

        solver = optax.lbfgs()
        opt_state = solver.init(params)
        val_and_grad_fn = optax.value_and_grad_from_state(loss_fn)

        def step(p, state):
            value, grad = val_and_grad_fn(p, state=state)
            updates, new_state = solver.update(grad, state, p, value=value, grad=grad, value_fn=loss_fn)
            return optax.apply_updates(p, updates), new_state, value

        jax.debug.print("  ->[Inner L-BFGS] Starting optimization (max_iter={max_iter}, tol={tol})...", max_iter=max_iter, tol=tol)

        # JAX native while loop (Checks convergence dynamically)
        def cond_fn(carry):
            i, p, state, prev_val, curr_val = carry

            # Calculate relative change: |curr - prev| / |prev|
            rel_change = jnp.abs(curr_val - prev_val) / (jnp.abs(prev_val) + 1e-8)

            # Stop if we hit tolerance (but force at least 5 iterations to get started)
            not_converged = jnp.logical_or(i < 5, rel_change > tol)
            not_max_iter = i < max_iter

            return jnp.logical_and(not_converged, not_max_iter)

        def body_fn(carry):
            i, p, state, prev_val, curr_val = carry
            p_next, state_next, val_next = step(p, state)

            def print_progress():
                jax.debug.print("      Iter {iter}: Loss = {loss}", iter=i+1, loss=val_next)

            jax.lax.cond(
                jnp.logical_or(i == 0, (i + 1) % 50 == 0),
                print_progress,
                lambda: None
            )

            return (i + 1, p_next, state_next, curr_val, val_next)

        # Initial carry state: (iteration_index, params, opt_state, prev_loss, curr_loss)
        # We fake the initial losses so the relative change triggers the loop to start
        init_carry = (0, params, opt_state, jnp.array(1e9, dtype=jnp.float32), jnp.array(1e8, dtype=jnp.float32))

        final_i, final_params, final_state, _, final_value = jax.lax.while_loop(cond_fn, body_fn, init_carry)

        jax.debug.print("  ->[Inner L-BFGS] Completed at Iter {i}. Final Loss = {loss}", i=final_i, loss=final_value)
        return final_params

    @jax.custom_vjp
    def train_model_cpu(feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates):
        return train_model_inner(feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates)

    def train_fwd(feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates):
        opt_params = train_model_inner(feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates)
        residuals = (opt_params, feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates)
        return opt_params, residuals

    def train_bwd(residuals, cotangents):
        opt_params, feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates = residuals
        v = cotangents 

        flat_opt_params, unravel_fn = ravel_pytree(opt_params)
        flat_v, _ = ravel_pytree(v)

        safe_l2_rates = {k: jnp.maximum(val, 1e-4) for k, val in l2_rates.items()}

        def flat_total_loss(p_flat):
            return ppml_objective_compositional(
                unravel_fn(p_flat), feature_ids, X_cont, patch_exposure, group_ids, 
                y_county_year, num_groups, l1_rates, safe_l2_rates
            )

        def damped_grad(p_flat, _):
            g = jax.grad(flat_total_loss)(p_flat)
            return g + 1.0 * p_flat

        hessian_op = lx.JacobianLinearOperator(damped_grad, flat_opt_params)

        jax.debug.print("  ->[Implicit VJP] Solving Hessian linear system for backward pass...")

        solver = lx.GMRES(rtol=1e-1, atol=1e-1, restart=50, max_steps=50)
        solution = lx.linear_solve(hessian_op, -flat_v, solver=solver)

        w_pytree = unravel_fn(solution.value)
        jax.debug.print("  -> [Implicit VJP] Linear solve completed.")

        def mixed_grad(l1_d, l2_d):
            safe_l2_d = {k: jnp.maximum(val, 1e-4) for k, val in l2_d.items()}
            return jax.grad(lambda p: ppml_objective_compositional(
                p, feature_ids, X_cont, patch_exposure, group_ids, y_county_year, num_groups, l1_d, safe_l2_d))(opt_params)

        _, vjp_fn = jax.vjp(mixed_grad, l1_rates, l2_rates)
        grad_l1, grad_l2 = vjp_fn(w_pytree)

        return (None, None, None, None, None, None, None, grad_l1, grad_l2)

    train_model_cpu.defvjp(train_fwd, train_bwd)
    return train_model_cpu, train_model_inner


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ALO-CV using lineax to take advantage of JAX autodiff
    """)
    return


@app.cell
def _(
    compute_patch_log_rate,
    jax,
    jnp,
    lx,
    patch_exposure,
    ppml_objective_compositional,
    ravel_pytree,
):
    def compute_alo_compositional(params, feature_ids, X_cont, acreage, group_ids, 
                                  y_county_year, num_groups, l1_rates, l2_rates):

        flat_p, unravel_fn = ravel_pytree(params)

        def flat_total_loss(p_flat):
            return ppml_objective_compositional(
                unravel_fn(p_flat), feature_ids, X_cont, acreage, group_ids, 
                y_county_year, num_groups, l1_rates, l2_rates
            )

        def all_groups_nll(p_flat):
            p = unravel_fn(p_flat)
            log_rate = compute_patch_log_rate(p, feature_ids, X_cont)
            workers = patch_exposure * jnp.exp(log_rate)
            mu_cy = jax.ops.segment_sum(workers, group_ids, num_groups)
            return mu_cy - y_county_year * jnp.log(mu_cy + 1e-7)

        # ---------------------------------------------------------
        # ISOLATE LEVERAGES AND STOP GRADIENTS (Prevents 3rd-order autodiff hang)
        # ---------------------------------------------------------
        def compute_leverages_no_grad(p_flat):
            def damped_grad(p, _):
                # 1.0 Damping to enforce strong conditioning
                return jax.grad(flat_total_loss)(p) + 1.0 * p

            hessian_op = lx.JacobianLinearOperator(damped_grad, p_flat)

            # MOVED max_steps HERE: Caps the XLA loop to guarantee it never hangs
            solver = lx.GMRES(rtol=1e-1, atol=1e-1, restart=50, max_steps=50)

            # Reduced from 10 to 2: We only need a rough approximation for hyperparameter tuning
            K = 2
            key = jax.random.PRNGKey(42)
            Z = jax.random.rademacher(key, (K, num_groups), dtype=p_flat.dtype)

            def compute_trace_sample(i):
                z = Z[i]
                jax.debug.print("    [ALO-CV] Solving sample {i}/{K}...", i=i+1, K=K)

                def z_weighted_nll(p):
                    return jnp.sum(z * all_groups_nll(p))

                v = jax.grad(z_weighted_nll)(p_flat)
                w = lx.linear_solve(hessian_op, v, solver=solver).value

                jax.debug.print("    [ALO-CV] Sample {i} solved.", i=i+1)
                return jnp.dot(v, w)

            trace_samples = jax.lax.map(compute_trace_sample, jnp.arange(K))
            return jnp.mean(trace_samples)

        # jax.lax.stop_gradient guarantees JAX will not try to take the derivative 
        # of the Hessian-inverse during the Meta-Adam update.
        sum_leverages = jax.lax.stop_gradient(compute_leverages_no_grad(flat_p))

        # 4. Mean ALO Score calculation
        # Gradients will flow normally through the NLL term
        group_nlls = all_groups_nll(flat_p)
        alo_score = jnp.mean(group_nlls) + (sum_leverages / (num_groups ** 2))

        jax.debug.print("  -> [ALO-CV] ALO Score calculated successfully: {alo}", alo=alo_score)

        return alo_score

    return (compute_alo_compositional,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bilevel meta-optimization to select optimal hyperparameters that minimize ALO-CV score
    """)
    return


@app.cell
def _(jnp):
    def jax_inv_softplus(x):
        """
        Stable inverse softplus.
        """
        return jnp.log(jnp.expm1(x) + 1e-8)

    return (jax_inv_softplus,)


@app.cell
def _(compute_alo_compositional, jax, train_model_cpu):
    def meta_loss_fn(raw_l1, raw_l2, feature_ids, X_cont, acreage, group_ids, y_county_year, num_groups, feature_sizes_tup):
        """
        Compute ALO-CV score given L1 and L2 params
        """
        # Enforce strict positivity constraints
        l1_rates = {k: jax.nn.softplus(v) for k, v in raw_l1.items()}
        l2_rates = {k: jax.nn.softplus(v) + 1e-4 for k, v in raw_l2.items()}

        # 1. Train Inner Model
        opt_params = train_model_cpu(feature_ids, X_cont, acreage, group_ids, y_county_year, num_groups, feature_sizes_tup, l1_rates, l2_rates)

        # 2. Evaluate with ALO-CV
        return compute_alo_compositional(opt_params, feature_ids, X_cont, acreage, group_ids, y_county_year, num_groups, l1_rates, l2_rates)

    # JIT the meta-gradient calculator
    meta_val_and_grad = jax.jit(jax.value_and_grad(meta_loss_fn, argnums=(0, 1)), static_argnames=['num_groups', 'feature_sizes_tup'])
    return (meta_val_and_grad,)


@app.cell
def _(
    code_path,
    jax,
    jax_inv_softplus,
    jnp,
    meta_val_and_grad,
    mo,
    np,
    optax,
    pickle,
):
    @mo.persistent_cache
    def run_meta_optimization(feature_ids, X_cont, acreage, group_ids, y_county_year, num_groups, feature_sizes_tup, meta_iters=150, patience=15, reset=False):

        # Define file paths for checkpointing and logging
        checkpoint_path = code_path / 'json' / "meta_ppml_opt_checkpoint.pkl"
        log_path = code_path / 'json' / "meta_ppml_opt_log.csv"

        meta_optimizer = optax.adam(learning_rate=0.01)

        # Allow forced reset if you want to start over
        if reset and checkpoint_path.exists():
            checkpoint_path.unlink()
            print("[-] Deleted old checkpoint. Starting fresh.")

        # -------------------------------------------------------------
        # 1. LOAD CHECKPOINT OR INITIALIZE FROM SCRATCH
        # -------------------------------------------------------------
        if checkpoint_path.exists():
            print(f"\n--- Resuming Bilevel Hyperparameter Tuning from {checkpoint_path.name} ---")
            with open(checkpoint_path, "rb") as f:
                chkpt = pickle.load(f)

            raw_l1, raw_l2 = chkpt['hparams']
            hparams = (raw_l1, raw_l2)
            state = chkpt['state']
            best_alo = chkpt['best_alo']
            best_hparams = chkpt['best_hparams']
            wait = chkpt['wait']
            start_iter = chkpt['iter'] + 1

            print(f"Resuming at Iteration {start_iter} (Best ALO so far: {best_alo:.4f} | Patience: {wait}/{patience})")

        else:
            print("\n--- Starting Bilevel Hyperparameter Tuning from Scratch ---")
            raw_l1 = {k: jnp.array(jax_inv_softplus(0.01 * (2.0**(k-1))), dtype=jnp.float32) for k in range(1, 4)}
            raw_l2 = {k: jnp.array(jax_inv_softplus(0.005 * (2.0**(k-1))), dtype=jnp.float32) for k in range(1, 4)}
            hparams = (raw_l1, raw_l2)
            state = meta_optimizer.init(hparams)

            best_alo = float('inf')
            best_hparams = hparams
            wait = 0
            start_iter = 0

            # Initialize the CSV log file
            with open(log_path, "w") as f:
                f.write("Step,ALO_CV_Score,Mean_L1_Penalty,Wait\n")

        print(f"\n{'Step':<6} | {'ALO-CV Score':<15} | {'Mean L1 Penalty':<15} | Patience Tracker")
        print("-" * 70)

        tol = 1e-3

        # -------------------------------------------------------------
        # 2. RUN META-OPTIMIZATION LOOP
        # -------------------------------------------------------------
        for i in range(start_iter, meta_iters):
            alo, (gl1, gl2) = meta_val_and_grad(raw_l1, raw_l2, feature_ids, X_cont, acreage, group_ids, y_county_year, num_groups, feature_sizes_tup)

            current_alo = alo.item()

            # Early Stopping Logic
            if current_alo < best_alo - tol:
                best_alo = current_alo
                best_hparams = hparams
                wait = 0
            else:
                wait += 1

            act_l1 = np.mean([jax.nn.softplus(v).item() for v in raw_l1.values()])

            # Print to console
            print(f"{i:<6} | {current_alo:<15.4f} | {act_l1:<15.6f} | {wait}/{patience}")

            # Append to CSV log
            with open(log_path, "a") as f:
                f.write(f"{i},{current_alo},{act_l1},{wait}\n")

            # -------------------------------------------------------------
            # 3. ATOMIC CHECKPOINT SAVE
            # -------------------------------------------------------------
            # We save to a temporary file and replace the old one. 
            # This prevents corruption if you cancel the run exactly mid-save.
            tmp_path = checkpoint_path.with_suffix('.pkl.tmp')
            with open(tmp_path, "wb") as f:
                pickle.dump({
                    'iter': i,
                    'hparams': hparams,
                    'state': state,
                    'best_alo': best_alo,
                    'best_hparams': best_hparams,
                    'wait': wait
                }, f)
            tmp_path.replace(checkpoint_path)

            # Break out if patience threshold is met
            if wait >= patience:
                print(f"\n[!] Convergence Reached: ALO-CV Score has not improved by {tol} in {patience} steps.")
                print(f"[!] Stopping early at Meta-Iteration {i}.")
                hparams = best_hparams
                break

            # Apply gradients
            updates, state = meta_optimizer.update((gl1, gl2), state, hparams)
            hparams = optax.apply_updates(hparams, updates)
            raw_l1, raw_l2 = hparams

        # Extract the absolute best hyperparameters found
        best_raw_l1, best_raw_l2 = best_hparams
        final_l1 = {k: jax.nn.softplus(v) for k, v in best_raw_l1.items()}
        final_l2 = {k: jax.nn.softplus(v) + 1e-4 for k, v in best_raw_l2.items()}

        return final_l1, final_l2

    return (run_meta_optimization,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Perform bilevel optimization to find optimal L1 and L2 rates
    """)
    return


@app.cell
def _(
    bea,
    cat_cols,
    climate,
    cont_cols,
    h2a,
    np,
    prep_jax_composition_arrays,
    soil,
):
    # Prep input data
    feature_ids, feature_sizes, feature_names, X_cont, patch_exposure, group_ids, y_target_rate, num_groups, merged_df = prep_jax_composition_arrays(h2a, bea, soil, climate, cont_cols, cat_cols, max_order = 3)
    feature_sizes_tup = tuple(sorted(feature_sizes.items()))

    # Check for any remaining NaNs or Infs
    assert not np.isnan(y_target_rate).any(), "NaNs detected in target rate!"
    assert not np.isinf(y_target_rate).any(), "Infs detected in target rate!"
    print(f"Target Max Rate: {np.max(y_target_rate):.4f}")
    print(f"Target Min Rate: {np.min(y_target_rate):.4f}")
    return (
        X_cont,
        feature_ids,
        feature_sizes,
        feature_sizes_tup,
        group_ids,
        merged_df,
        num_groups,
        patch_exposure,
        y_target_rate,
    )


@app.cell
def _(
    X_cont,
    feature_ids,
    feature_sizes_tup,
    group_ids,
    num_groups,
    patch_exposure,
    run_meta_optimization,
    y_target_rate,
):
    # Note that inner loop is set at max_iter = 50 as default value
    best_l1, best_l2 = run_meta_optimization(feature_ids, X_cont, patch_exposure, group_ids, y_target_rate, num_groups, feature_sizes_tup, meta_iters=150, patience=15, reset=False)
    return best_l1, best_l2


@app.cell
def _(
    X_cont,
    best_l1,
    best_l2,
    feature_ids,
    feature_sizes,
    group_ids,
    num_groups,
    patch_exposure,
    train_model_inner,
    y_target_rate,
):
    # Final training run uses the same default parameters as the ones used for meta-Adam (max_iter=1000, tol=1e-5)
    trained_params = train_model_inner(feature_ids, X_cont, patch_exposure, group_ids, y_target_rate, num_groups, feature_sizes, best_l1, best_l2)

    print("\n--- Model Estimation with Optimized Penalty Parameters ---")

    print("\nOptimization Complete!")
    print(f"Final Global Base Rate (Bias): {trained_params['bias']:.4f}")
    return (trained_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Obtain predictions
    """)
    return


@app.cell
def _(jnp, trained_params):
    print("\nSparsity Report (Threshold 1e-3):")
    for order in range(1, 4):
        # Look inside the 'weights' dictionary
        w = trained_params['weights'][order] 
        active_weights = jnp.sum(jnp.abs(w) > 1e-3)
        total_weights = w.size
        print(f"  Order {order} total weights active: {active_weights} / {total_weights}")
    return


@app.cell
def _(compute_patch_log_worker, jax, jnp, mo):
    @mo.persistent_cache
    def predicted_h2a_county_year(params, feature_ids, X_cont, patch_exposure, group_ids, num_groups):
    # 1. Get the log(rate) for all patches
        log_worker_per_patch = compute_patch_log_worker(params, feature_ids, X_cont)

        # 2. Patch Exposure * exp(log_worker_per_patch)) = Predicted workers for that patch
        workers_in_patch = patch_exposure * jnp.exp(log_worker_per_patch)

        # 3. Sum up the workers to the county level
        mu_count_cy = jax.ops.segment_sum(
            data=workers_in_patch, 
            segment_ids=group_ids, 
            num_segments=num_groups
        )

        return mu_count_cy

    return (predicted_h2a_county_year,)


@app.cell
def _(
    X_cont,
    feature_ids,
    group_ids,
    merged_df,
    np,
    num_groups,
    patch_exposure,
    pl,
    predicted_h2a_county_year,
    trained_params,
    y_pred,
    y_target_counts,
    y_true,
):
    print("\n--- Generating Predictions ---")
    # Get count predictions (returns a JAX array)
    y_pred_counts_jax = predicted_h2a_county_year(trained_params, feature_ids, X_cont, patch_exposure, group_ids, num_groups)

    # Retrieve the county employment denominator to convert counts to rates
    unique_emp = merged_df.group_by("group_id").agg(pl.first("bea_farm_emp_2011")).sort("group_id")
    county_emp_arr = unique_emp["bea_farm_emp_2011"].to_numpy()

    # Convert JAX arrays to NumPy and calculate the rates
    y_pred_counts = np.array(y_pred_counts_jax)
    y_pred_rates = y_pred_counts / county_emp_arr

    y_true_counts = np.array(y_target_counts)
    y_true_rates = y_true_counts / county_emp_arr

    # Create a Polars DataFrame to compare True vs Predicted
    results_df = pl.DataFrame({
        "group_id": np.arange(num_groups),
        "true_h2a_share": y_true_rates,
        "predicted_h2a_share": y_pred_rates,
        "true_h2a_count": y_true_counts,
        "predicted_h2a_count": y_pred_counts
    })

    # Calculate some basic Econometric metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    print(f"Mean Absolute Error (MAE): {mae:.2f} rate")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} rate")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
