import marimo

__generated_with = "0.23.4"
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
    import json
    import time
    from functools import partial

    return (
        cs,
        itertools,
        jax,
        jnp,
        json,
        lx,
        mo,
        np,
        optax,
        partial,
        pl,
        pyprojroot,
        ravel_pytree,
        time,
    )


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion="pyproject.toml")
    binary_path = root_path / "binaries"
    code_path = root_path / "code"
    return binary_path, code_path


@app.cell
def _(jax):
    # Check CUDA working
    print(jax.devices())
    print(jax.default_backend())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load and preprocess data for JAX
    """)
    return


@app.cell
def _(binary_path, pl):
    # BEA farm employment in 2011 as baseline for calculating shares
    bea = pl.read_parquet(binary_path / "bea_farm_nonfarm_emp.parquet")
    bea = (
        bea.filter(pl.col("year") == 2011)
        .rename({"bea_farm_emp": "bea_farm_emp_2011", "county_fips": "county_ansi"})
        .drop(["bea_nonfarm_emp", "year"])
    )
    return (bea,)


@app.cell
def _(binary_path, pl):
    h2a = (
        pl.read_parquet(binary_path / "h2a_aggregated.parquet")
        .with_columns(
            (pl.col("state_fips_code") + pl.col("county_fips_code")).alias("county_ansi"),
            pl.col("year").cast(pl.Int32).alias("year"),
        )
        .rename({"nbr_workers_certified_start_year": "h2a_certified"})
        .select(
            [
                "year",
                "county_ansi",
                "h2a_certified",
            ]
        )
        .filter(pl.col("year") >= 2008, pl.col("year") <= 2011)
    )
    return (h2a,)


@app.cell
def _(binary_path, cs, pl):
    # Grab climate variables we actually want to use
    climate = pl.read_parquet(
        binary_path / "county_h2a_prediction_climate_gdd_annual.parquet"
    )
    climate = (
        climate.select(
            [
                "year",
                "fips",
                cs.contains("days_D"),
                cs.contains("days_P"),
                cs.starts_with("GDD_"),
                "tmin_ann",
                "tmax_ann",
                "tavg_ann",
                "prcp_ann",
                "prcp_gs",
                "prcp_spring",
                "n_wet_days",
                "max_cdd_gs",
            ]
        )
        .filter(pl.col("year") >= 2008, pl.col("year") <= 2011)
        .rename({"fips": "county_ansi"})
    ).drop(
        ["GDD_sorghum", "GDD_barley"]
    )  # Duplicates: GDD sorghum = GDD corn, GDD barley = GDD spring wheat
    return (climate,)


@app.cell
def _(binary_path, pl):
    soil = pl.read_parquet(binary_path / "county_h2a_prediction_gnatsgo.parquet")
    return (soil,)


@app.cell
def _(climate):
    county_cont_cols = climate.columns.copy()
    county_cont_cols.remove("year")
    county_cont_cols.remove("county_ansi")

    patch_cont_cols = [
        "slope_r",
        "resdept_r",
        "cropprodindex",
        "aws0150wta",
    ]
    cat_cols = [
        "taxorder",
        "drainagecl",
        "nirrcapcl",
    ]
    return cat_cols, county_cont_cols, patch_cont_cols


@app.cell
def _(itertools, jnp, np, pl):
    def prep_jax_composition_arrays(
        h2a,
        bea,
        soil,
        climate,
        county_cont_cols,
        patch_cont_cols,
        cat_cols,
        max_order,
    ):
        """
        Merge, standardize continuous variables, map categorical variables
        """
        merged = soil.join(climate, on="county_ansi", how="inner")
        merged = merged.join(h2a, on=["county_ansi", "year"], how="left").with_columns(
            pl.col("h2a_certified").fill_null(0).alias("h2a_certified")
        )
        merged = merged.join(bea, on="county_ansi", how="inner")

        # Drop missing or zero employment counties entirely)
        merged = merged.filter(
            pl.col("bea_farm_emp_2011").is_not_null() & (pl.col("bea_farm_emp_2011") > 0)
        )

        # Optional: cap target rate at a reasonable ceiling like 2.0 to prevent
        # extreme BEA data artifacts from destroying gradient magnitudes
        merged = merged.with_columns(
            (pl.col("h2a_certified") / pl.col("bea_farm_emp_2011"))
            .clip(0.0, 2.0)
            .alias("h2a_rate")
        ).with_columns(
            (pl.col("h2a_rate") * pl.col("bea_farm_emp_2011")).alias("h2a_target_count")
        )

        # For continuous variables, fill missing with the column mean
        # For categorical variables, create new missing category for missing
        merged = (
            merged.with_columns(
                pl.col(c).fill_null(pl.col(c).mean()) for c in county_cont_cols
            )
            .with_columns(pl.col(c).fill_null(pl.col(c).mean()) for c in patch_cont_cols)
            .with_columns(pl.col(c).fill_null("MISSING") for c in cat_cols)
        )

        # Generate IDs for each county-year
        merged = merged.with_columns(
            [pl.struct(["county_ansi", "year"]).rank("dense").alias("group_id") - 1]
        )

        # Calculate acreage fraction AND Patch Exposure (frac of total BEA farm emp allocated that patch)
        merged = merged.with_columns(
            (pl.col("total_acres") / pl.col("total_acres").sum().over("group_id")).alias(
                "acreage_frac"
            )
        ).with_columns(
            (pl.col("acreage_frac") * pl.col("bea_farm_emp_2011")).alias("patch_exposure")
        )

        # Extract arrays
        patch_exposure = jnp.array(merged["patch_exposure"].to_numpy(), dtype=jnp.float32)
        group_ids = jnp.array(merged["group_id"].to_numpy(), dtype=jnp.int32)
        num_groups = merged["group_id"].max() + 1

        # Table of just unique county-years (drops patch variation)
        county_design = (
            merged.select(["group_id"] + county_cont_cols).unique().sort("group_id")
        )
        assert county_design.height == num_groups  # Check we have every group

        # Extract Unique target COUNTS per County-Year (Not Rates)
        unique_targets = (
            merged.group_by("group_id").agg(pl.first("h2a_target_count")).sort("group_id")
        )
        y_count_county_year = jnp.array(
            unique_targets["h2a_target_count"].to_numpy(), dtype=jnp.float32
        )

        # Standardize continuous variables
        # County-specific continuous variables
        X_county_cont = county_design.select(county_cont_cols).to_numpy().astype(np.float32)
        X_county_cont = (X_county_cont - X_county_cont.mean(axis=0)) / (
            X_county_cont.std(axis=0) + 1e-8
        )
        X_county_cont = jnp.array(X_county_cont, dtype=jnp.float32)

        # Patch-specific continuous variables
        X_patch_cont = merged.select(patch_cont_cols).to_numpy().astype(np.float32)
        X_patch_cont = (X_patch_cont - X_patch_cont.mean(axis=0)) / (
            X_patch_cont.std(axis=0) + 1e-8
        )
        X_patch_cont = jnp.array(X_patch_cont, dtype=jnp.float32)

        # Build categorical interaction matrices
        feature_ids = {}
        feature_sizes = {}
        feature_names = {}

        for order in range(1, max_order + 1):
            cols = list(itertools.combinations(cat_cols, order))
            order_id_matrix = []
            order_names = []
            current_offset = 0
            for combo in cols:
                combined_str = merged.select(
                    pl.concat_str(pl.col(combo), separator="|")
                ).to_series()
                uniques, integer_ids = np.unique(combined_str, return_inverse=True)

                # Create readable labels
                combo_label = "|".join(combo)
                readable_names = [f"{combo_label}: {val}" for val in uniques]
                order_names.extend(readable_names)

                order_id_matrix.append((integer_ids + current_offset).astype(np.int32))
                current_offset += len(uniques)

            feature_ids[order] = jnp.array(
                np.column_stack(order_id_matrix), dtype=jnp.int32
            )
            feature_sizes[order] = current_offset
            feature_names[order] = order_names

        return (
            feature_ids,
            feature_sizes,
            feature_names,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_count_county_year,
            num_groups,
            merged,
        )

    return (prep_jax_composition_arrays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPML objective function
    """)
    return


@app.cell
def _(jax, jnp):
    def initialize_params(
        feature_sizes,
        num_continuous: int,
    ) -> dict:
        """
        Strategy C: 1 Intercept + 27 Slopes per interaction combo.
        Give categorical IDs 1 slope + N-categorical slopes from interacting with each continuous variables.
        This sets random initial slope guess.
        """
        # Nest the parameters so the root keys are both strings ('bias' and 'weights')
        params = {"bias": jnp.array(0.0, dtype=jnp.float32), "weights": {}}

        key = jax.random.PRNGKey(42)
        feature_size_items = (
            feature_sizes.items() if hasattr(feature_sizes, "items") else feature_sizes
        )
        for order, size in sorted(feature_size_items):
            key, subkey = jax.random.split(key)
            shape = (size, 1 + num_continuous)
            # Store the integer-keyed interaction weights inside the 'weights' dictionary
            params["weights"][order] = (
                jax.random.normal(subkey, shape, dtype=jnp.float32) * 0.01
            )

        return params

    return (initialize_params,)


@app.cell
def _(jnp):
    def compute_patch_log_worker(
        params: dict,
        feature_ids: dict,
        X_county_cont: jnp.ndarray,
        X_patch_cont: jnp.ndarray,
        group_ids,
    ) -> jnp.ndarray:
        """
        Computes log(worker) for EACH specific soil group in a county.
        """
        num_county = X_county_cont.shape[1]
        log_mu = params["bias"]

        for order in feature_ids.keys():
            gathered_weights = params["weights"][order][feature_ids[order]]
            intercepts = gathered_weights[:, :, 0]
            slopes = gathered_weights[:, :, 1:]

            county_slopes = slopes[:, :, :num_county]
            patch_slopes = slopes[:, :, num_county:]

            log_mu += jnp.sum(intercepts, axis=1)

            # 'nkc' is slopes, 'nc' is X_cont. We multiply them and sum over 'k' and 'c' for each 'n'
            # county effects
            log_mu += jnp.einsum("nkc,nc->n", county_slopes, X_county_cont[group_ids])
            # patch effects
            log_mu += jnp.einsum("nkc,nc->n", patch_slopes, X_patch_cont)

        return jnp.clip(log_mu, max=15.0)  # Clipped per-acre rates to prevent overflow

    return (compute_patch_log_worker,)


@app.cell
def _(compute_patch_log_worker, jax, jnp):
    def ppml_objective_compositional(
        params: dict,
        feature_ids: dict,
        X_county_cont: jnp.ndarray,
        X_patch_cont: jnp.ndarray,
        patch_exposure: jnp.ndarray,
        group_ids: jnp.ndarray,
        y_count_county_year: jnp.ndarray,
        num_groups: int,
        l1_rates: dict,
        l2_rates: dict,
    ) -> float:
        """
        The FULL objective function.
        Used for logging only! Gradients are NOT taken through this function anymore.
        It predicts the workers for every patch, sums them to the county level,
        and THEN calculates the Poisson Error.
        """
        # 1. Get the log(rate) for all patches (identical to original)
        log_worker_per_patch = compute_patch_log_worker(
            params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            group_ids,
        )

        # 2. Patch Exposure * exp(log_worker_per_patch)) = Predicted workers for that patch (identical)
        workers_in_patch = patch_exposure * jnp.exp(log_worker_per_patch)

        # 3. Sum up the workers to the county level (identical)
        mu_count_cy = jax.ops.segment_sum(workers_in_patch, group_ids, num_groups)

        # 4. POISSON NLL targeting the COUNTS (identical)
        nll = jnp.mean(mu_count_cy - y_count_county_year * jnp.log(mu_count_cy + 1e-7))

        total_penalty = 0.0
        for k in feature_ids.keys():
            w_matrix = params["weights"][k]

            # Exact L1 Heredity Multiplier
            if k == 1:
                multiplier = 1.0
            else:
                parent_magnitude = jax.lax.stop_gradient(
                    jnp.mean(params["weights"][k - 1] ** 2)
                )
                multiplier = jnp.maximum(1.0, 1e-3 / (parent_magnitude + 1e-8))

            # EXACT L1 (jnp.abs) instead of Softplus approximation, so not differentiable
            l1_penalty = l1_rates[k] * jnp.sum(multiplier * jnp.abs(w_matrix))
            l2_penalty = l2_rates[k] * jnp.sum(w_matrix**2)

            total_penalty += l1_penalty + l2_penalty

        return nll + total_penalty

    return (ppml_objective_compositional,)


@app.cell
def _(compute_patch_log_worker, jax, jnp):
    def ppml_objective_compositional_smooth_only(
        params: dict,
        feature_ids: dict,
        X_county_cont: jnp.ndarray,
        X_patch_cont: jnp.ndarray,
        patch_exposure: jnp.ndarray,
        group_ids: jnp.ndarray,
        y_count_county_year: jnp.ndarray,
        num_groups: int,
        l2_rates: dict,
    ) -> float:
        """
        The smooth part of the PPML objective function (Poisson NLL + L2 Ridge).
        Used exclusively by FISTA's jax.grad and the VJP's exact_hvp.
        """
        log_worker_per_patch = compute_patch_log_worker(
            params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            group_ids,
        )
        workers_in_patch = patch_exposure * jnp.exp(log_worker_per_patch)
        mu_count_cy = jax.ops.segment_sum(workers_in_patch, group_ids, num_groups)
        nll = jnp.mean(mu_count_cy - y_count_county_year * jnp.log(mu_count_cy + 1e-7))

        # L2 ridge penalty only
        total_l2_penalty = 0.0
        for k in feature_ids.keys():
            w_matrix = params["weights"][k]
            total_l2_penalty += l2_rates[k] * jnp.sum(w_matrix**2)

        return nll + total_l2_penalty

    return (ppml_objective_compositional_smooth_only,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # JAX custom autodiff
    """)
    return


@app.cell
def _(jax, jnp):
    # This replaces softplus function used to make L1 penalty differentiable
    # Adds L1 penalty parameters to objective function using soft-thresholding and imposes heredity on higher order interaction terms
    def prox_g(
        param_pytree,
        step_size,
        l1_rates,
    ):
        """
        Applies soft-thresholding to weights, considering heredity and zero-padding for bias.
        Bias is not regularized.
        """
        updated_params = {
            "bias": param_pytree["bias"],  # Bias is not regularized
            "weights": {},  # Initialize empty weights dictionary
        }

        # Get flattened current parameters to calculate parent magnitudes for heredity (stop_gradient)
        # We need the *current* state of the parameters (p, not y_mom) for heredity.
        # This implies that the heredity multipliers are based on the parameters from the *previous* FISTA iteration's `p`.
        # For simplicity in this structure, we'll assume `param_pytree` itself is the reference for heredity.
        # A more rigorous implementation might pass the previous `p` as a separate argument.

        for order in sorted(param_pytree["weights"]):
            w_matrix = param_pytree["weights"][order]
            l1_rate_k = l1_rates[order]

            if order == 1:
                # No parent for order 1, multiplier is 1.0
                multiplier = 1.0
            else:
                # Heredity: use stop_gradient on parent magnitude from the *reference* params
                # This ensures we don't take gradients through the thresholding logic for heredity
                parent_magnitude = jax.lax.stop_gradient(
                    jnp.mean(param_pytree["weights"][order - 1] ** 2)
                )
                multiplier = jnp.maximum(
                    1.0, 1e-3 / (parent_magnitude + 1e-8)
                )  # Ensure multiplier >= 1.0

            # Calculate dynamic threshold
            threshold = step_size * l1_rate_k * multiplier

            # Apply EXACT SOFT-THRESHOLDING (Vectorized in JAX)
            updated_weights = jnp.sign(w_matrix) * jnp.maximum(
                0.0, jnp.abs(w_matrix) - threshold
            )

            updated_params["weights"][order] = updated_weights

        return updated_params

    return (prox_g,)


@app.cell
def _(
    jax,
    jnp,
    ppml_objective_compositional,
    ppml_objective_compositional_smooth_only,
    prox_g,
    ravel_pytree,
):
    # This is the inner loop that is actually solving the PPML optimization problem
    def train_model_inner(
        init_params,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_county_year,
        num_groups,
        feature_sizes_tup,
        l1_rates,
        l2_rates,
        inner_iters,
        inner_tol,
    ):  # note smaller tol, we are using FISTA
        num_continuous = X_county_cont.shape[1] + X_patch_cont.shape[1]

        # FISTA initializations
        p = init_params  # Current parameters
        y_mom = init_params  # Momentum point (y_k)
        t = jnp.array(1.0, dtype=jnp.float32)  # Nesterov momentum scalar
        L = jnp.array(
            1.0, dtype=jnp.float32
        )  # Initial Lipschitz constant (can be adaptive or fixed)

        # Smooth loss function (for backtracking and gradient calculation)
        def smooth_loss_fn(p_val):
            return ppml_objective_compositional_smooth_only(
                p_val,
                feature_ids,
                X_county_cont,
                X_patch_cont,
                patch_exposure,
                group_ids,
                y_county_year,
                num_groups,
                l2_rates,
            )

        # JIT the value and gradient for efficiency
        val_and_grad_smooth_fn = jax.jit(jax.value_and_grad(smooth_loss_fn))

        # Implement FISTA while loop
        # Placeholder for the previous max_param_change to initiate the loop
        prev_max_param_change = jnp.array(jnp.inf, dtype=jnp.float32)

        def cond_fn_outer(carry):
            i, _, _, _, _, mpc, _, _ = carry
            # Loop continues if max_iter not reached AND not converged
            return jnp.logical_and(i < inner_iters, mpc > inner_tol)

        def body_fn_outer(carry):
            (i, p, y_mom, t, L, prev_max_param_change, best_p, best_loss) = carry

            # --- Backtracking Line Search (Nested While Loop) ---
            # The line search requires the smooth part of the objective only
            loss_y_val, grad_y_val = val_and_grad_smooth_fn(y_mom)

            def cond_fn_bt(carry_bt):
                (k_bt, _, loss_p_next_candidate, L_bt_current) = carry_bt
                # Majorization condition: f(p_next) <= f(y) + <grad f(y), p_next - y> + (L/2)*||p_next - y||^2

                # Compute p_next_candidate for the current L_bt_current
                grad_y_at_L = jax.tree_util.tree_map(
                    lambda ym, gy: ym - (1.0 / L_bt_current) * gy,
                    y_mom,
                    grad_y_val,
                )
                p_next_candidate_val = prox_g(
                    grad_y_at_L,
                    1.0 / L_bt_current,
                    l1_rates,
                )

                # Re-evaluate loss_p_next_candidate with the new p_next_candidate_val
                loss_p_next_candidate_re_eval = smooth_loss_fn(p_next_candidate_val)

                # Sum each leaf to a scalar first, then sum the scalars
                diff_sq_leaves = jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(
                        lambda pn, ym: jnp.sum((pn - ym) ** 2),
                        p_next_candidate_val,
                        y_mom,
                    )
                )
                diff_p_y_norm_sq = jnp.sum(jnp.array(diff_sq_leaves))

                grad_dot_diff_leaves = jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(
                        lambda gy, pn, ym: jnp.sum(gy * (pn - ym)),
                        grad_y_val,
                        p_next_candidate_val,
                        y_mom,
                    )
                )
                grad_dot_diff = jnp.sum(jnp.array(grad_dot_diff_leaves))

                lhs = loss_p_next_candidate_re_eval
                rhs = loss_y_val + grad_dot_diff + (0.5 * L_bt_current * diff_p_y_norm_sq)

                # Stop after 20 iterations to prevent infinite loop for ill-conditioned problems
                return jnp.logical_and(lhs > rhs, k_bt < 20)

            def body_fn_bt(carry_bt):
                (k_bt, _, _, L_bt_current) = carry_bt
                L_new_bt = L_bt_current * 1.5  # Grow L
                # Recalculate p_next_candidate and its loss in the condition, no need here.
                return k_bt + 1, None, None, L_new_bt  # Only L is updated inside the loop

            # Initialize backtracking with current L
            bt_carry_init = (0, None, None, L)  # (k_bt, dummy_p, dummy_loss, L_init)
            final_k_bt, _, _, L_final_bt = jax.lax.while_loop(
                cond_fn_bt, body_fn_bt, bt_carry_init
            )

            # Decay L for next outer iteration. L should always be positive.
            L_new = L_final_bt * 0.9
            # Ensure L does not drop too low or become zero.
            L_new = jnp.maximum(L_new, 1e-8)

            # Compute p_next using the final L from backtracking
            grad_y = jax.tree_util.tree_map(
                lambda ym, gy: ym - (1.0 / L_final_bt) * gy,
                y_mom,
                grad_y_val,
            )
            p_next = prox_g(
                grad_y,
                1.0 / L_final_bt,
                l1_rates,
            )

            # --- Adaptive Restart Check ---
            # Flatten for vdot comparison
            flat_p_prev, _ = ravel_pytree(p)
            flat_p_next, _ = ravel_pytree(p_next)
            flat_y_mom, _ = ravel_pytree(y_mom)

            # Check gradient-momentum alignment: <y_k - p_{k-1}, p_k - p_{k-1}> > 0
            restart_condition = (
                jnp.vdot(flat_y_mom - flat_p_prev, flat_p_next - flat_p_prev) > 0
            )

            # Update t based on restart
            t_next_raw = (1.0 + jnp.sqrt(1.0 + 4.0 * t**2)) / 2.0
            t_next = jnp.where(restart_condition, 1.0, t_next_raw)

            # Update y_mom (momentum point)
            # If restart, y_mom_next = p_next. Otherwise, standard FISTA momentum.
            y_mom_next = jax.tree_util.tree_map(
                lambda pn, pp: jnp.where(
                    restart_condition, pn, pn + ((t - 1.0) / t_next) * (pn - pp)
                ),
                p_next,
                p,
            )

            # Convergence check: max parameter change
            max_param_change = jnp.max(jnp.abs(flat_p_next - flat_p_prev))

            # JAX debug print for progress (streams to stdout asynchronously)
            def print_progress():
                # Use the full objective (NLL + L1 + L2) for logging
                full_loss = ppml_objective_compositional(
                    p_next,
                    feature_ids,
                    X_county_cont,
                    X_patch_cont,
                    patch_exposure,
                    group_ids,
                    y_county_year,
                    num_groups,
                    l1_rates,
                    l2_rates,
                )
                jax.debug.print(
                    "      Iter {iter}: Loss = {loss:.4f}, MaxParamChange = {mpc:.6f}, L = {L:.4f}, BT_iters={k_bt}",
                    iter=i + 1,
                    loss=full_loss,
                    mpc=max_param_change,
                    L=L_final_bt,
                    k_bt=final_k_bt,
                )

            jax.lax.cond(
                jnp.logical_or(i == 0, (i + 1) % 50 == 0), print_progress, lambda: None
            )

            return (
                i + 1,
                p_next,
                y_mom_next,
                t_next,
                L_new,
                max_param_change,
                best_p,
                best_loss,
            )  # Pass best_p/best_loss along, though not used in FISTA logic for now

        # Initial carry state for FISTA outer while_loop
        init_carry = (
            0,  # i (iteration index)
            p,  # p (current parameters)
            y_mom,  # y_mom (momentum point)
            t,  # t (Nesterov momentum scalar)
            L,  # L (Lipschitz constant)
            prev_max_param_change,  # prev_max_param_change (for convergence check)
            p,  # best_p (not directly used by FISTA, but useful to pass along)
            jnp.array(jnp.inf, dtype=jnp.float32),  # best_loss (not directly used)
        )
        # Run the while loop
        final_i, final_params, _, _, _, final_mpc, _, _ = jax.lax.while_loop(
            cond_fn_outer, body_fn_outer, init_carry
        )
        jax.debug.print(
            "  ->[Inner FISTA] Completed at Iter {i}. Final Max Param Change = {mpc:.6f}",
            i=final_i,
            mpc=final_mpc,
        )
        return final_params

    return (train_model_inner,)


@app.cell
def _(
    jax,
    jnp,
    lx,
    ppml_objective_compositional_smooth_only,
    ravel_pytree,
    train_model_inner,
):
    # These functions are forwards and backwards mode solvers that we join to our inner loop solver
    # This is the inner loop solver
    @jax.custom_vjp
    def train_model_cpu(
        init_params,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_county_year,
        num_groups,
        feature_sizes_tup,
        l1_rates,
        l2_rates,
        inner_iters,
        inner_tol,
    ):
        return train_model_inner(
            init_params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_county_year,
            num_groups,
            feature_sizes_tup,
            l1_rates,
            l2_rates,
            inner_iters,
            inner_tol,
        )


    # Forwards pass is just solving the inner loop
    def train_fwd(
        init_params,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_county_year,
        num_groups,
        feature_sizes_tup,
        l1_rates,
        l2_rates,
        inner_iters,
        inner_tol,
    ):
        opt_params = train_model_inner(
            init_params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_county_year,
            num_groups,
            feature_sizes_tup,
            l1_rates,
            l2_rates,
            inner_iters=inner_iters,
            inner_tol=inner_tol,
        )

        # Calculate active mask at the optimum
        active_mask = jax.tree_util.tree_map(lambda x: jnp.abs(x) > 1e-6, opt_params)
        active_mask["bias"] = jnp.array(True)  # Bias is always active

        residuals = (
            opt_params,
            active_mask,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_county_year,
            num_groups,
            feature_sizes_tup,
            l1_rates,
            l2_rates,
            inner_iters,
            inner_tol,
        )
        return opt_params, residuals


    # Backwards pass requires tracing out the chain of operations to obtain derivatives
    def train_bwd(residuals, cotangents):
        (
            opt_params,
            active_mask,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_county_year,
            num_groups,
            feature_sizes_tup,
            l1_rates,
            l2_rates,
            inner_iters,
            inner_tol,
        ) = residuals

        v = cotangents  # This 'v' is a pytree matching opt_params

        # 1. Mask the RHS (cotangents 'v') for inactive components
        v_masked = jax.tree_util.tree_map(
            lambda val, mask: jnp.where(mask, val, 0.0), v, active_mask
        )

        # Ravel and unravel functions for convenience
        flat_opt_params, unravel_fn = ravel_pytree(opt_params)
        flat_v_masked, _ = ravel_pytree(v_masked)

        # 2. Define the flat smooth-only objective for Hessian calculations
        def flat_smooth_only_loss(p_flat):
            return ppml_objective_compositional_smooth_only(
                unravel_fn(p_flat),
                feature_ids,
                X_county_cont,
                X_patch_cont,
                patch_exposure,
                group_ids,
                y_county_year,
                num_groups,
                l2_rates,
            )

        # 3. Construct the Subspace Preconditioned Operator
        # This operator should apply H_smooth + ridge for active components, and identity for inactive.

        # Map l2_rates back to parameter structure for the ridge diagonal
        l2_param_pytree = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x, dtype=x.dtype), opt_params
        )
        for k, l2_val in l2_rates.items():
            l2_param_pytree["weights"][k] = jnp.full_like(
                opt_params["weights"][k], l2_val, dtype=opt_params["weights"][k].dtype
            )
        flat_l2_param_pytree, _ = ravel_pytree(l2_param_pytree)
        flat_active_mask, _ = ravel_pytree(active_mask)

        def full_hvp_plus_ridge(tangent_vec_flat):
            # Hessian-vector product of the smooth loss
            # The grad_grad_fn computes H_smooth @ v
            _, hvp_active_flat = jax.jvp(
                jax.grad(flat_smooth_only_loss), (flat_opt_params,), (tangent_vec_flat,)
            )

            # Apply ridge only to active components of the tangent vector
            # The IFT formulation usually has H^-1 (I - P_G)v, where P_G is projection onto subgradient.
            # Here, we simplify by adding ridge to H_smooth for the active set and identity for inactive.

            # For active components: (H_smooth + diag(l2_rates)) @ v_active
            # Add a strict 1e-4 numerical jitter to the active diagonal.
            # This forcibly caps the condition number to prevent solver divergence in float32.
            damping_factor = 0.1
            active_contrib_flat = jnp.where(
                flat_active_mask,
                hvp_active_flat
                + (flat_l2_param_pytree + damping_factor) * tangent_vec_flat,
                0.0,
            )
            # For inactive components: 1 * v_inactive (identity)
            inactive_contrib_flat = jnp.where(~flat_active_mask, tangent_vec_flat, 0.0)

            return active_contrib_flat + inactive_contrib_flat

        hessian_op = lx.FunctionLinearOperator(
            full_hvp_plus_ridge,
            jax.eval_shape(
                lambda: flat_opt_params
            ),  # Pass ShapeDtypeStruct; flat_opt_params.shape is tuple, breaks FLO
            tags=lx.positive_semidefinite_tag,
        )

        jax.debug.print(
            "  ->[Implicit VJP] Solving Hessian linear system for backward pass..."
        )

        # 4. float32 precision often breaks strict matrix symmetry, which instantly crashes CG
        # GMRES is immune to this.
        solver = lx.CG(rtol=1e-2, atol=1e-2, max_steps=150)
        solution = lx.linear_solve(hessian_op, -flat_v_masked, solver=solver, throw=False)

        # 5. If the matrix is still too hostile and GMRES fails, replace NaNs with 0.0
        # This safely zeroes the meta-gradient for this step  rather than corrupting the hyper-parameters
        safe_solution_value = jnp.where(jnp.isnan(solution.value), 0.0, solution.value)
        w_pytree = unravel_fn(safe_solution_value)

        jax.debug.print("  -> [Implicit VJP] Linear solve completed.")

        # 5. Algebraic KKT Output
        grad_l1 = {}
        grad_l2 = {}

        for k in feature_ids.keys():
            w_matrix_k = w_pytree["weights"][k]
            opt_w_matrix_k = opt_params["weights"][k]
            active_sub_mask_k = active_mask["weights"][k]

            # Recalculate heredity multiplier at the optimum for this order (from opt_params)
            if k == 1:
                multiplier_k = 1.0
            else:
                # Use stop_gradient on the parent magnitude at the *optimum*
                parent_magnitude_k = jax.lax.stop_gradient(
                    jnp.mean(opt_params["weights"][k - 1] ** 2)
                )
                multiplier_k = jnp.maximum(1.0, 1e-3 / (parent_magnitude_k + 1e-8))

            # Partial derivative of ALO w.r.t. L1_k (for active components)
            # This is: sum_j in S_k { w_j * multiplier_j * sign(beta_j) }
            grad_l1_k = jnp.sum(
                jnp.where(
                    active_sub_mask_k,
                    w_matrix_k * multiplier_k * jnp.sign(opt_w_matrix_k),
                    0.0,
                )
            )
            grad_l1[k] = grad_l1_k

            # Partial derivative of ALO w.r.t. L2_k (for active components)
            # This is: sum_j in S_k { w_j * 2 * beta_j }
            grad_l2_k = jnp.sum(
                jnp.where(active_sub_mask_k, w_matrix_k * 2.0 * opt_w_matrix_k, 0.0)
            )
            grad_l2[k] = grad_l2_k

        # Note: If lx.CG stagnates in float32 during finite difference tests due to high condition numbers,
        # consider locally promoting CG solver inputs to jnp.float64 or temporarily increasing RIDGE to 1e-4.
        return (
            None,  # init_params
            None,  # feature_ids
            None,  # X_county_cont
            None,  # X_patch_cont
            None,  # patch_exposure
            None,  # group_ids
            None,  # y_county_year
            None,  # num_groups
            None,  # feature_sizes_tup
            grad_l1,  # l1_rates
            grad_l2,  # l2_rates,
            None,  # inner_iters
            None,  # inner_tols
        )


    train_model_cpu.defvjp(train_fwd, train_bwd)
    return (train_model_cpu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ALO-CV using lineax to take advantage of JAX autodiff
    """)
    return


@app.cell
def _(
    compute_patch_log_worker,
    jax,
    jnp,
    lx,
    ppml_objective_compositional_smooth_only,
    ravel_pytree,
):
    def compute_alo_compositional(
        params,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_county_year,
        num_groups,
        l1_rates,
        l2_rates,
        alo_k,
        alo_cg_rtol,
        alo_cg_atol,
        alo_cg_max_steps,
    ):

        flat_p, unravel_fn = ravel_pytree(params)

        # 1. Active Mask & L2 Parameter Mapping (For the exact operator)
        active_mask = jax.tree_util.tree_map(lambda x: jnp.abs(x) > 1e-6, params)
        active_mask["bias"] = jnp.array(True)
        flat_active_mask, _ = ravel_pytree(active_mask)

        l2_param_pytree = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        for k, l2_val in l2_rates.items():
            l2_param_pytree["weights"][k] = jnp.full_like(params["weights"][k], l2_val)
        flat_l2_param_pytree, _ = ravel_pytree(l2_param_pytree)

        # 2. Smooth-Only Loss (Crucial: Do not take Hessian of L1 penalty!)
        def flat_smooth_only_loss(p_flat):
            return ppml_objective_compositional_smooth_only(
                unravel_fn(p_flat),
                feature_ids,
                X_county_cont,
                X_patch_cont,
                patch_exposure,
                group_ids,
                y_county_year,
                num_groups,
                l2_rates,
            )

        def all_groups_nll(p_flat):
            p = unravel_fn(p_flat)
            log_w = compute_patch_log_worker(
                p,
                feature_ids,
                X_county_cont,
                X_patch_cont,
                group_ids,
            )
            workers_in_patch = patch_exposure * jnp.exp(log_w)
            mu_cy = jax.ops.segment_sum(workers_in_patch, group_ids, num_groups)
            return mu_cy - y_county_year * jnp.log(mu_cy + 1e-7)

        # 3. Fast, Matrix-Free Trace Estimator
        def compute_leverages_no_grad(p_flat):
            # Use the stabilized HVP operator from Phase 3
            def full_hvp_plus_ridge(tangent_vec_flat):
                _, hvp_active_flat = jax.jvp(
                    jax.grad(flat_smooth_only_loss), (p_flat,), (tangent_vec_flat,)
                )

                # 0.1 damping to stabilize CG
                active_contrib = jnp.where(
                    flat_active_mask,
                    hvp_active_flat + (flat_l2_param_pytree + 0.1) * tangent_vec_flat,
                    0.0,
                )
                inactive_contrib = jnp.where(~flat_active_mask, tangent_vec_flat, 0.0)
                return active_contrib + inactive_contrib

            hessian_op = lx.FunctionLinearOperator(
                full_hvp_plus_ridge,
                jax.eval_shape(lambda: p_flat),
                tags=lx.positive_semidefinite_tag,
            )

            solver = lx.CG(
                rtol=alo_cg_rtol,
                atol=alo_cg_atol,
                max_steps=alo_cg_max_steps,
            )

            # Hutchinson estimator. Keep K low because each draw requires a large
            # gradient and linear solve over the full patch panel.
            K = alo_k

            key = jax.random.PRNGKey(42)
            Z = jax.random.rademacher(key, (K, num_groups), dtype=p_flat.dtype)

            def compute_trace_sample(z):
                def z_weighted_nll(p):
                    return jnp.sum(z * all_groups_nll(p))

                v = jax.grad(z_weighted_nll)(p_flat)

                # Mask the RHS
                v_masked = jnp.where(flat_active_mask, v, 0.0)

                solution = lx.linear_solve(hessian_op, v_masked, solver=solver, throw=False)

                # NaN Guard
                w_safe = jnp.where(jnp.isnan(solution.value), 0.0, solution.value)

                return jnp.dot(v_masked, w_safe)

            def scan_trace(total, z):
                return total + compute_trace_sample(z), None

            # Scan the draws sequentially. vmap launches the solves in parallel and
            # duplicates the HVP working set, which is the allocation that breaks
            # once crop-specific GDD columns are included.

            trace_total, _ = jax.lax.scan(scan_trace, jnp.array(0.0, p_flat.dtype), Z)
            return trace_total / K

        # 4. Stop gradients on the leverage calculation to prevent infinite meta-autodiff loops
        if alo_k == 0:
            sum_leverages = jnp.array(0.0, dtype=flat_p.dtype)
        else:
            sum_leverages = jax.lax.stop_gradient(compute_leverages_no_grad(flat_p))

        # 5. ALO Score calculation
        group_nlls = all_groups_nll(flat_p)
        alo_score = jnp.mean(group_nlls) + (sum_leverages / (num_groups**2))

        jax.debug.print(
            "  -> [ALO-CV] Stochastic ALO: {alo:.4f} | Active Params: {k}",
            alo=alo_score,
            k=jnp.sum(flat_active_mask),
        )

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
    def meta_loss_fn(
        raw_l1,
        raw_l2,
        init_params,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        acreage,
        group_ids,
        y_county_year,
        num_groups,
        feature_sizes_tup,
        inner_iters,
        inner_tol,
        alo_k,
        alo_cg_rtol,
        alo_cg_atol,
        alo_cg_max_steps,
    ):
        """
        Compute ALO-CV score given L1 and L2 params
        """
        # Enforce strict positivity constraints
        l1_rates = {k: jax.nn.softplus(v) for k, v in raw_l1.items()}
        l2_rates = {k: jax.nn.softplus(v) + 1e-4 for k, v in raw_l2.items()}

        # 1. Train Inner Model
        opt_params = train_model_cpu(
            init_params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            acreage,
            group_ids,
            y_county_year,
            num_groups,
            feature_sizes_tup,
            l1_rates,
            l2_rates,
            inner_iters,
            inner_tol,
        )

        # 2. Evaluate with ALO-CV
        alo = compute_alo_compositional(
            opt_params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            acreage,
            group_ids,
            y_county_year,
            num_groups,
            l1_rates,
            l2_rates,
            alo_k,
            alo_cg_rtol,
            alo_cg_atol,
            alo_cg_max_steps,
        )

        return alo, opt_params


    # JIT the meta-gradient calculator
    meta_val_and_grad = jax.jit(
        jax.value_and_grad(meta_loss_fn, argnums=(0, 1), has_aux=True),
        static_argnames=[
            "num_groups",
            "feature_sizes_tup",
            "inner_iters",
            "inner_tol",
            "alo_k",
            "alo_cg_rtol",
            "alo_cg_atol",
            "alo_cg_max_steps",
        ],
    )
    return (meta_val_and_grad,)


@app.cell
def _(jax, jnp, meta_val_and_grad, optax, partial):
    @partial(
        jax.jit,
        static_argnames=[
            "num_groups",
            "feature_sizes_tup",
            "target_iters",
            "patience",
            "outer_tol",
            "meta_optimizer",
            "inner_iters",
            "inner_tol",
            "alo_k",
            "alo_cg_rtol",
            "alo_cg_atol",
            "alo_cg_max_steps",
        ],
    )
    def compiled_meta_loop(
        carry_init,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        acreage,
        group_ids,
        y_county_year,
        num_groups,
        feature_sizes_tup,
        target_iters,
        patience,
        outer_tol,
        meta_optimizer,
        inner_iters,
        inner_tol,
        alo_k,
        alo_cg_rtol,
        alo_cg_atol,
        alo_cg_max_steps,
    ):
        """
        Natively compiled XLA outer While Loop. Runs entirely on GPU.
        """

        def cond_fn(carry):
            # 1. Unpack the exact state we need to check
            i, chunk_i, _, _, _, _, wait, _, _ = carry

            # 2. Loop continues ONLY if both conditions are True
            not_max_iter = i < target_iters
            not_out_of_patience = wait < patience

            return jnp.logical_and(not_max_iter, not_out_of_patience)

        def body_fn(carry):
            # 1. Unpack the carry
            (
                i,
                chunk_i,
                hparams,
                opt_state,
                best_hparams,
                best_alo,
                wait,
                history_buffer,
                inner_params,
            ) = carry
            raw_l1, raw_l2 = hparams

            # 2. Execute the Inner Bilevel Step (Computes Loss & Gradients)
            # Passes previous opt_params as initial inner_params
            (alo, new_inner_params), (gl1, gl2) = meta_val_and_grad(
                raw_l1,
                raw_l2,
                inner_params,
                feature_ids,
                X_county_cont,
                X_patch_cont,
                acreage,
                group_ids,
                y_county_year,
                num_groups,
                feature_sizes_tup,
                inner_iters,
                inner_tol,
                alo_k,
                alo_cg_rtol,
                alo_cg_atol,
                alo_cg_max_steps,
            )

            # 3. Early Stopping Math
            improved = alo < (best_alo - outer_tol)
            new_best_alo = jnp.where(improved, alo, best_alo)
            new_wait = jnp.where(improved, 0, wait + 1)

            # 4. Conditionally update PyTrees (Dictionaries)
            # We cannot use jnp.where directly on dicts. We map over the leaves.
            new_best_hparams = jax.tree_util.tree_map(
                lambda current_val, best_val: jnp.where(improved, current_val, best_val),
                hparams,
                best_hparams,
            )

            # 5. Apply Meta-Optimizer Updates
            # We always apply gradients to explore the space, even if we didn't improve.
            updates, new_opt_state = meta_optimizer.update((gl1, gl2), opt_state, hparams)
            new_hparams = optax.apply_updates(hparams, updates)

            # 6. Calculate and store telemetry in VRAM; telemetry extracted asynchronously
            act_l1_vals = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(jax.nn.softplus, raw_l1)
            )
            mean_l1 = jnp.mean(jnp.array(act_l1_vals))

            # Pack the current step metrics into a 1D vector
            new_metric_row = jnp.array([i, alo, mean_l1, new_wait], dtype=jnp.float32)

            # Insert the vector into the history buffer matrix at the current chunk index
            new_history_buffer = history_buffer.at[chunk_i].set(new_metric_row)

            # jax.debug.print does not break JIT; it streams to stdout asynchronously
            jax.debug.print(
                "Step {i} | ALO-CV: {alo} | Mean L1: {l1} | Wait: {w}/{p}",
                i=i,
                alo=alo,
                l1=mean_l1,
                w=new_wait,
                p=patience,
            )

            # 7. Repack carry, incrementing both the global step (i) and local step (chunk_i)
            return (
                i + 1,
                chunk_i + 1,
                new_hparams,
                new_opt_state,
                new_best_hparams,
                new_best_alo,
                new_wait,
                new_history_buffer,
                new_inner_params,
            )

        # Execute the loop
        final_carry = jax.lax.while_loop(cond_fn, body_fn, carry_init)
        return final_carry

    return (compiled_meta_loop,)


@app.cell
def _(optax):
    # We want to use ADAM as our hyperparameter optimizer
    def make_meta_optimizer(meta_lr=0.01, clip_norm=1.0):
        return optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(learning_rate=meta_lr),
        )

    return (make_meta_optimizer,)


@app.cell
def _(jnp):
    def restore_leaves(loaded_leaves, template_leaves):
        restored = []
        for loaded, template in zip(loaded_leaves, template_leaves):
            if hasattr(template, "dtype"):
                restored.append(jnp.array(loaded, dtype=template.dtype))
            else:
                restored.append(loaded)
        return restored

    return (restore_leaves,)


@app.function
def to_serializable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return x


@app.cell
def _(
    code_path,
    compiled_meta_loop,
    get_hparam_templates,
    initialize_params,
    jax,
    jax_inv_softplus,
    jnp,
    load_meta_checkpoint,
    make_meta_optimizer,
    save_meta_checkpoint,
    time,
):
    def run_meta_optimization_chunked(
        feature_ids,
        X_county_cont,
        X_patch_cont,
        acreage,
        group_ids,
        y_county_year,
        num_groups,
        feature_sizes_tup,
        outer_iters=400,
        outer_tol=1e-3,
        patience=30,
        chunk_size=20,
        inner_iters=250,
        inner_tol=1e-4,
        alo_k=1,
        alo_cg_rtol=5e-2,
        alo_cg_atol=5e-2,
        alo_cg_max_steps=50,
        checkpoint_path=None,
        log_path=None,
        meta_lr=0.01,  # ADAM params
        clip_norm=1.0,
        reset=False,
    ):
        if checkpoint_path is None:
            checkpoint_path = code_path / "json" / "meta_ppml_opt_checkpoint.json"
        if log_path is None:
            log_path = code_path / "json" / "meta_ppml_opt_log.csv"

        meta_optimizer = make_meta_optimizer(meta_lr=meta_lr, clip_norm=clip_norm)

        # ---------------------------------------------------------
        # STATE INITIALIZATION & ROBUST DESERIALIZATION
        # ---------------------------------------------------------
        if checkpoint_path.exists() and not reset:
            hparams_template, state_template = get_hparam_templates(
                meta_optimizer, jax_inv_softplus
            )
            num_continuous = X_county_cont.shape[1] + X_patch_cont.shape[1]
            inner_params_template = initialize_params(feature_sizes_tup, num_continuous)
            checkpoint = load_meta_checkpoint(
                checkpoint_path,
                hparams_template=hparams_template,
                state_template=state_template,
                inner_params_template=inner_params_template,
            )
            hparams = checkpoint["hparams"]
            state = checkpoint["opt_state"]
            best_hparams = checkpoint["best_hparams"]
            best_alo = checkpoint["best_alo"]
            wait = checkpoint["wait"]
            inner_params = checkpoint["inner_params"]
            start_iter = checkpoint["iter"]

            print(
                f"\n--- Resuming Bilevel Hyperparameter Tuning from {checkpoint_path.name} ---"
            )
            print(
                f"Resuming at Iteration {start_iter} (Best ALO so far: {float(best_alo):.4f} | Patience: {int(wait)}/{patience})"
            )

        else:
            print("\n--- Starting Bilevel Hyperparameter Tuning from Scratch ---")
            hparams, state = get_hparam_templates(meta_optimizer, jax_inv_softplus)
            best_hparams = hparams
            best_alo = jnp.array(jnp.inf, dtype=jnp.float32)
            wait = jnp.array(0, dtype=jnp.int32)
            start_iter = 0
            # Generate initial weights from scratch if not previously logged
            num_continuous = X_county_cont.shape[1] + X_patch_cont.shape[1]
            inner_params = initialize_params(feature_sizes_tup, num_continuous)

            with open(log_path, "w") as f:
                f.write("Step,ALO_CV_Score,Mean_L1_Penalty,Wait\n")

        print(
            f"--- Chunked XLA Meta-Optimization (Chunk Size: {chunk_size}, Total Target: {outer_iters}) ---"
        )

        # CHUNKING loop starts here
        current_iter = start_iter

        # wait defensively wrapped with int()
        while current_iter < outer_iters and int(wait) < patience:
            iters_this_chunk = min(chunk_size, outer_iters - current_iter)
            target_iter = current_iter + iters_this_chunk

            # Pre-allocate telemetry buffer
            history_buffer = jnp.zeros((chunk_size, 4), dtype=jnp.float32)
            chunk_idx_start = jnp.array(0, dtype=jnp.int32)

            carry_init = (
                jnp.array(current_iter, dtype=jnp.int32),
                chunk_idx_start,
                hparams,
                state,
                best_hparams,
                best_alo,
                wait,
                history_buffer,
                inner_params,
            )

            # Dispatch to GPU for 20 loop chunk
            t0 = time.time()
            final_carry = compiled_meta_loop(
                carry_init,
                feature_ids,
                X_county_cont,
                X_patch_cont,
                acreage,
                group_ids,
                y_county_year,
                num_groups,
                feature_sizes_tup,
                target_iters=target_iter,
                patience=patience,
                outer_tol=outer_tol,
                meta_optimizer=meta_optimizer,
                inner_iters=inner_iters,
                inner_tol=inner_tol,
                alo_k=alo_k,
                alo_cg_rtol=alo_cg_rtol,
                alo_cg_atol=alo_cg_atol,
                alo_cg_max_steps=alo_cg_max_steps,
            )
            final_carry = jax.block_until_ready(final_carry)
            t1 = time.time()

            # Unpack carry
            (
                current_iter_jax,
                chunk_steps_completed,
                hparams,
                state,
                best_hparams,
                best_alo,
                wait,
                finished_history,
                inner_params,
            ) = final_carry
            current_iter = int(current_iter_jax)

            print(
                f"  [Host Sync] Chunk completed in {t1 - t0:.2f}s. Global Iter: {current_iter}. Best ALO: {float(best_alo):.4f}"
            )

            # Write telemetry to CSV, async after each GPU chunk
            with open(log_path, "a") as f:
                for row in range(int(chunk_steps_completed)):
                    step_val, alo_val, l1_val, wait_val = finished_history[row]
                    f.write(
                        f"{int(step_val)},{float(alo_val):.4f},{float(l1_val):.6f},{int(wait_val)}\n"
                    )

            save_meta_checkpoint(
                checkpoint_path,
                iteration=current_iter,
                hparams=hparams,
                opt_state=state,
                best_hparams=best_hparams,
                best_alo=best_alo,
                wait=wait,
                inner_params=inner_params,
            )

        # Final extraction (defensively wrapped with int())
        if int(wait) >= patience:
            print(
                f"\n[!] Convergence Reached: ALO-CV Score has not improved by {outer_tol} in {patience} steps."
            )
        elif current_iter >= outer_iters:
            print(f"\n[!] Maximum target iterations ({outer_iters}) reached.")

        best_raw_l1, best_raw_l2 = best_hparams
        final_l1 = {k: jax.nn.softplus(v) for k, v in best_raw_l1.items()}
        final_l2 = {k: jax.nn.softplus(v) + 1e-4 for k, v in best_raw_l2.items()}

        return final_l1, final_l2, inner_params

    return (run_meta_optimization_chunked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create functions for robust restarting and refinement
    """)
    return


@app.cell
def _(jax, json):
    def save_meta_checkpoint(
        path,
        *,
        iteration,
        hparams,
        opt_state,
        best_hparams,
        best_alo,
        wait,
        inner_params,
    ):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = {
            "iter": int(iteration),
            "hparams_leaves": [
                to_serializable(x) for x in jax.tree_util.tree_leaves(hparams)
            ],
            "state_leaves": [
                to_serializable(x) for x in jax.tree_util.tree_leaves(opt_state)
            ],
            "best_hparams_leaves": [
                to_serializable(x) for x in jax.tree_util.tree_leaves(best_hparams)
            ],
            "best_alo": float(best_alo),
            "wait": int(wait),
            "inner_params_leaves": [
                to_serializable(x) for x in jax.tree_util.tree_leaves(inner_params)
            ],
        }
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)
        tmp_path.replace(path)

    return (save_meta_checkpoint,)


@app.cell
def _(jax, jnp, json, restore_leaves):
    def load_meta_checkpoint(
        path,
        *,
        hparams_template,
        state_template,
        inner_params_template,
    ):
        with open(path, "r") as f:
            chkpt = json.load(f)

        hparams_template_leaves, hparams_treedef = jax.tree_util.tree_flatten(
            hparams_template
        )
        state_template_leaves, state_treedef = jax.tree_util.tree_flatten(state_template)
        inner_template_leaves, inner_treedef = jax.tree_util.tree_flatten(
            inner_params_template
        )

        hparams = jax.tree_util.tree_unflatten(
            hparams_treedef,
            restore_leaves(chkpt["hparams_leaves"], hparams_template_leaves),
        )
        opt_state = jax.tree_util.tree_unflatten(
            state_treedef,
            restore_leaves(chkpt["state_leaves"], state_template_leaves),
        )
        best_hparams = jax.tree_util.tree_unflatten(
            hparams_treedef,
            restore_leaves(chkpt["best_hparams_leaves"], hparams_template_leaves),
        )
        inner_params = jax.tree_util.tree_unflatten(
            inner_treedef,
            restore_leaves(chkpt["inner_params_leaves"], inner_template_leaves),
        )

        return {
            "iter": int(chkpt["iter"]),
            "hparams": hparams,
            "opt_state": opt_state,
            "best_hparams": best_hparams,
            "best_alo": jnp.array(chkpt["best_alo"], dtype=jnp.float32),
            "wait": jnp.array(chkpt["wait"], dtype=jnp.int32),
            "inner_params": inner_params,
        }

    return (load_meta_checkpoint,)


@app.cell
def _(jnp):
    def get_hparam_templates(meta_optimizer, jax_inv_softplus):
        raw_l1 = {
            k: jnp.array(jax_inv_softplus(0.01 * (2.0 ** (k - 1))), dtype=jnp.float32)
            for k in range(1, 4)
        }
        raw_l2 = {
            k: jnp.array(jax_inv_softplus(0.005 * (2.0 ** (k - 1))), dtype=jnp.float32)
            for k in range(1, 4)
        }
        hparams = (raw_l1, raw_l2)
        state = meta_optimizer.init(hparams)
        return hparams, state

    return (get_hparam_templates,)


@app.cell
def _(get_hparam_templates, jax, load_meta_checkpoint, save_meta_checkpoint):
    def create_restart_from_best_checkpoint(
        source_checkpoint_path,
        restart_checkpoint_path,
        *,
        meta_optimizer,
        initialize_params,
        train_model_inner,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_target_count,
        num_groups,
        feature_sizes_tup,
        jax_inv_softplus,
        refit_inner_iters=1000,
        refit_inner_tol=1e-6,
        restart_iter=0,
    ):
        hparams_template, state_template = get_hparam_templates(
            meta_optimizer,
            jax_inv_softplus,
        )

        num_continuous = X_county_cont.shape[1] + X_patch_cont.shape[1]
        inner_template = initialize_params(feature_sizes_tup, num_continuous)

        loaded = load_meta_checkpoint(
            source_checkpoint_path,
            hparams_template=hparams_template,
            state_template=state_template,
            inner_params_template=inner_template,
        )

        best_hparams = loaded["best_hparams"]
        best_raw_l1, best_raw_l2 = best_hparams

        best_l1 = {k: jax.nn.softplus(v) for k, v in best_raw_l1.items()}
        best_l2 = {k: jax.nn.softplus(v) + 1e-4 for k, v in best_raw_l2.items()}

        # Use loaded inner params as warm start, but refit under best hparams.
        best_inner_params = train_model_inner(
            loaded["inner_params"],
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_target_count,
            num_groups,
            feature_sizes_tup,
            best_l1,
            best_l2,
            inner_iters=refit_inner_iters,
            inner_tol=refit_inner_tol,
        )

        fresh_state = meta_optimizer.init(best_hparams)

        save_meta_checkpoint(
            restart_checkpoint_path,
            iteration=restart_iter,
            hparams=best_hparams,
            opt_state=fresh_state,
            best_hparams=best_hparams,
            best_alo=loaded["best_alo"],
            wait=0,
            inner_params=best_inner_params,
        )

        return restart_checkpoint_path

    return


@app.cell
def _(get_hparam_templates, jax, load_meta_checkpoint):
    def evaluate_checkpoint_best_hparams(
        checkpoint_path,
        *,
        meta_optimizer,
        initialize_params,
        train_model_inner,
        compute_alo_compositional,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_target_count,
        num_groups,
        feature_sizes_tup,
        jax_inv_softplus,
        inner_iters=1000,
        inner_tol=1e-6,
        alo_k=2,
        alo_cg_rtol=1e-2,
        alo_cg_atol=1e-2,
        alo_cg_max_steps=150,
    ):
        hparams_template, state_template = get_hparam_templates(
            meta_optimizer,
            jax_inv_softplus,
        )

        num_continuous = X_county_cont.shape[1] + X_patch_cont.shape[1]
        inner_template = initialize_params(feature_sizes_tup, num_continuous)

        loaded = load_meta_checkpoint(
            checkpoint_path,
            hparams_template=hparams_template,
            state_template=state_template,
            inner_params_template=inner_template,
        )

        best_raw_l1, best_raw_l2 = loaded["best_hparams"]
        best_l1 = {k: jax.nn.softplus(v) for k, v in best_raw_l1.items()}
        best_l2 = {k: jax.nn.softplus(v) + 1e-4 for k, v in best_raw_l2.items()}

        params = train_model_inner(
            loaded["inner_params"],
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_target_count,
            num_groups,
            feature_sizes_tup,
            best_l1,
            best_l2,
            inner_iters=inner_iters,
            inner_tol=inner_tol,
        )

        alo = compute_alo_compositional(
            params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            y_target_count,
            num_groups,
            best_l1,
            best_l2,
            alo_k,
            alo_cg_rtol,
            alo_cg_atol,
            alo_cg_max_steps,
        )

        return alo, params, best_l1, best_l2

    return (evaluate_checkpoint_best_hparams,)


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
    county_cont_cols,
    h2a,
    np,
    patch_cont_cols,
    prep_jax_composition_arrays,
    soil,
):
    # Prep input data
    (
        feature_ids,
        feature_sizes,
        feature_names,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_target_count,
        num_groups,
        merged_df,
    ) = prep_jax_composition_arrays(
        h2a,
        bea,
        soil,
        climate,
        county_cont_cols,
        patch_cont_cols,
        cat_cols,
        max_order=3,
    )
    feature_sizes_tup = tuple(sorted(feature_sizes.items()))

    # Check for any remaining NaNs or Infs
    assert not np.isnan(y_target_count).any(), "NaNs detected in target count!"
    assert not np.isinf(y_target_count).any(), "Infs detected in target count!"
    print(f"Target Max Count: {np.max(y_target_count):.4f}")
    print(f"Target Min Count: {np.min(y_target_count):.4f}")
    return (
        X_county_cont,
        X_patch_cont,
        feature_ids,
        feature_sizes_tup,
        group_ids,
        merged_df,
        num_groups,
        patch_exposure,
        y_target_count,
    )


@app.cell
def _(
    X_county_cont,
    X_patch_cont,
    code_path,
    feature_ids,
    feature_sizes_tup,
    group_ids,
    num_groups,
    patch_exposure,
    run_meta_optimization_chunked,
    time,
    y_target_count,
):
    start_time = time.perf_counter()
    # Tune L1 and L2 (full GPU dispatch)
    checkpoint_path = code_path / "json" / "meta_ppml_opt_checkpoint.json"
    log_path = code_path / "json" / "meta_ppml_opt_log.csv"
    _best_l1, _best_l2, _final_inner_param = run_meta_optimization_chunked(
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        y_target_count,
        num_groups,
        feature_sizes_tup,
        outer_iters=1000,
        patience=50,
        chunk_size=10,
        inner_iters=500,
        inner_tol=1e-4,
        alo_k=1,
        alo_cg_rtol=5e-2,
        alo_cg_atol=5e-2,
        alo_cg_max_steps=50,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        meta_lr=0.01,
        clip_norm=1.0,
        reset=False,
    )
    return checkpoint_path, start_time


@app.cell
def _(start_time, time):
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check best iter with stricter parameters
    """)
    return


@app.cell
def _(
    X_county_cont,
    X_patch_cont,
    checkpoint_path,
    compute_alo_compositional,
    evaluate_checkpoint_best_hparams,
    feature_ids,
    feature_sizes_tup,
    group_ids,
    initialize_params,
    jax_inv_softplus,
    make_meta_optimizer,
    num_groups,
    patch_exposure,
    train_model_inner,
    y_target_count,
):
    meta_optimizer = make_meta_optimizer(meta_lr=0.01, clip_norm=1.0)
    alo_check, trained_params, best_l1, best_l2 = evaluate_checkpoint_best_hparams(
        checkpoint_path,
        meta_optimizer=meta_optimizer,
        initialize_params=initialize_params,
        train_model_inner=train_model_inner,
        compute_alo_compositional=compute_alo_compositional,
        feature_ids=feature_ids,
        X_county_cont=X_county_cont,
        X_patch_cont=X_patch_cont,
        patch_exposure=patch_exposure,
        group_ids=group_ids,
        y_target_count=y_target_count,
        num_groups=num_groups,
        feature_sizes_tup=feature_sizes_tup,
        jax_inv_softplus=jax_inv_softplus,
        inner_iters=1000,
        inner_tol=1e-6,
        alo_k=2,
        alo_cg_rtol=1e-2,
        alo_cg_atol=1e-2,
        alo_cg_max_steps=150,
    )
    print("\n--- Model Estimation with Optimized Penalty Parameters ---")
    print(f"Best-check ALO-CV: {float(alo_check):.4f}")
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
    for order, w in sorted(trained_params["weights"].items()):
        active_weights = jnp.sum(jnp.abs(w) > 1e-3)
        print(f"  Order {order} total weights active: {active_weights} / {w.size}")
    return


@app.cell
def _(compute_patch_log_worker, jax, jnp):
    def predicted_h2a_county_year(
        params,
        feature_ids,
        X_county_cont,
        X_patch_cont,
        patch_exposure,
        group_ids,
        num_groups,
    ):
        log_worker_per_patch = compute_patch_log_worker(
            params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            group_ids,
        )
        workers_in_patch = patch_exposure * jnp.exp(log_worker_per_patch)
        return jax.ops.segment_sum(workers_in_patch, group_ids, num_groups)

    return (predicted_h2a_county_year,)


@app.cell
def _(
    X_county_cont,
    X_patch_cont,
    binary_path,
    feature_ids,
    group_ids,
    merged_df,
    np,
    num_groups,
    patch_exposure,
    pl,
    predicted_h2a_county_year,
    trained_params,
):
    y_pred_count = np.array(
        predicted_h2a_county_year(
            trained_params,
            feature_ids,
            X_county_cont,
            X_patch_cont,
            patch_exposure,
            group_ids,
            num_groups,
        )
    )
    county_ansi_year_group_id = merged_df.select(
        ["county_ansi", "year", "group_id"]
    ).unique()
    results_df = (
        pl.DataFrame(
            {
                "group_id": np.arange(num_groups),
                "predicted_h2a_count": y_pred_count,
            }
        )
        .join(county_ansi_year_group_id, on="group_id")
        .group_by("county_ansi")
        .agg(pl.mean("predicted_h2a_count"))
    )
    results_df.write_parquet(binary_path / "h2a_prediction_using_elastic_net.parquet")
    return (results_df,)


@app.cell
def _(results_df):
    results_df
    return


if __name__ == "__main__":
    app.run()
