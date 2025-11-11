import numpy as np
from scipy.stats import rankdata, norm
import pandas as pd



def transform_dataset_into_gaussian(df):
    """
    Transform each column of the input DataFrame into values following
    a standard normal distribution by:
      1) Computing the empirical CDF (continuous or categorical)
      2) Applying the inverse Gaussian (probit) transform
    """
    z_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            u = empirical_cdf_continuous(df[col])
        else:
            labels = list(np.sort(df[col].dropna().unique()))
            u = empirical_cdf_categorical_column(df[col], labels)
        z_df[col] = uniform_to_gaussian(u)
    return z_df



def empirical_cdf_continuous(column, integer_tolerance=1e-12, seed=None):
    """
    Numeric ECDF → (0,1).

    - Continuous values: average-rank plotting positions (rank-0.5)/n (deterministic).
    - Integer-valued columns (even if dtype=float like 4.0):
        For each unique integer value v with mass p_v and left edge L_v of its ECDF bin,
        assign u = L_v + r * p_v with r ~ Uniform(0,1), independently per occurrence.
        (Vectorized; uses optional `seed` for reproducibility.)

    NaNs are preserved as NaN in u.
    """
    rng = np.random.default_rng(seed)

    arr = np.asarray(column, float)
    mask_nan = np.isnan(arr)
    valid = arr[~mask_nan]
    n_valid = valid.size

    if n_valid == 0:
        return np.full(arr.shape, np.nan, dtype=float)

    # Detect integer-valued column (values like 4.0 count as integer)
    is_integer_valued = np.allclose(valid, np.round(valid), atol=integer_tolerance, rtol=0.0)

    u = np.full(arr.shape, np.nan, dtype=float)

    if not is_integer_valued:
        # ---- Continuous case: deterministic average-rank plotting positions ----
        ranks_valid = rankdata(valid, method='average')  # 1..n
        u_valid = (ranks_valid - 0.5) / n_valid
        u[~mask_nan] = u_valid
        return u

    # ---- Integer-valued case: random Uniform(L, R) within each bin (vectorized) ----
    # Unique sorted integer values and counts
    uniques, counts = np.unique(valid, return_counts=True)
    counts = counts.astype(float)
    total = counts.sum()

    # Bin probabilities and cumulative edges
    p = counts / total                       # p_k
    P = np.cumsum(p)                         # [P1, P2, ..., PK]
    L = np.concatenate([[0.0], P[:-1]])      # left edges L_k

    # Map each valid value to its index k in uniques
    # (uniques is sorted; values are exactly integer-valued by construction)
    idx_map = {val: k for k, val in enumerate(uniques)}
    idx = np.fromiter((idx_map[v] for v in valid), dtype=int, count=n_valid)

    # Draw one r ~ U(0,1) per occurrence and place inside its bin
    r = rng.random(n_valid)
    u_valid = L[idx] + r * p[idx]

    # Keep strictly inside (0,1) for numerical stability (e.g., probit later)
    eps = 1e-12
    u_valid = np.clip(u_valid, eps, 1 - eps)

    u[~mask_nan] = u_valid
    return u

def empirical_cdf_categorical_column(column, sorted_labels, seed=None, treat_nan_as_category=True):
    """
    Fast, vectorized categorical → (0,1).
    For each occurrence of label ℓ, draw u = P_prev[ℓ] + r * p[ℓ], r~U(0,1).

    If treat_nan_as_category=True, NaNs are treated as a separate category with
    their own probability mass (instead of being left as NaN).
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    arr = np.asarray(column, dtype=object)

    # 🔹 Optionally treat NaN as a valid category
    if treat_nan_as_category:
        # replace actual NaNs with a string token
        arr_filled = np.array(["<NaN>" if pd.isna(v) else v for v in arr], dtype=object)
        labels = sorted_labels + ["<NaN>"] if "<NaN>" not in sorted_labels else sorted_labels
    else:
        arr_filled = arr[~pd.isna(arr)]
        labels = sorted_labels

    if arr_filled.size == 0:
        return np.full(arr.shape, np.nan, dtype=float)

    # counts in the provided label order
    uniq, cnt = np.unique(arr_filled, return_counts=True)
    count_map = {l: c for l, c in zip(uniq, cnt)}
    counts_in_order = np.array([count_map.get(l, 0) for l in labels], dtype=float)

    total = counts_in_order.sum()
    if total == 0:
        return np.full(arr.shape, np.nan, dtype=float)

    # probabilities and cumulative edges
    p = counts_in_order / total
    P = np.cumsum(p)
    P_prev = np.concatenate([[0.0], P[:-1]])

    # lookups
    p_map = {l: pk for l, pk in zip(labels, p)}
    L_map = {l: Lk for l, Lk in zip(labels, P_prev)}

    # draw uniforms for all entries (NaNs included if treated as category)
    p_vec = np.array([p_map.get(v, 0.0) for v in arr_filled], dtype=float)
    L_vec = np.array([L_map.get(v, 0.0) for v in arr_filled], dtype=float)
    r = rng.random(size=arr_filled.shape[0])
    u_obs = L_vec + r * p_vec

    # 🔹 Return array aligned to original input
    if treat_nan_as_category:
        return u_obs
    else:
        u = np.full(arr.shape, np.nan, dtype=float)
        u[~pd.isna(arr)] = u_obs
        return u



def transform_dataset_from_gaussian(z_df, df_original):
    """
    Transforms a DataFrame of correlated Gaussian samples back to the original
    data's domain, preserving marginal distributions and enforcing original data types.
    """
    print("Transforming data from Gaussian space...")

    # --- Step 1: Efficiently classify columns based on their data type ---
    # This is cleaner than looping through dtypes.
    categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df_original.select_dtypes(include=np.number).columns.tolist()
    
    integer_cols = []
    decimal_cols = []
    for col in numeric_cols:
        # Your original logic was good for detecting integer-like floats.
        if not np.allclose(df_original[col].dropna() % 1, 0):
            decimal_cols.append(col)
        else:
            integer_cols.append(col)

    # --- Step 2: Generate synthetic data using the inverse transform for each column ---
    df_synthetic = pd.DataFrame(index=z_df.index, columns=z_df.columns)

    for col in z_df.columns:
        # This core logic remains the same as your original method.
        u_samples = gaussian_to_uniform(z_df[col])
        
        if col in integer_cols:
            sorted_vals = np.sort(df_original[col].dropna().values)
            nan_frac = df_original[col].isna().mean()
            df_synthetic[col] = inverse_empirical_cdf_integer(u_samples, sorted_vals, nan_frac=nan_frac)

        elif col in categorical_cols:
            orig_obj = df_original[col].astype(object)
            vals_for_counts = np.where(pd.isna(orig_obj), "NaN", orig_obj)
            non_nan_labels = list(np.sort(df_original[col].dropna().unique()))
            sorted_labels = non_nan_labels + ["NaN"]
            
            uniq, cnt = np.unique(vals_for_counts, return_counts=True)
            count_map = {l: c for l, c in zip(uniq, cnt)}
            counts_in_order = np.array([count_map.get(l, 0) for l in sorted_labels], dtype=float)
            
            out_labels = inverse_empirical_cdf_categorical(u_samples, sorted_labels, counts_in_order, nan_frac=0.0)
            
            out_labels = out_labels.astype(object)
            out_labels[out_labels == "NaN"] = np.nan
            df_synthetic[col] = out_labels

        else: 
            sorted_vals = np.sort(df_original[col].dropna().values)
            nan_frac = df_original[col].isna().mean()
            df_synthetic[col] = inverse_empirical_cdf_decimal_interp(u_samples, sorted_vals, nan_frac=nan_frac)

    print("Enforcing original data types for structural similarity...")
    for col in df_original.columns:
        original_dtype = df_original[col].dtype
        if pd.api.types.is_integer_dtype(original_dtype) and df_synthetic[col].isnull().any():
            df_synthetic[col] = df_synthetic[col].astype('Int64')
        else:
            try:
                df_synthetic[col] = df_synthetic[col].astype(original_dtype)
            except Exception as e:
                print(f"Warning: Could not cast column '{col}' to {original_dtype}. Error: {e}")

    return df_synthetic


def gaussian_to_uniform(z):
    """
    Maps z∼N(0,1) → u in (0,1) via Φ, preserving NaNs.
    """
    z = np.asarray(z, float)
    mask_nan = np.isnan(z)
    u = norm.cdf(z)
    u[mask_nan] = np.nan
    return u

def uniform_to_gaussian(u):
    """
    Maps u in (0,1) → z∼N(0,1) via Φ⁻¹, preserving NaNs.
    """
    u = np.asarray(u, float)
    mask_nan = np.isnan(u)
    z = norm.ppf(u)
    z[mask_nan] = np.nan
    return z

def inverse_empirical_cdf_integer(u_values, sorted_vals, nan_frac=0.0):
    u = np.asarray(u_values, float)
    n = len(sorted_vals)
    result = np.full(u.shape, np.nan, dtype=float)

    mask_nan = np.isnan(u)
    result[mask_nan] = np.nan

    # TOP slice for NaN: u > 1 - nan_frac
    if nan_frac > 0:
        mask_cat = (~mask_nan) & (u > 1.0 - nan_frac)
        result[mask_cat] = np.nan
    else:
        mask_cat = np.zeros_like(u, bool)

    mask_valid = ~(mask_nan | mask_cat)
    u_valid = u[mask_valid]

    # Rescale [0, 1 - nan_frac) → [0, 1)
    denom = max(1.0 - nan_frac, 1e-12)
    u_adj = u_valid / denom
    u_adj = np.clip(u_adj, 0, 1 - 1e-8)

    knots_u = (np.arange(1, n+1) - 0.5) / n
    x_cont = np.interp(u_adj, knots_u, sorted_vals)

    uniques = np.unique(sorted_vals)
    diffs = np.abs(x_cont[:, None] - uniques[None, :])
    nearest_idx = diffs.argmin(axis=1)
    result[mask_valid] = uniques[nearest_idx]
    return result

def inverse_empirical_cdf_categorical(u_values, sorted_labels, counts, nan_frac=0.0):
    """
    Invert categorical ECDF intervals (Option B).
    If NAN_TOKEN is present in sorted_labels, its interval maps back to np.nan.
    `nan_frac` is ignored here (NaN has its own interval).
    """
    u = np.asarray(u_values, float)
    out = np.full(u.shape, np.nan, dtype=object)

    mask_valid = ~np.isnan(u)
    if not np.any(mask_valid):
        return out

    u_adj = np.clip(u[mask_valid], 0.0, 1.0 - 1e-12)

    counts = np.asarray(counts, float)
    total = counts.sum()
    if total <= 0:
        return out

    p = counts / total
    P = np.cumsum(p)                                   # right edges
    inds = np.searchsorted(P, u_adj, side="right")     # 0..K-1

    labels_arr = np.asarray(sorted_labels, dtype=object)
    chosen = labels_arr[inds].astype(object)
    # map the NAN_TOKEN interval back to true NaNs
    chosen[chosen == "<NaN>"] = np.nan

    out[mask_valid] = chosen
    return out


def inverse_empirical_cdf_decimal_interp(u_values, sorted_vals, nan_frac=0.0):
    u = np.asarray(u_values, float)
    n = len(sorted_vals)
    result = np.full(u.shape, np.nan, dtype=float)

    mask_nan = np.isnan(u)
    result[mask_nan] = np.nan

    # TOP slice for NaN
    if nan_frac > 0:
        mask_cat = (~mask_nan) & (u > 1.0 - nan_frac)
        result[mask_cat] = np.nan
    else:
        mask_cat = np.zeros_like(u, bool)

    mask_valid = ~(mask_nan | mask_cat)
    u_valid = u[mask_valid]

    # Rescale [0, 1 - nan_frac) → [0, 1)
    denom = max(1.0 - nan_frac, 1e-12)
    u_adj = u_valid / denom
    u_adj = np.clip(u_adj, 1e-8, 1 - 1e-8)

    knots_u = (np.arange(1, n+1) - 0.5) / n
    result[mask_valid] = np.interp(u_adj, knots_u, sorted_vals)
    return result

def generate_correlations(df_z_original, x_vector):
    """
    Given:
      - df_z_original: DataFrame of your original z-scores (shape [n, N])
      - x_vector:      array or DataFrame of IID Gaussian samples (shape [m, N])
    Returns:
      - X: DataFrame [m, N] whose columns have the target correlation structure.
    """
    # 1) Empirical Pearson correlation
    targetR = df_z_original.corr(method='pearson')
    
    # 2) Try a straight Cholesky on the raw matrix
    try:
        L = np.linalg.cholesky(targetR.values)
    except np.linalg.LinAlgError:
        # If it fails, do the cleaning + PD-conversion
        # --- FIX for NaNs / zero-variance columns ---
        targetR = targetR.fillna(0.0)
        np.fill_diagonal(targetR.values, 1.0)
        # ---------------------------------------------
        
        # 3) Make it positive-definite via eigen-clipping
        vals, vecs   = np.linalg.eigh(targetR.values)
        eps          = 1e-8
        vals_clipped = np.clip(vals, eps, None)
        R_pd         = (vecs * vals_clipped) @ vecs.T
        
        # 4) Re-enforce unit diagonals
        D    = np.sqrt(np.diag(R_pd))
        R_pd = R_pd / np.outer(D, D)
        
        # 5) Cholesky on the fixed PD matrix
        L = np.linalg.cholesky(R_pd)
    
    # 6) Apply the transform
    X_arr = np.asarray(x_vector) @ L.T
    X     = pd.DataFrame(X_arr, columns=df_z_original.columns)
    return X

def generate_synthetic_data(original_data,n_samples):
    """Generates synthetic data using the custom Gaussian Copula method."""
    print("-> Step 1/4: Transforming training data to Gaussian space...")
    data_train_z = transform_dataset_into_gaussian(original_data)

    print("-> Step 2/4: Generating new independent Gaussian samples...")
    n_cols = len(original_data.columns)
    z_independent = pd.DataFrame(np.random.randn(n_samples, n_cols), columns=original_data.columns)

    print("-> Step 3/4: Applying learned correlations to new samples...")
    z_correlated = generate_correlations(data_train_z, z_independent)

    print("-> Step 4/4: Transforming correlated samples back to original data space...")
    synthetic_data = transform_dataset_from_gaussian(z_correlated, original_data)
    
    return synthetic_data
