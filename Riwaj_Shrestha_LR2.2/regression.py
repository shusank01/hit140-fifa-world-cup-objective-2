"""
FIFA World Cup 2026 - Linear Regression 2.2
Predict the number of goals a team scores in one match (208 rows = 104 matches x 2 teams)
using exactly 8 explanatory variables that are known before the match.

Input:  Data_2.csv
Outputs: Data_2_2_final.csv, model_comparison.csv,
         Figures/fig1_goals_and_correlation.png, Figures/fig2_predicted_vs_actual.png,
         Figures/fig3_residual_diagnostics.png, Figures/fig4_top5_vs_goals.png

"""

import os

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Data_2.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Data_2_2_final.csv")
COMPARISON_FILE = os.path.join(BASE_DIR, "model_comparison.csv")
FIG_DIR = os.path.join(BASE_DIR, "Figures")
SEED = 42

os.makedirs(FIG_DIR, exist_ok=True)

ID_COLS = ["match_id", "date", "team", "opponent"]
TARGET = "goals_for"
FEATURES = [
    "rest_days",
    "stage",
    "avg_age",
    "prev_avg_scored",
    "prev_avg_conceded",
    "opp_prev_avg_scored",
    "opp_prev_avg_conceded",
    "team_top5_players",
]

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")


def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def metrics(y_true, y_pred):
    """Return RMSE, MAE and R-squared using unmodified values."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    r2 = np.nan if denominator == 0 else 1 - np.sum(residuals ** 2) / denominator
    return rmse, mae, r2


# 1. Load and validate the data
heading("1. LOAD AND CHECK THE DATA")
if DATA_FILE.lower().endswith(".csv"):
    raw = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
else:
    raw = pd.read_excel(DATA_FILE)

required_columns = ID_COLS + [TARGET] + FEATURES
missing_columns = [col for col in required_columns if col not in raw.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in {os.path.basename(DATA_FILE)}: {missing_columns}")

# Convert named tournament stages into ordered numeric values, if necessary.
if not pd.api.types.is_numeric_dtype(raw["stage"]):
    stage_num = {
        "Group": 1,
        "R32": 2,
        "R16": 3,
        "Quarter": 4,
        "Semi": 5,
        "Third": 6,
        "Final": 6,
    }
    raw["stage"] = raw["stage"].map(stage_num)

assert len(FEATURES) == 8, "The assignment requires exactly 8 explanatory variables."
df = raw[required_columns].copy()

print("Rows:", len(df))
print(
    "Matches:", df["match_id"].nunique(),
    "| rows per match:", df.groupby("match_id").size().unique(),
)
print("Teams:", df["team"].nunique(), "(a 48-team tournament should contain 48 teams)")
print("Missing values in target/features:", int(df[FEATURES + [TARGET]].isna().sum().sum()))

if len(df) != 208:
    raise ValueError(f"Expected 208 rows, but found {len(df)}.")
if df[FEATURES + [TARGET]].isna().sum().sum() > 0:
    raise ValueError("Missing values found in explanatory variables or target.")
if df["match_id"].nunique() != 104:
    print("WARNING: Expected 104 unique matches; check match_id values.")
if not (df.groupby("match_id").size() == 2).all():
    raise ValueError("Each match_id must have exactly two team rows.")

team_counts = df["team"].value_counts()
print(
    "Teams with fewer than 3 rows (possible spelling/data issue):",
    list(team_counts[team_counts < 3].index),
)

# 2. Descriptive statistics and plots
heading("2. DESCRIPTIVE STATISTICS")
print(df[[TARGET] + FEATURES].describe().T.round(3))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
counts = df[TARGET].value_counts().sort_index()
ax[0].bar(counts.index, counts.values, color="#3b6ea5")
ax[0].set_xlabel("Goals scored by a team in a match")
ax[0].set_ylabel("Number of rows")
ax[0].set_title(f"Distribution of goals_for (mean = {df[TARGET].mean():.2f})")
ax[0].set_xticks(counts.index)

sns.heatmap(
    df[[TARGET] + FEATURES].corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax[1],
    annot_kws={"size": 8},
    cbar=False,
)
ax[1].set_title("Correlation matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_goals_and_correlation.png"), dpi=160, bbox_inches="tight")
plt.close()

# 2b. Top-five-league players vs goals.
# Add jitter ONLY to duplicate coordinate pairs for visibility. The fitted line,
# correlation, p-value, and all model calculations use the original values.
heading("2b. TOP-5 LEAGUE PLAYERS VS GOALS")
xv = df["team_top5_players"].astype(float)
yv = df[TARGET].astype(float)
slope, intercept, r_val, p_val, _ = stats.linregress(xv, yv)
print(
    f"Simple fit: goals = {intercept:.3f} + {slope:.4f} x top5_players | "
    f"r = {r_val:.3f}, R-squared = {r_val ** 2:.3f}, p = {p_val:.2e}"
)

fig, ax = plt.subplots(figsize=(7, 5.2))
plot_data = pd.DataFrame({"x": xv, "y": yv})
# A coordinate is duplicated when multiple rows share the same x and y values.
duplicate_mask = plot_data.duplicated(subset=["x", "y"], keep=False).to_numpy()
plot_x = xv.to_numpy(dtype=float).copy()
plot_y = yv.to_numpy(dtype=float).copy()

if duplicate_mask.any():
    rng = np.random.default_rng(SEED)
    plot_x[duplicate_mask] += rng.normal(0, 0.12, duplicate_mask.sum())
    plot_y[duplicate_mask] += rng.normal(0, 0.045, duplicate_mask.sum())
    point_label = "Team-match observations (jittered duplicates only)"
else:
    point_label = "Team-match observations"

ax.scatter(plot_x, plot_y, alpha=0.55, color="#3b6ea5", label=point_label)
grid = np.linspace(xv.min(), xv.max(), 100)
ax.plot(grid, intercept + slope * grid, color="red", linewidth=2, label="Fitted line")
ax.set_xlabel("Number of the team's players in the top-five European leagues")
ax.set_ylabel("Goals scored by the team in the match")
ax.set_title(
    f"Squad quality vs goals scored\nr = {r_val:.2f}, slope = {slope:.3f} goals per extra player",
    fontsize=10,
)
ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_top5_vs_goals.png"), dpi=160, bbox_inches="tight")
plt.close()

# 3. Train/test split: both team rows from a match remain in the same partition.
heading("3. TRAIN / TEST SPLIT")
X = df[FEATURES]
y = df[TARGET]
groups = df["match_id"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

# 4. Fit ordinary least-squares linear regression on training data.
heading("4. LINEAR REGRESSION MODEL (TRAINING DATA)")
model = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
print(model.summary())

ci = model.conf_int()
coef_table = pd.DataFrame(
    {
        "coefficient": model.params,
        "std_error": model.bse,
        "p_value": model.pvalues,
        "CI_low": ci[0],
        "CI_high": ci[1],
    }
).round(4)
coef_table["significant (5%)"] = np.where(coef_table["p_value"] < 0.05, "yes", "no")
print("\nCoefficient table:")
print(coef_table)

# Standardised coefficients help compare relative coefficient sizes.
X_train_std = X_train.std(ddof=0).replace(0, np.nan)
Z = (X_train - X_train.mean()) / X_train_std
Z = Z.fillna(0)
y_train_std = y_train.std(ddof=0)
if y_train_std > 0:
    zmodel = sm.OLS(
        (y_train - y_train.mean()) / y_train_std,
        sm.add_constant(Z, has_constant="add"),
    ).fit()
    std_beta = zmodel.params.drop("const").round(3)
    print("\nStandardised coefficients (largest absolute effect first):")
    print(std_beta.reindex(std_beta.abs().sort_values(ascending=False).index).to_string())

# 5. Evaluate the test set
heading("5. TEST SET EVALUATION")
pred_test = model.predict(sm.add_constant(X_test, has_constant="add"))
rmse, mae, r2 = metrics(y_test, pred_test)
base_rmse = np.sqrt(np.mean((y_test - y_train.mean()) ** 2))
base_mae = np.mean(np.abs(y_test - y_train.mean()))

print(f"Training R-squared: {model.rsquared:.3f} | Adjusted R-squared: {model.rsquared_adj:.3f}")
print(f"F-test: F = {model.fvalue:.2f}, p = {model.f_pvalue:.2e}")
print(f"Test RMSE: {rmse:.3f} (mean-only baseline: {base_rmse:.3f})")
print(f"Test MAE : {mae:.3f} (mean-only baseline: {base_mae:.3f})")
print(f"Test R-squared: {r2:.3f}")

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(
    pred_test,
    y_test,
    alpha=0.65,
    color="#3b6ea5",
    edgecolors="black",
    linewidths=0.35,
)
low = min(-0.2, float(np.min(pred_test)) - 0.1, float(np.min(y_test)) - 0.1)
high = max(float(np.max(pred_test)), float(np.max(y_test))) + 0.3
ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_xlabel("Predicted goals")
ax.set_ylabel("Actual goals")
ax.set_title(
    f"Test set: predicted vs actual (n = {len(y_test)})\n"
    f"RMSE = {rmse:.2f} (baseline {base_rmse:.2f}), MAE = {mae:.2f}, R² = {r2:.2f}",
    fontsize=10,
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_predicted_vs_actual.png"), dpi=160, bbox_inches="tight")
plt.close()

# 6. Five-fold cross-validation for ordinary linear regression.
heading("6. FIVE-FOLD CROSS-VALIDATION")
cv_rows = []
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=groups), start=1):
    lr = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
    predictions = lr.predict(X.iloc[te])
    fold_rmse, fold_mae, fold_r2 = metrics(y.iloc[te], predictions)
    baseline = np.sqrt(np.mean((y.iloc[te] - y.iloc[tr].mean()) ** 2))
    cv_rows.append(
        {
            "fold": fold,
            "RMSE": fold_rmse,
            "MAE": fold_mae,
            "R-squared": fold_r2,
            "Baseline RMSE": baseline,
        }
    )
cv = pd.DataFrame(cv_rows).set_index("fold")
print(cv.round(3))
print("\nMean fold metrics:")
print(cv.mean().round(3))

# 6b. Out-of-fold predictions for all rows. The model predicting a row did not
# train on either team row from that row's match.
heading(f"6b. OUT-OF-FOLD PREDICTIONS (saved in {os.path.basename(OUTPUT_FILE)})")
oof_pred = cross_val_predict(
    LinearRegression(),
    X,
    y,
    groups=groups,
    cv=GroupKFold(n_splits=5),
)
out = df.copy()

# Evaluate the original, unrounded model predictions before any post-processing.
oof_rmse, oof_mae, oof_r2 = metrics(y, oof_pred)
print(
    f"Out-of-fold metrics (original predictions): RMSE = {oof_rmse:.3f}, "
    f"MAE = {oof_mae:.3f}, R-squared = {oof_r2:.3f}"
)

# Clip negative values and round only for the saved CSV. These adjusted values
# are for the output file, not for calculating model-performance metrics.
out["predicted_goals"] = np.clip(oof_pred, 0, None).round(2)
out.to_csv(OUTPUT_FILE, index=False)
print(out[["match_id", "team", "opponent", TARGET, "predicted_goals"]].head(10).to_string(index=False))
print(
    f"\nSaved {os.path.basename(OUTPUT_FILE)} with {out.shape[1]} columns "
    f"({len(ID_COLS)} ID + 1 target + {len(FEATURES)} explanatory + predicted_goals)."
)

# 6c. Compare competing models using the same grouped folds.
heading("6c. MODEL COMPARISON (FIVE-FOLD CV, GROUPED BY MATCH)")
cv_split = GroupKFold(n_splits=5)


def cv_scores(estimator, cols=None):
    """Return mean CV RMSE, fold RMSE standard deviation, MAE and R-squared."""
    if cols is None:
        cols = FEATURES
    result = cross_validate(
        estimator,
        df[cols],
        y,
        groups=groups,
        cv=cv_split,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
    )
    return {
        "RMSE": -result["test_rmse"].mean(),
        "RMSE std": result["test_rmse"].std(),
        "MAE": -result["test_mae"].mean(),
        "R-squared": result["test_r2"].mean(),
    }

grids = {
    "Ridge": (Ridge, [0.1, 1, 10, 30, 100]),
    "Lasso": (Lasso, [0.005, 0.01, 0.03, 0.1]),
    "Poisson": (PoissonRegressor, [0.001, 0.01, 0.1, 1]),
}
best_alpha = {}
for name, (estimator_class, alphas) in grids.items():
    sweep = {}
    for alpha in alphas:
        estimator = make_pipeline(StandardScaler(), estimator_class(alpha=alpha))
        sweep[alpha] = cv_scores(estimator)["RMSE"]
    best_alpha[name] = min(sweep, key=sweep.get)
    print(
        f"{name:8s} CV RMSE by alpha: "
        + ", ".join(f"{alpha}: {score:.3f}" for alpha, score in sweep.items())
        + f" -> best alpha = {best_alpha[name]}"
    )

candidates = {
    "Baseline (always the mean)": DummyRegressor(strategy="mean"),
    "OLS, top-5 players only": None,
    "OLS, all 8 variables": LinearRegression(),
    f"Ridge (alpha={best_alpha['Ridge']})": make_pipeline(
        StandardScaler(), Ridge(alpha=best_alpha["Ridge"])
    ),
    f"Lasso (alpha={best_alpha['Lasso']})": make_pipeline(
        StandardScaler(), Lasso(alpha=best_alpha["Lasso"])
    ),
    f"Poisson (alpha={best_alpha['Poisson']})": make_pipeline(
        StandardScaler(), PoissonRegressor(alpha=best_alpha["Poisson"], max_iter=1000)
    ),
    "Random forest (depth 3)": RandomForestRegressor(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=10,
        random_state=SEED,
    ),
}

comparison_results = {}
for name, estimator in candidates.items():
    if estimator is None:
        comparison_results[name] = cv_scores(
            LinearRegression(), cols=["team_top5_players"]
        )
    else:
        comparison_results[name] = cv_scores(estimator)

comparison = pd.DataFrame(comparison_results).T.round(3)
baseline_rmse_cv = comparison.loc["Baseline (always the mean)", "RMSE"]
comparison["RMSE vs baseline (%)"] = (
    (comparison["RMSE"] / baseline_rmse_cv - 1) * 100
).round(1)
print("\n" + comparison.to_string())
comparison.to_csv(COMPARISON_FILE, index_label="Model")
best_model = comparison["RMSE"].idxmin()
print(f"\nLowest cross-validated RMSE: {best_model}")

# 7. Linear regression assumption diagnostics on the training residuals.
heading("7. ASSUMPTION CHECKS")
Xc = sm.add_constant(X_train, has_constant="add")
vif_values = [
    variance_inflation_factor(Xc.values, i + 1)
    for i in range(len(FEATURES))
]
vif = pd.DataFrame({"variable": FEATURES, "VIF": vif_values}).round(2)
print("VIF (values above about 5 can be a warning; investigate context):")
print(vif.to_string(index=False))

resid = model.resid
bp_stat, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(X_train, has_constant="add"))
jb_stat, jb_p = stats.jarque_bera(resid)
dw = durbin_watson(resid)
print(
    f"\nBreusch-Pagan (heteroscedasticity): statistic = {bp_stat:.3f}, p = {bp_p:.3f} -> "
    + ("evidence of uneven residual variance" if bp_p < 0.05 else "no strong evidence of uneven residual variance")
)
print(
    f"Jarque-Bera (residual normality): statistic = {jb_stat:.3f}, p = {jb_p:.3f} -> "
    + ("residuals depart from normality" if jb_p < 0.05 else "no strong evidence against normality")
)
print(
    f"Durbin-Watson statistic: {dw:.3f}. Interpret cautiously because rows are grouped by match."
)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].scatter(model.fittedvalues, resid, alpha=0.6, color="#3b6ea5")
ax[0].axhline(0, color="red", linestyle="--")
ax[0].set_xlabel("Fitted values")
ax[0].set_ylabel("Residuals")
ax[0].set_title("Residuals vs fitted")
stats.probplot(resid, dist="norm", plot=ax[1])
ax[1].set_title("Normal Q-Q plot")
ax[2].hist(resid, bins=15, color="#3b6ea5", edgecolor="white")
ax[2].set_xlabel("Residuals")
ax[2].set_ylabel("Frequency")
ax[2].set_title("Residual histogram")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_residual_diagnostics.png"), dpi=160, bbox_inches="tight")
plt.close()

# 8. Optional full-data coefficient robustness summary. 
heading("8. ROBUSTNESS CHECK (ALL 208 ROWS; DESCRIPTIVE ONLY)")
Xf = sm.add_constant(X, has_constant="add")
full = sm.OLS(y, Xf).fit()
full_cluster_robust = sm.OLS(y, Xf).fit(
    cov_type="cluster", cov_kwds={"groups": df["match_id"]}
)
robust = pd.DataFrame(
    {
        "coef (all data)": full.params,
        "p (training model)": model.pvalues,
        "p (all data, OLS)": full.pvalues,
        "p (all data, cluster-robust)": full_cluster_robust.pvalues,
    }
).round(4).drop(index="const", errors="ignore")
print(
    f"All-data model: R-squared = {full.rsquared:.3f}, "
    f"adjusted R-squared = {full.rsquared_adj:.3f}"
)
print(robust)

heading(
    "DONE - saved: "
    + os.path.basename(OUTPUT_FILE)
    + ", model_comparison.csv, Figures/fig1_goals_and_correlation.png, "
    + "Figures/fig2_predicted_vs_actual.png, Figures/fig3_residual_diagnostics.png, "
    + "Figures/fig4_top5_vs_goals.png"
)