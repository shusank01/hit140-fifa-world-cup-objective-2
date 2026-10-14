"""
FIFA World Cup 2026 - Linear Regression 2.2
Predict the number of goals a team scores in one match (208 rows = 104 matches x 2 teams)
using exactly 8 explanatory variables that are known before the match.

Output: Data_2_2_final.csv now also contains predicted_goals for every row.
"""

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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.linear_model import LinearRegression


DATA_FILE = "Data_2.csv"
OUTPUT_FILE = "Data_2_2_final.csv"
SEED = 42

ID_COLS = ["match_id", "date", "team", "opponent"]
TARGET = "goals_for"
FEATURES = ["rest_days", "stage", "avg_age",
            "prev_avg_scored", "prev_avg_conceded",
            "opp_prev_avg_scored", "opp_prev_avg_conceded",
            "team_top5_players"]

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")


def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# 1. Load the data

heading("1. LOAD AND CHECK THE DATA")
if DATA_FILE.lower().endswith(".csv"):
    raw = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
else:
    raw = pd.read_excel(DATA_FILE)


if not pd.api.types.is_numeric_dtype(raw["stage"]):
    stage_num = {"Group": 1, "R32": 2, "R16": 3, "Quarter": 4,
                 "Semi": 5, "Third": 6, "Final": 6}
    raw["stage"] = raw["stage"].map(stage_num)

assert len(FEATURES) == 8, "The assignment needs exactly 8 explanatory variables"
df = raw[ID_COLS + [TARGET] + FEATURES].copy()

print("Rows:", len(df))
print("Matches:", df["match_id"].nunique(), "| rows per match:", df.groupby("match_id").size().unique())
print("Teams:", df["team"].nunique(), "(a World Cup with 48 teams should show 48)")
print("Missing values:", int(df[FEATURES + [TARGET]].isna().sum().sum()))
assert len(df) == 208 and df[FEATURES + [TARGET]].isna().sum().sum() == 0

# Spelling check: each team should appear about 3 to 8 times
team_counts = df["team"].value_counts()
print("Teams with fewer than 3 rows (possible spelling problem):",
      list(team_counts[team_counts < 3].index))


# 2. Descriptive statistics and plots

heading("2. DESCRIPTIVE STATISTICS")
print(df[[TARGET] + FEATURES].describe().T.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
counts = df[TARGET].value_counts().sort_index()
ax[0].bar(counts.index, counts.values, color="#3b6ea5")
ax[0].set_xlabel("Goals scored by a team in a match")
ax[0].set_ylabel("Number of rows")
ax[0].set_title(f"Distribution of goals_for (mean = {df[TARGET].mean():.2f})")
sns.heatmap(df[[TARGET] + FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax[1], annot_kws={"size": 7}, cbar=False)
ax[1].set_title("Correlation matrix")
plt.tight_layout()
plt.savefig("fig1_goals_and_correlation.png", dpi=150)
plt.close()


# 3. Train / test split (both rows of a match stay together)

heading("3. TRAIN / TEST SPLIT")
X = df[FEATURES]
y = df[TARGET]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=df["match_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")


# 4. Fit the linear regression model

heading("4. LINEAR REGRESSION MODEL (training data)")
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(model.summary())

ci = model.conf_int()
coef_table = pd.DataFrame({"coefficient": model.params, "std_error": model.bse,
                           "p_value": model.pvalues,
                           "CI_low": ci[0], "CI_high": ci[1]}).round(4)
coef_table["significant (5%)"] = np.where(coef_table["p_value"] < 0.05, "yes", "no")
print("\nCoefficient table:")
print(coef_table)


Z = (X_train - X_train.mean()) / X_train.std()
zmodel = sm.OLS((y_train - y_train.mean()) / y_train.std(), sm.add_constant(Z)).fit()
std_beta = zmodel.params.drop("const").round(3)
print("\nStandardised coefficients (largest effect first):")
print(std_beta.reindex(std_beta.abs().sort_values(ascending=False).index).to_string())


# 5. Evaluate on the test data

heading("5. TEST SET EVALUATION")


def metrics(y_true, y_pred):
    resid = y_true - y_pred
    rmse = np.sqrt((resid ** 2).mean())
    mae = resid.abs().mean()
    r2 = 1 - (resid ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    return rmse, mae, r2


pred_test = model.predict(sm.add_constant(X_test, has_constant="add"))
rmse, mae, r2 = metrics(y_test, pred_test)
base_rmse = np.sqrt(((y_test - y_train.mean()) ** 2).mean())
base_mae = (y_test - y_train.mean()).abs().mean()

print(f"Training R-squared: {model.rsquared:.3f} | Adjusted R-squared: {model.rsquared_adj:.3f}")
print(f"F-test: F = {model.fvalue:.2f}, p = {model.f_pvalue:.2e}")
print(f"Test RMSE: {rmse:.3f}  (baseline of always guessing the mean: {base_rmse:.3f})")
print(f"Test MAE : {mae:.3f}  (baseline: {base_mae:.3f})")
print(f"Test R2  : {r2:.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
jitter = np.random.default_rng(SEED).normal(0, 0.06, len(y_test))
ax.scatter(pred_test, y_test + jitter, alpha=0.7, color="#3b6ea5")
lims = [min(pred_test.min(), 0) - 0.2, max(pred_test.max(), y_test.max()) + 0.2]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Predicted goals")
ax.set_ylabel("Actual goals (small jitter added)")
ax.set_title("Test set: predicted vs actual")
ax.legend()
plt.tight_layout()
plt.savefig("fig2_predicted_vs_actual.png", dpi=150)
plt.close()


# 6. Five-fold cross-validation

heading("6. 5-FOLD CROSS-VALIDATION")
rows = []
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=df["match_id"]), start=1):
    lr = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
    p = pd.Series(lr.predict(X.iloc[te]), index=y.iloc[te].index)
    rm, ma, r2k = metrics(y.iloc[te], p)
    base = np.sqrt(((y.iloc[te] - y.iloc[tr].mean()) ** 2).mean())
    rows.append({"fold": k, "RMSE": rm, "MAE": ma, "R-squared": r2k, "Baseline RMSE": base})
cv = pd.DataFrame(rows).set_index("fold").round(3)
cv.loc["mean"] = cv.mean().round(3)
print(cv)


# 6b. Predicted goals for every row, saved in the final CSV

heading("6b. PREDICTED GOALS FOR ALL 208 ROWS (saved in " + OUTPUT_FILE + ")")
# Out-of-fold predictions: each row is predicted by a model that never saw its match
# (both rows of a match stay in the same fold), so the predictions are honest.
oof_pred = cross_val_predict(LinearRegression(), X, y,
                             groups=df["match_id"],
                             cv=GroupKFold(n_splits=5))

out = df.copy()                                                # df and X stay unchanged
out["predicted_goals"] = np.clip(oof_pred, 0, None).round(2)   # goals cannot be below 0

out.to_csv(OUTPUT_FILE, index=False)
print(out[["match_id", "team", "opponent", TARGET, "predicted_goals"]].head(10).to_string(index=False))
print(f"\nSaved {OUTPUT_FILE} with {out.shape[1]} columns "
      f"({len(ID_COLS)} ID + 1 target + {len(FEATURES)} explanatory + predicted_goals)")
o_rmse, o_mae, o_r2 = metrics(out[TARGET], out["predicted_goals"])
print(f"Out-of-fold accuracy on all 208 rows: RMSE = {o_rmse:.3f}, MAE = {o_mae:.3f}, R2 = {o_r2:.3f}")


# 7. Check the model assumptions

heading("7. ASSUMPTION CHECKS")
Xc = sm.add_constant(X_train)
vif = pd.DataFrame({"variable": FEATURES,
                    "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(FEATURES))]}).round(2)
print("VIF (above about 5 is a warning):")
print(vif.to_string(index=False))

resid = model.resid
bp_stat, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(X_train))
jb_stat, jb_p = stats.jarque_bera(resid)
dw = durbin_watson(resid)
print(f"\nBreusch-Pagan (equal spread):   statistic = {bp_stat:.3f}, p = {bp_p:.3f} -> "
      + ("spread is uneven" if bp_p < 0.05 else "spread looks fine"))
print(f"Jarque-Bera (normal residuals): statistic = {jb_stat:.3f}, p = {jb_p:.3f} -> "
      + ("residuals are skewed" if jb_p < 0.05 else "residuals look normal"))
print(f"Durbin-Watson (independence):   {dw:.3f} -> "
      + ("no sign of autocorrelation" if 1.5 < dw < 2.5 else "check autocorrelation"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].scatter(model.fittedvalues, resid, alpha=0.6, color="#3b6ea5")
ax[0].axhline(0, color="red", linestyle="--")
ax[0].set_xlabel("Fitted values")
ax[0].set_ylabel("Residuals")
ax[0].set_title("Residuals vs fitted")
stats.probplot(resid, dist="norm", plot=ax[1])
ax[1].set_title("Normal Q-Q plot")
ax[2].hist(resid, bins=15, color="#3b6ea5", edgecolor="white")
ax[2].set_xlabel("Residuals")
ax[2].set_title("Residual histogram")
plt.tight_layout()
plt.savefig("fig3_residual_diagnostics.png", dpi=150)
plt.close()


# 8. Robustness check on all 208 rows

heading("8. ROBUSTNESS CHECK (all 208 rows)")
Xf = sm.add_constant(X)
full = sm.OLS(y, Xf).fit()
full_cl = sm.OLS(y, Xf).fit(cov_type="cluster", cov_kwds={"groups": df["match_id"]})
robust = pd.DataFrame({"coef (all data)": full.params,
                       "p (training model)": model.pvalues,
                       "p (all data, OLS)": full.pvalues,
                       "p (all data, cluster-robust)": full_cl.pvalues}).round(4).drop("const")
print(f"All-data model: R-squared = {full.rsquared:.3f}, adjusted R-squared = {full.rsquared_adj:.3f}")
print(robust)

heading("DONE - saved: " + OUTPUT_FILE + ", fig1_goals_and_correlation.png, "
        "fig2_predicted_vs_actual.png, fig3_residual_diagnostics.png")