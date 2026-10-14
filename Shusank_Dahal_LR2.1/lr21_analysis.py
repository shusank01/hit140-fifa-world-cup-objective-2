"""Linear Regression 2.1 - predicting goal difference in FIFA World Cup 2026 matches."""
import warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm, statsmodels.formula.api as smf
import os
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, HuberRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

#  0. Load data
df = pd.read_csv('final_world_cup_dataset.csv')
FEATURES = ['knockout', 'qualified_2022_diff', 'hosting_difference', 'avg_age_difference',
            'fifa_point_difference', 'scored_avg_diff', 'conceded_avg_diff', 'points_before_diff']
assert not df[FEATURES].isna().any().any(), 'dataset has missing values'
y = df.goal_diff.values
df['f'] = df.fifa_point_difference / 100          # FIFA-points gap in units of 100 points

#  1. Full OLS (all 8 variables)
Xc = sm.add_constant(df[FEATURES])
full = sm.OLS(df.goal_diff, Xc).fit()
print(full.summary())
print('VIF:', {c: round(variance_inflation_factor(Xc.values, i), 2) for i, c in enumerate(Xc.columns) if c != 'const'})

#  2. Simple model: FIFA gap only
final = smf.ols('goal_diff ~ f', df).fit()
print(final.summary().tables[1])
print('adj R2 %.3f | AIC %.1f vs %.1f | BIC %.1f vs %.1f' % (final.rsquared_adj, final.aic, full.aic, final.bic, full.bic))
print('Do the other 7 variables add anything? (partial F, p):', full.compare_f_test(final)[:2])
print('Robust (HC3) p-value of slope:', smf.ols('goal_diff ~ f', df).fit(cov_type='HC3').pvalues['f'])
print('Slope if forced through origin:', smf.ols('goal_diff ~ f - 1', df).fit().params['f'])
print('Shapiro p of residuals:', stats.shapiro(final.resid).pvalue)
rng = np.random.default_rng(0)                    # bootstrap 95% CI for the slope
boot = [np.polyfit(s.f, s.goal_diff, 1)[0]
        for s in (df.sample(len(df), replace=True, random_state=int(rng.integers(1e9))) for _ in range(5000))]
print('Bootstrap 95% CI of slope:', np.percentile(boot, [2.5, 97.5]))
df['pred'] = final.predict(df)
df['res'] = df.goal_diff - df.pred

#  3. Compare 12 models with cross-validation
ALL, FIFA = FEATURES, ['fifa_point_difference']
models = {   # name: (columns used, estimator)
    'Mean-only baseline':           (ALL,  DummyRegressor()),
    'OLS - all 8':                  (ALL,  LinearRegression()),
    'OLS - FIFA pts only':          (FIFA, LinearRegression()),
    'OLS - FIFA+hosting+knockout':  (FIFA + ['hosting_difference', 'knockout'], LinearRegression()),
    'Ridge':                        (ALL,  RidgeCV(alphas=np.logspace(-2, 2, 50))),
    'Lasso':                        (ALL,  LassoCV(cv=5, max_iter=20000)),
    'Elastic Net':                  (ALL,  ElasticNetCV(cv=5, l1_ratio=[.2, .5, .8], max_iter=20000)),
    'Huber (robust)':               (ALL,  HuberRegressor(max_iter=1000)),
    'SVR (RBF)':                    (ALL,  SVR(C=2, epsilon=.3)),
    'k-NN (k=15)':                  (ALL,  KNeighborsRegressor(15)),
    'Random Forest':                (ALL,  RandomForestRegressor(300, min_samples_leaf=5, random_state=0)),
    'Gradient Boosting':            (ALL,  GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.05,
                                                                     subsample=.7, random_state=0)),
}
rmse = lambda a, b: mean_squared_error(a, b) ** .5
folds = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=42).split(df))   # 5-fold CV x 10 repeats
is_group = (df['round'] == 'Group stage').values
non_draw = y != 0
rows, fold_rmse = [], {}
for name, (cols, est) in models.items():
    X, m = df[cols], make_pipeline(StandardScaler(), est)
    fold_rmse[name] = np.array([rmse(y[te], m.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])) for tr, te in folds])
    ko_rmse = rmse(y[~is_group], m.fit(X[is_group], y[is_group]).predict(X[~is_group]))     # train group -> test knockout
    cv_pred = cross_val_predict(m, X, y, cv=10)
    rows.append(dict(model=name, cv_rmse=fold_rmse[name].mean(), cv_sd=fold_rmse[name].std(), ko_holdout_rmse=ko_rmse,
                     winner_acc=(np.sign(cv_pred[non_draw]) == np.sign(y[non_draw])).mean()))
res = pd.DataFrame(rows)

# Corrected resampled t-test (Nadeau-Bengio) of every model against OLS (all 8)
base, k = fold_rmse['OLS - all 8'], len(folds)
def p_vs_base(r):
    d = r - base
    t = d.mean() / np.sqrt((1 / k + 0.25) * d.var(ddof=1))       # 0.25 = n_test / n_train for 5-fold CV
    return 2 * stats.t.sf(abs(t), k - 1)
res['p_vs_ols_all8'] = [np.nan if n == 'OLS - all 8' else p_vs_base(fold_rmse[n]) for n in res.model]
res.round(3).to_csv('model_comparison.csv', index=False)
print(res.round(3))

#  4. Figures
os.makedirs("figures", exist_ok=True)
NAVY, TEAL, ORANGE, GREY, RED = '#1F2A44', '#2A9D8F', '#E76F51', '#8D99AE', '#C0392B'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 10})
save = lambda name: (plt.tight_layout(), plt.savefig(f'figures/{name}.png', dpi=300), plt.close())

# Figure 1: scatter + fitted line, and effect of each variable (per +1 standard deviation)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1.35, 1]})
xs = np.linspace(df.f.min() - .2, df.f.max() + .2, 100)
band = final.get_prediction(sm.add_constant(pd.DataFrame({'f': xs}))).summary_frame(alpha=.05)
a.fill_between(xs, band.obs_ci_lower, band.obs_ci_upper, color=GREY, alpha=.15, label='95% prediction band')
a.fill_between(xs, band.mean_ci_lower, band.mean_ci_upper, color=TEAL, alpha=.35, label='95% CI of mean')
a.plot(xs, band['mean'], color=NAVY, lw=2)
for ko, seed, colour, marker, size, label in [(0, 1, TEAL, 'o', 22, 'Group stage'), (1, 2, ORANGE, 'D', 26, 'Knockout')]:
    s = df[df.knockout == ko]
    a.scatter(s.f, s.goal_diff + np.random.default_rng(seed).uniform(-.12, .12, len(s)), s=size, c=colour,
              edgecolor='white', lw=.4, marker=marker, label=label)
for i in df.res.abs().nlargest(5).index:          # label the 5 biggest misses
    a.annotate(f"{df.team_a[i]} v {df.team_b[i]}", (df.f[i], df.goal_diff[i]), fontsize=7, color=RED,
               xytext=((-45, 8) if df.f[i] > 3.5 else (6, 4 if df.goal_diff[i] > 0 else -10)), textcoords='offset points')
a.axhline(0, color='k', lw=.5); a.axvline(0, color='k', lw=.5)
a.set(xlabel='FIFA-points gap, team A minus team B (per 100 pts)', ylabel='Goal difference (A - B)')
a.set_title('Every +100 FIFA points is worth ≈ 0.57 goals', loc='left'); a.legend(frameon=False, fontsize=7, loc='lower right')

z = sm.OLS(df.goal_diff, sm.add_constant((df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std())).fit()
coef, ci = z.params.drop('const'), z.conf_int().drop('const')
names = {'fifa_point_difference': 'FIFA points gap', 'knockout': 'Knockout stage', 'conceded_avg_diff': 'Goals conceded avg gap',
         'hosting_difference': 'Host advantage', 'scored_avg_diff': 'Goals scored avg gap', 'qualified_2022_diff': 'Qualified 2022 gap',
         'avg_age_difference': 'Squad age gap', 'points_before_diff': 'Tournament points gap'}
order = coef.abs().sort_values().index
for j, c in enumerate(order):
    colour = TEAL if (ci.loc[c, 0] > 0 or ci.loc[c, 1] < 0) else GREY        # teal = CI excludes zero
    b.plot([ci.loc[c, 0], ci.loc[c, 1]], [j, j], color=colour, lw=2); b.scatter(coef[c], j, color=colour, s=35, zorder=3)
b.axvline(0, color='k', lw=.6); b.set_yticks(range(len(order))); b.set_yticklabels([names[c] for c in order])
b.set(xlabel='Effect on goal difference per +1 SD (95% CI)'); b.set_title('Only 1 of 8 predictors is non-zero', loc='left')
save('fig1_story')

# Figure 2: model comparison (CV error and train-on-group/test-on-knockout error)
r2 = res.sort_values('cv_rmse', ascending=False).reset_index(drop=True)
LINEAR = ('OLS - all 8', 'Ridge', 'Lasso', 'Elastic Net', 'Huber (robust)', 'OLS - FIFA+hosting+knockout')
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for j, r in r2.iterrows():
    colour = NAVY if r.model == 'OLS - FIFA pts only' else GREY if 'baseline' in r.model else TEAL if r.model in LINEAR else ORANGE
    ax[0].errorbar(r.cv_rmse, j, xerr=r.cv_sd / np.sqrt(k) * 1.96, fmt='o', color=colour, capsize=2)
    ax[1].barh(j, r.ko_holdout_rmse, color=colour, height=.6); ax[1].text(r.ko_holdout_rmse + .01, j, f'{r.ko_holdout_rmse:.2f}', va='center', fontsize=7)
for colour, label in [(NAVY, 'Chosen: 1-variable OLS'), (TEAL, 'Other linear models'), (ORANGE, 'Non-linear models'), (GREY, 'Baseline')]:
    ax[0].scatter([], [], color=colour, label=label)
ax[0].legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 0.08))
ax[0].set_yticks(range(len(r2))); ax[0].set_yticklabels(r2.model); ax[1].set_xlim(1.0, 1.75)
ax[0].set(xlabel='Repeated 5-fold CV RMSE (goals; lower = better)'); ax[1].set(xlabel='Train on group stage → test on 32 knockout games (RMSE)')
ax[0].set_title('Extra complexity buys nothing', loc='left'); ax[1].set_title('...and it holds up on unseen knockout games', loc='left')
save('fig2_models')

# Figure 3: residual checks + the eight biggest surprises
fig, ax = plt.subplots(1, 3, figsize=(10, 3.5), gridspec_kw={'width_ratios': [1, 1, 1.35]})
ax[0].scatter(final.fittedvalues, final.resid, s=14, c=TEAL, edgecolor='white', lw=.3); ax[0].axhline(0, color='k', lw=.6)
ax[0].set(xlabel='Fitted goal diff', ylabel='Residual'); ax[0].set_title('Residuals: no pattern', loc='left')
(q, r), (slope, icpt, _) = stats.probplot(final.resid, dist='norm')
ax[1].scatter(q, r, s=14, c=TEAL, edgecolor='white', lw=.3); ax[1].plot(q, slope * np.array(q) + icpt, color=NAVY)
ax[1].set(xlabel='Normal quantiles', ylabel='Residual quantiles'); ax[1].set_title(f'Normal (Shapiro p={stats.shapiro(final.resid).pvalue:.2f})', loc='left')
top = df.reindex(df.res.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
for j, (_, r) in enumerate(top.iterrows()):
    ax[2].plot([r.pred, r.goal_diff], [j, j], color=GREY, lw=1.5)
    ax[2].scatter(r.pred, j, color=NAVY, s=28, zorder=3); ax[2].scatter(r.goal_diff, j, color=ORANGE, s=34, zorder=3)
ax[2].set_yticks(range(len(top))); ax[2].set_yticklabels([f'{r.team_a} v {r.team_b}' for _, r in top.iterrows()], fontsize=7)
ax[2].axvline(0, color='k', lw=.5); ax[2].set(xlabel='Goal difference'); ax[2].set_title('Biggest surprises', loc='left')
ax[2].scatter([], [], color=NAVY, label='Predicted'); ax[2].scatter([], [], color=ORANGE, label='Actual'); ax[2].legend(frameon=False, fontsize=7, loc='lower right')
save('fig3_diagnostics')
