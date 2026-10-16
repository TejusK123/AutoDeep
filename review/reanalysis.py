"""Reviewer spot-check of AutoDeep's bundled training data (commit 133bf14).
Usage: python3 -I reanalysis.py <training_data_dir> <miRNA_model.json>
Compares the paper's protocol (repeated random stratified 80/20 splits) with
leave-one-species-out (LOSO) evaluation, simple baselines and a feature ablation."""
import sys, glob, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
warnings.filterwarnings('ignore')
d, model_json = sys.argv[1], sys.argv[2]
SPECIES = {'Andreii':'E. andrei','Arabidopsis_fp_candidate':'A. thaliana','Arabidopsis_mirgenedb':'A. thaliana',
           'Bemisia':'B. tabaci','Drosophila_fp_candidate':'D. melanogaster','Drosophila_mirgenedb':'D. melanogaster',
           'FAW':'S. frugiperda','Human_fp_candidate':'H. sapiens','Human_mirgenedb':'H. sapiens',
           'Pomonella':'C. pomonella','Xylostella':'P. xylostella'}
fr = []
for f in sorted(glob.glob(os.path.join(d, '*.csv'))):
    df = pd.read_csv(f); k = os.path.basename(f)[len('original_data_'):-4]
    df['src'] = k; df['species'] = SPECIES[k]; fr.append(df)
a = pd.concat(fr, ignore_index=True)
LAB = 'Candidate/Confident/no-mirna'; CLS = ['Candidate', 'Confident', 'falsepositive']
y = a[LAB].map({c: i for i, c in enumerate(CLS)}).to_numpy()
FEAT = [c for c in a.columns if c not in ('provisional_id', LAB, 'src', 'species')]
X = a[FEAT].copy()
X['significant_randfold_p-value'] = X['significant_randfold_p-value'].map({'yes': 1, 'no': 0})
X['homologous_seed_in_miRBase'] = X['homologous_seed_in_miRBase'].map({True: 1, False: 0, 'True': 1, 'False': 0})
X["mature_5'u_or_3'u"] = X["mature_5'u_or_3'u"].map({True: 1, False: 0, 'True': 1, 'False': 0})
X = X.astype(float)
DIRECT = ['miRDeep2_score', 'mirDeep_sensitivity', 'sensitivity_uncertainty', 'total_read_count', 'mature_read_count',
          'loop_read_count', 'star_read_count', 'significant_randfold_p-value', 'homologous_seed_in_miRBase']
ENG = [c for c in FEAT if c not in DIRECT]
COUNTS = ['miRDeep2_score', 'total_read_count', 'mature_read_count', 'loop_read_count', 'star_read_count', 'num_locis']
homolog = X['homologous_seed_in_miRBase'].to_numpy() == 1
groups = a['species'].to_numpy()
print(f'n={len(a)}; features={len(FEAT)} (direct {len(DIRECT)}, engineered {len(ENG)}); class counts', dict(zip(CLS, np.bincount(y))))

def slog(v): return np.sign(v) * np.log1p(np.abs(v))
def make(name):
    if name == 'majority':
        return ('maj', None)
    if name == 'seed+score tree':
        return (['homologous_seed_in_miRBase', 'miRDeep2_score'], DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=0))
    if name == 'logistic (18 feat)':
        def tf(Z):
            Z = Z.copy()
            for c in COUNTS: Z[c] = slog(Z[c])
            return Z
        return (FEAT, Pipeline([('log', FunctionTransformer(tf)), ('imp', SimpleImputer(strategy='median', add_indicator=True)),
                                ('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=5000, class_weight='balanced', C=1.0))]))
    xp = dict(n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.9, min_child_weight=2,
              objective='multi:softprob', tree_method='hist', n_jobs=4, random_state=0, verbosity=0)
    if name == 'XGB direct (9)': return (DIRECT, xgb.XGBClassifier(**xp))
    if name == 'XGB all (18)': return (FEAT, xgb.XGBClassifier(**xp))
    if name == 'XGB no seed/sens (15)':
        return ([c for c in FEAT if c not in ('homologous_seed_in_miRBase', 'mirDeep_sensitivity', 'sensitivity_uncertainty')], xgb.XGBClassifier(**xp))
    raise ValueError(name)

def fit_predict(name, tr, te):
    cols, mdl = make(name)
    if cols == 'maj':
        return np.full(len(te), np.bincount(y[tr]).argmax())
    sw = compute_sample_weight('balanced', y[tr])
    if isinstance(mdl, xgb.XGBClassifier): mdl.fit(X.iloc[tr][cols], y[tr], sample_weight=sw)
    else: mdl.fit(X.iloc[tr][cols], y[tr])
    return mdl.predict(X.iloc[te][cols])

def metrics(yt, yp):
    out = dict(acc=accuracy_score(yt, yp), macroF1=f1_score(yt, yp, average='macro', labels=[0,1,2], zero_division=0),
               balAcc=balanced_accuracy_score(yt, yp))
    r = recall_score(yt, yp, labels=[0,1,2], average=None, zero_division=0)
    p = precision_score(yt, yp, labels=[0,1,2], average=None, zero_division=0)
    out.update({'rec_Cand': r[0], 'rec_Conf': r[1], 'rec_FP': r[2], 'prec_Conf': p[1]})
    return out

MODELS = ['majority', 'seed+score tree', 'logistic (18 feat)', 'XGB direct (9)', 'XGB all (18)', 'XGB no seed/sens (15)']
rows = []
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
splits = list(sss.split(X, y))
for m in MODELS:
    allm, nh = [], []
    for tr, te in splits:
        yp = fit_predict(m, tr, te)
        allm.append(metrics(y[te], yp))
        mask = ~homolog[te]
        nh.append(metrics(y[te][mask], yp[mask]))
    A = pd.DataFrame(allm).agg(['mean', 'std']); N = pd.DataFrame(nh).mean()
    rows.append(dict(protocol='random 80/20 x10', model=m, acc=f"{A.loc['mean','acc']:.3f} ± {A.loc['std','acc']:.3f}",
                     macroF1=round(A.loc['mean','macroF1'], 3), balAcc=round(A.loc['mean','balAcc'], 3),
                     nonhom_acc=round(N['acc'], 3), nonhom_macroF1=round(N['macroF1'], 3), nonhom_rec_Conf=round(N['rec_Conf'], 3),
                     nonhom_prec_Conf=round(N['prec_Conf'], 3)))
# LOSO
loso_detail = []
for m in MODELS:
    yt_all, yp_all, hom_all, per = [], [], [], []
    for sp in sorted(set(groups)):
        te = np.where(groups == sp)[0]; tr = np.where(groups != sp)[0]
        yp = fit_predict(m, tr, te)
        yt_all.append(y[te]); yp_all.append(yp); hom_all.append(homolog[te])
        mm = metrics(y[te], yp); mm.update(species=sp, model=m, n=len(te)); per.append(mm)
    yt_all = np.concatenate(yt_all); yp_all = np.concatenate(yp_all); hom_all = np.concatenate(hom_all)
    P = metrics(yt_all, yp_all); N = metrics(yt_all[~hom_all], yp_all[~hom_all])
    perdf = pd.DataFrame(per)
    rows.append(dict(protocol='leave-one-species-out', model=m, acc=f"{P['acc']:.3f} (fold range {perdf['acc'].min():.2f}-{perdf['acc'].max():.2f})",
                     macroF1=round(P['macroF1'], 3), balAcc=round(P['balAcc'], 3), nonhom_acc=round(N['acc'], 3),
                     nonhom_macroF1=round(N['macroF1'], 3), nonhom_rec_Conf=round(N['rec_Conf'], 3), nonhom_prec_Conf=round(N['prec_Conf'], 3)))
    if m == 'XGB all (18)':
        loso_detail = perdf
        cm = confusion_matrix(yt_all, yp_all, labels=[0,1,2]); cmn = confusion_matrix(yt_all[~hom_all], yp_all[~hom_all], labels=[0,1,2])
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
R = pd.DataFrame(rows)
print('\n=== Protocol comparison (pooled metrics; nonhom = loci without a homologous seed) ===')
print(R.to_string(index=False))
print('\n=== LOSO per-species, XGB all (18) ===')
print(loso_detail[['species','n','acc','macroF1','rec_Cand','rec_Conf','rec_FP','prec_Conf']].round(3).to_string(index=False))
print('\nLOSO pooled confusion matrix (rows true Cand/Conf/FP; cols predicted):\n', cm)
print('LOSO pooled confusion, non-homologous loci only:\n', cmn)
# Dataset fingerprint: predict species from the two miRDeep2 'estimated probability' columns alone
Z = X[['mirDeep_sensitivity', 'sensitivity_uncertainty']].fillna(-1)
fp_acc = cross_val_score(DecisionTreeClassifier(random_state=0), Z, groups, cv=StratifiedKFold(5, shuffle=True, random_state=0))
print(f"\nSpecies identifiable from 'mirDeep_sensitivity' + 'sensitivity_uncertainty' alone: 5-fold CV accuracy {fp_acc.mean():.3f} (chance {pd.Series(groups).value_counts(normalize=True).max():.3f})")
# Released model: TreeSHAP attributions on the bundled data
bst = xgb.Booster(); bst.load_model(model_json)
fn = bst.feature_names
dm = xgb.DMatrix(X[fn], feature_names=fn)
contrib = bst.predict(dm, pred_contribs=True)  # (n, n_class, n_feat+1)
ms = np.abs(contrib[:, :, :-1]).mean(axis=(0, 1))
imp = pd.Series(ms, index=fn).sort_values(ascending=False)
print('\nReleased model, mean |SHAP| (share of total):')
print((imp / imp.sum()).round(3).to_string())
gain = pd.Series(bst.get_score(importance_type='total_gain')); print('\nReleased model, share of total gain:'); print((gain / gain.sum()).sort_values(ascending=False).round(3).head(8).to_string())
