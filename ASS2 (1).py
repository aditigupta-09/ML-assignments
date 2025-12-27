import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard

df = pd.read_csv(r"C:\Users\HP\Downloads\AWCustomers.csv")

selected_columns = ['CommuteDistance','YearlyIncome','Age','Gender','MaritalStatus','Education','BikeBuyer']
df_selected = df[[c for c in selected_columns if c in df.columns]].copy()

dtype_map = {}
for c in df_selected.columns:
    t = df_selected[c].dtype
    if pd.api.types.is_integer_dtype(t):
        dtype_map[c] = 'Discrete'
    elif pd.api.types.is_float_dtype(t):
        dtype_map[c] = 'Continuous'
    else:
        dtype_map[c] = 'Nominal'
print(dtype_map)

df_pp = df_selected.copy()
num_cols = [c for c in df_pp.columns if pd.api.types.is_numeric_dtype(df_pp[c])]
cat_cols = [c for c in df_pp.columns if c not in num_cols]

for c in num_cols:
    df_pp[c] = df_pp[c].fillna(df_pp[c].median())
for c in cat_cols:
    df_pp[c] = df_pp[c].fillna(df_pp[c].mode().iloc[0] if not df_pp[c].mode().empty else "")

scaler_mm = MinMaxScaler()
if num_cols:
    df_mm = df_pp.copy()
    df_mm[num_cols] = scaler_mm.fit_transform(df_mm[num_cols])
else:
    df_mm = df_pp.copy()
print(df_mm.head())

if 'Age' in df_pp.columns:
    df_pp['AgeBin'] = pd.cut(df_pp['Age'], bins=5, labels=False)
else:
    df_pp['AgeBin'] = pd.Series(dtype='int')

kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
cont_cols = [c for c in num_cols if c not in ['BikeBuyer']]
if cont_cols:
    df_pp[[c+'_bin' for c in cont_cols]] = kb.fit_transform(df_pp[cont_cols])
else:
    pass
print(df_pp[[c for c in df_pp.columns if c.endswith('_bin')]].head())

scaler_std = StandardScaler()
if num_cols:
    df_std = df_pp.copy()
    df_std[num_cols] = scaler_std.fit_transform(df_std[num_cols])
else:
    df_std = df_pp.copy()
print(df_std[num_cols].head() if num_cols else None)

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
if cat_cols:
    ohe_cols = ohe.fit_transform(df_pp[cat_cols])
    ohe_df = pd.DataFrame(ohe_cols, columns=ohe.get_feature_names_out(cat_cols), index=df_pp.index)
else:
    ohe_df = pd.DataFrame(index=df_pp.index)
df_final = pd.concat([df_std[num_cols].reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

bin_df = pd.get_dummies(df_pp[[c for c in cat_cols if c in df_pp.columns]].astype(str))
if 'AgeBin' in df_pp.columns:
    bin_df = pd.concat([bin_df, pd.get_dummies(df_pp['AgeBin'].astype(str))], axis=1)
if 'BikeBuyer' in df_pp.columns:
    bin_df['BikeBuyer'] = df_pp['BikeBuyer'].astype(int)
bin_df = bin_df.loc[:,~bin_df.columns.duplicated()]

if len(bin_df) >= 2:
    x = bin_df.iloc[0].values
    y = bin_df.iloc[1].values
    simple_matching = np.sum(x==y)/len(x)
    jac_dist = jaccard(x,y)
else:
    simple_matching = None
    jac_dist = None

if len(df_final) >= 2:
    vec1 = np.hstack([df_final.iloc[0].values.reshape(1,-1), bin_df.iloc[0].values.reshape(1,-1)]) if not df_final.empty else bin_df.iloc[0].values.reshape(1,-1)
    vec2 = np.hstack([df_final.iloc[1].values.reshape(1,-1), bin_df.iloc[1].values.reshape(1,-1)]) if not df_final.empty else bin_df.iloc[1].values.reshape(1,-1)
    cos_sim = cosine_similarity(vec1,vec2)[0][0]
else:
    cos_sim = None

if 'CommuteDistance' in df_pp.columns and 'YearlyIncome' in df_pp.columns:
    try:
        comm_codes = pd.Categorical(df_pp['CommuteDistance']).codes
        corr = pd.Series(comm_codes).corr(df_pp['YearlyIncome'])
    except Exception:
        corr = None
else:
    corr = None

print(simple_matching)
print(jac_dist)
print(cos_sim)
print(corr)
