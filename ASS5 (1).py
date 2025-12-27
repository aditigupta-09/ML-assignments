import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
n=1000
mean=np.zeros(8)
rho=0.95
cov=np.full((8,8),rho)
np.fill_diagonal(cov,1)
data=np.random.multivariate_normal(mean,cov,size=n)
X_synth=data[:,:7]
coef_true=np.array([3.5, -2.1, 1.7, 0.5, -1.2, 0.9, 2.3])
y_synth=X_synth.dot(coef_true)+np.random.normal(0,1,size=n)
X_train,X_test,y_train,y_test=train_test_split(X_synth,y_synth,test_size=0.3,random_state=42)
def ridge_gd(X,y,lr,alpha,epochs):
    m,n=X.shape
    Xb=np.hstack([np.ones((m,1)),X])
    theta=np.zeros(n+1)
    for _ in range(epochs):
        preds=Xb.dot(theta)
        grad=(2/m)*Xb.T.dot(preds-y)
        grad[1:]+=2*alpha*theta[1:]
        theta-=lr*grad
    return theta
lrs=[0.0001,0.001,0.01,0.1,1,10]
alphas=[10**i for i in [-15,-10,-5,-3,-1,0,1,10,20]]
best=None
best_stats=None
for lr in lrs:
    for a in alphas:
        th=ridge_gd(X_train,y_train,lr,a,1000)
        Xb_test=np.hstack([np.ones((X_test.shape[0],1)),X_test])
        ypred=Xb_test.dot(th)
        r=r2_score(y_test,ypred)
        cost=np.mean((Xb_test.dot(th)-y_test)**2)+a*np.sum(th[1:]**2)
        if best is None or (cost<best and r> (best_stats[2] if best_stats is not None else -np.inf)):
            best=cost
            best_stats=(lr,a, r, th)
print("Q1_best_lr,alpha,r2",best_stats[0],best_stats[1],best_stats[2])
print("Q1_best_theta_shape",best_stats[3].shape)

hitters_path=r"/mnt/data/56acb462-bef2-44b6-b4ae-86a817f1f437.png"
try:
    hitters=pd.read_csv(hitters_path)
except Exception:
    hitters=pd.read_csv(hitters_path, delim_whitespace=True, engine='python')
cols = hitters.columns.tolist()
if 'Salary' in hitters.columns:
    target='Salary'
elif 'salary' in hitters.columns:
    target='salary'
elif 'Price' in hitters.columns:
    target='Price'
elif 'Hitters' in hitters.columns:
    target='Hitters'
else:
    target=hitters.columns[-1]
X=hitters.drop(columns=[target])
y=pd.to_numeric(hitters[target], errors='coerce')
X = X.apply(lambda col: pd.to_numeric(col, errors='ignore'))
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
for c in num_cols:
    X[c]=X[c].fillna(X[c].median())
for c in cat_cols:
    X[c]=X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "")
if cat_cols:
    X=pd.concat([X[num_cols].reset_index(drop=True), pd.get_dummies(X[cat_cols], drop_first=True).reset_index(drop=True)], axis=1)
X = X.fillna(X.median())
y = y.fillna(y.median())
X_train,X_test,y_train,y_test = train_test_split(X.values,y.values,test_size=0.3,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
lr = LinearRegression()
lr.fit(X_train,y_train)
r_lr = r2_score(y_test, lr.predict(X_test))
ridge = Ridge(alpha=0.5748)
ridge.fit(X_train,y_train)
r_ridge = r2_score(y_test, ridge.predict(X_test))
lasso = Lasso(alpha=0.5748, max_iter=10000)
lasso.fit(X_train,y_train)
r_lasso = r2_score(y_test, lasso.predict(X_test))
print("Q2_linear_r2",r_lr)
print("Q2_ridge_r2",r_ridge)
print("Q2_lasso_r2",r_lasso)

try:
    boston = load_boston()
    Xb=boston.data
    yb=boston.target
except Exception:
    from sklearn.datasets import fetch_openml
    b = fetch_openml(name="Boston", version=1, as_frame=False)
    Xb=b.data
    yb=b.target.astype(float)
alphas_cv = np.logspace(-3,3,50)
ridgecv = RidgeCV(alphas=alphas_cv, cv=5)
ridgecv.fit(Xb,yb)
lassocv = LassoCV(alphas=None, cv=5, max_iter=10000)
lassocv.fit(Xb,yb)
print("Q3_ridgecv_alpha",ridgecv.alpha_,"score",ridgecv.score(Xb,yb))
print("Q3_lassocv_alpha",lassocv.alpha_,"score",lassocv.score(Xb,yb))

iris = load_iris()
Xir = iris.data
yir = iris.target
Xtr,Xte,ytr,yte = train_test_split(Xir,yir,test_size=0.3,random_state=42)
sc_iri=StandardScaler()
Xtr=sc_iri.fit_transform(Xtr)
Xte=sc_iri.transform(Xte)
clf = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000)
clf.fit(Xtr,ytr)
print("Q4_ovr_test_score",clf.score(Xte,yte))
