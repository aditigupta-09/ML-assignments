import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA

house_path = r"C:\Users\HP\Downloads\USA_House_Price.csv"
car_path = r"C:\Users\HP\Downloads\imports-85.data"

house = pd.read_csv(house_path)
house = house.dropna(subset=['price'])
X_house = house.drop(columns=['price'])
X_house = pd.get_dummies(X_house, drop_first=True)
y_house = house['price'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
best_r2 = -np.inf
best_beta = None
for train_index, test_index in kf.split(X_house):
    fold += 1
    X_train, X_test = X_house.values[train_index], X_house.values[test_index]
    y_train, y_test = y_house[train_index], y_house[test_index]
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    beta = np.concatenate([[lr.intercept_], lr.coef_])
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    if r2 > best_r2:
        best_r2 = r2
        best_beta = beta
    print(f"fold{fold}", r2)
print("best_r2", best_r2)
print("best_beta_shape", best_beta.shape)

X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X_house.values, y_house, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_holdout = scaler.transform(X_holdout)

def gradient_descent(X, y, lr, epochs):
    m, n = X.shape
    X_b = np.hstack([np.ones((m,1)), X])
    theta = np.zeros(n+1)
    for _ in range(epochs):
        preds = X_b.dot(theta)
        grad = (2/m) * X_b.T.dot(preds - y)
        theta -= lr * grad
    return theta

lrs = [0.001,0.01,0.1,1,3]
best_params = None
best_val_r2 = -np.inf
for lr in lrs:
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=1)
    theta = gradient_descent(X_tr, y_tr, lr, 1000)
    X_val_b = np.hstack([np.ones((X_val.shape[0],1)), X_val])
    y_val_pred = X_val_b.dot(theta)
    val_r2 = r2_score(y_val, y_val_pred)
    X_hold_b = np.hstack([np.ones((X_holdout.shape[0],1)), X_holdout])
    y_hold_pred = X_hold_b.dot(theta)
    hold_r2 = r2_score(y_holdout, y_hold_pred)
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_params = (lr, theta, val_r2, hold_r2)
    print(lr, val_r2, hold_r2)
print("best_lr_and_stats", best_params)

col_names = ["symboling","normalized_losses","make","fuel_type","aspiration","num_doors","body_style","drive_wheels","engine_location","wheel_base","length","width","height","curb_weight","engine_type","num_cylinders","engine_size","fuel_system","bore","stroke","compression_ratio","horsepower","peak_rpm","city_mpg","highway_mpg","price"]
car = pd.read_csv(car_path, names=col_names, na_values='?')
car = car.replace('?', np.nan)
car = car.dropna(subset=['price'])
car['num_doors'] = car['num_doors'].map({'two':2,'four':4}).fillna(car['num_doors'])
car['num_cylinders'] = car['num_cylinders'].map({'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12}).fillna(car['num_cylinders'])
car['num_doors'] = pd.to_numeric(car['num_doors'], errors='coerce').fillna(2).astype(int)
car['num_cylinders'] = pd.to_numeric(car['num_cylinders'], errors='coerce').fillna(4).astype(int)
label_cols = ['make','aspiration','engine_location','fuel_type','fuel_system']
le = LabelEncoder()
for c in label_cols:
    if c in car.columns:
        car[c] = le.fit_transform(car[c].astype(str))
dummy_cols = ['body_style','drive_wheels']
car = pd.get_dummies(car, columns=[c for c in dummy_cols if c in car.columns], drop_first=True)
car = car.dropna(subset=['price'])
y_car = car['price'].astype(float).values
X_car = car.drop(columns=['price'])
num_cols_car = X_car.select_dtypes(include=['int64','float64']).columns.tolist()
X_car[num_cols_car] = X_car[num_cols_car].apply(pd.to_numeric, errors='coerce')
X_car = X_car.fillna(X_car.median())

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_car.values, y_car, test_size=0.3, random_state=42)
sc_car = StandardScaler()
Xc_train = sc_car.fit_transform(Xc_train)
Xc_test = sc_car.transform(Xc_test)
lr_car = LinearRegression()
lr_car.fit(Xc_train, yc_train)
ycar_pred = lr_car.predict(Xc_test)
r2_car = r2_score(yc_test, ycar_pred)
print("car_r2", r2_car)

pca = PCA(n_components=0.95)
Xc_train_p = pca.fit_transform(Xc_train)
Xc_test_p = pca.transform(Xc_test)
lr_car_p = LinearRegression()
lr_car_p.fit(Xc_train_p, yc_train)
ycar_p_pred = lr_car_p.predict(Xc_test_p)
r2_car_p = r2_score(yc_test, ycar_p_pred)
print("car_r2_pca", r2_car_p)
