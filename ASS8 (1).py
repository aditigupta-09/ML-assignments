import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

np.random.seed(0)

spam_path = r"C:\Users\HP\Downloads\s.csv"
heart_path = r"C:\Users\HP\Downloads\heartdisese.csv"
wisdm_path = r"C:\Users\HP\Downloads\WISDM.txt"

# Q1 SMS Spam
df_spam = pd.read_csv(spam_path, encoding='utf-8', engine='python')
if 'label' not in df_spam.columns and 'Label' in df_spam.columns:
    df_spam.rename(columns={'Label':'label'}, inplace=True)
if 'text' not in df_spam.columns and 'message' in df_spam.columns:
    df_spam.rename(columns={'message':'text'}, inplace=True)
df_spam = df_spam.loc[:, ['label','text']]
df_spam = df_spam.dropna(subset=['label','text'])
df_spam['label_num'] = df_spam['label'].map({'spam':1,'ham':0})
df_spam = df_spam[df_spam['label_num'].isin([0,1])]
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=r'(?u)\b\w+\b')
X_text = vectorizer.fit_transform(df_spam['text'])
y_text = df_spam['label_num'].values
X_train, X_test, y_train, y_test = train_test_split(X_text, y_text, test_size=0.2, random_state=42, stratify=y_text)
vals, counts = np.unique(y_train, return_counts=True)
print("SMS class distribution train", dict(zip(vals,counts)))
stump = DecisionTreeClassifier(max_depth=1, random_state=0)
stump.fit(X_train, y_train)
y_tr_pred = stump.predict(X_train)
y_te_pred = stump.predict(X_test)
print("SMS stump train acc", accuracy_score(y_train, y_tr_pred))
print("SMS stump test acc", accuracy_score(y_test, y_te_pred))
print("SMS stump confusion test\n", confusion_matrix(y_test, y_te_pred))

def manual_adaboost(X, y, T):
    m = X.shape[0]
    w = np.ones(m)/m
    learners = []
    alphas = []
    errors = []
    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=1, random_state= t)
        clf.fit(X, y, sample_weight=w)
        pred = clf.predict(X)
        miss = (pred != y).astype(int)
        err = np.dot(w, miss)/w.sum()
        if err <= 0:
            alpha = 1.0
        elif err >= 0.5:
            break
        else:
            alpha = 0.5*np.log((1-err)/err)
        w = w * np.exp(-alpha * (2*y-1)*(2*pred-1))
        w = w / w.sum()
        learners.append(clf)
        alphas.append(alpha)
        errors.append(err)
        mis_idx = np.where(miss==1)[0]
        print("Iter", t+1, "mis_idx", mis_idx.tolist(), "mis_weights", w[mis_idx].tolist() if len(mis_idx)>0 else [], "alpha", alpha)
    return learners, alphas, errors

X_train_arr = X_train
y_train_arr = y_train
learners, alphas, errors = manual_adaboost(X_train_arr, y_train_arr, 15)
plt.figure(); plt.plot(range(1,len(errors)+1), errors, marker='o'); plt.xlabel('Iteration'); plt.ylabel('Weighted Error'); plt.title('SMS: Iter vs Weighted Error'); plt.savefig('sms_error.png')
plt.figure(); plt.plot(range(1,len(alphas)+1), alphas, marker='o'); plt.xlabel('Iteration'); plt.ylabel('Alpha'); plt.title('SMS: Iter vs Alpha'); plt.savefig('sms_alpha.png')
def predict_ensemble(X, learners, alphas):
    if len(learners)==0:
        return np.zeros(X.shape[0], dtype=int)
    agg = np.zeros((X.shape[0],))
    for clf, a in zip(learners, alphas):
        agg += a * (2*clf.predict(X)-1)
    return (agg>0).astype(int)
y_train_pred_manual = predict_ensemble(X_train_arr, learners, alphas)
y_test_pred_manual = predict_ensemble(X_test, learners, alphas)
print("SMS manual boost train acc", accuracy_score(y_train_arr, y_train_pred_manual))
print("SMS manual boost test acc", accuracy_score(y_test, y_test_pred_manual))
print("SMS manual confusion\n", confusion_matrix(y_test, y_test_pred_manual))
adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.6, random_state=0)
adb.fit(X_train, y_train)
y_train_adb = adb.predict(X_train)
y_test_adb = adb.predict(X_test)
print("SMS sklearn adb train acc", accuracy_score(y_train, y_train_adb))
print("SMS sklearn adb test acc", accuracy_score(y_test, y_test_adb))
print("SMS sklearn confusion\n", confusion_matrix(y_test, y_test_adb))

# Q2 Heart Disease
df_heart = pd.read_csv(heart_path, engine='python')
df_heart.columns = [c.strip() for c in df_heart.columns]
possible_target_names = ['target','Target','disease','condition','has_disease','HeartDisease','heartdisease']
target_col = None
for t in possible_target_names:
    if t in df_heart.columns:
        target_col = t
        break
if target_col is None:
    target_col = df_heart.columns[-1]
Xh = df_heart.drop(columns=[target_col])
yh = df_heart[target_col]
yh = pd.to_numeric(yh, errors='coerce')
Xh = Xh.apply(lambda col: pd.to_numeric(col, errors='ignore'))
num_cols = Xh.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in Xh.columns if c not in num_cols]
for c in num_cols:
    Xh[c] = Xh[c].fillna(Xh[c].median())
for c in cat_cols:
    Xh[c] = Xh[c].fillna(Xh[c].mode().iloc[0] if not Xh[c].mode().empty else "NA")
if len(cat_cols)>0:
    Xh = pd.concat([Xh[num_cols].reset_index(drop=True), pd.get_dummies(Xh[cat_cols], drop_first=True).reset_index(drop=True)], axis=1)
yh = yh.fillna(yh.mode().iloc[0] if not yh.mode().empty else 0).astype(int)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(Xh.values, yh.values, test_size=0.3, random_state=42, stratify=yh)
scaler_h = StandardScaler()
X_train_h = scaler_h.fit_transform(X_train_h)
X_test_h = scaler_h.transform(X_test_h)
stump_h = DecisionTreeClassifier(max_depth=1, random_state=0)
stump_h.fit(X_train_h, y_train_h)
print("Heart stump train acc", accuracy_score(y_train_h, stump_h.predict(X_train_h)))
print("Heart stump test acc", accuracy_score(y_test_h, stump_h.predict(X_test_h)))
print("Heart stump confusion\n", confusion_matrix(y_test_h, stump_h.predict(X_test_h)))
n_estimators_list = [5,10,25,50,100]
learning_rates = [0.1,0.5,1.0]
results = {}
for lr in learning_rates:
    accs = []
    for n in n_estimators_list:
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n, learning_rate=lr, random_state=0)
        model.fit(X_train_h, y_train_h)
        acc = accuracy_score(y_test_h, model.predict(X_test_h))
        accs.append(acc)
        results[(lr,n)] = acc
    plt.plot(n_estimators_list, accs, marker='o', label=f'lr={lr}')
plt.xlabel('n_estimators'); plt.ylabel('Accuracy'); plt.title('Heart: n_estimators vs accuracy'); plt.legend(); plt.savefig('heart_n_estimators_accuracy.png')
best_cfg = max(results.items(), key=lambda x: x[1])[0]
best_acc = results[best_cfg]
print("Heart best config (lr,n)", best_cfg, "acc", best_acc)
best_lr, best_n = best_cfg
best_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=best_n, learning_rate=best_lr, random_state=0)
best_model.fit(X_train_h, y_train_h)
print("Heart best test acc", accuracy_score(y_test_h, best_model.predict(X_test_h)))
print("Heart classification report\n", classification_report(y_test_h, best_model.predict(X_test_h)))
if hasattr(best_model, 'estimator_errors_'):
    errs = best_model.estimator_errors_
else:
    errs = None
if hasattr(best_model, 'estimator_weights_'):
    est_w = best_model.estimator_weights_
else:
    est_w = None
print("Heart estimator errors", errs)
print("Heart estimator weights", est_w)
feature_importances = best_model.feature_importances_
feat_names = Xh.columns.tolist()
feat_imp_df = pd.DataFrame({'feature':feat_names,'importance':feature_importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
top5 = feat_imp_df.head(5)
print("Heart top5 features\n", top5)

# Manual AdaBoost for heart to collect errors and sample weights
def manual_adaboost_collect(X, y, T):
    m = X.shape[0]
    w = np.ones(m)/m
    learners = []
    alphas = []
    errors = []
    weights_history = []
    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=1, random_state=t)
        clf.fit(X, y, sample_weight=w)
        pred = clf.predict(X)
        miss = (pred != y).astype(int)
        err = np.dot(w, miss)/w.sum()
        if err <= 0:
            alpha = 1.0
        elif err >= 0.5:
            break
        else:
            alpha = 0.5*np.log((1-err)/err)
        w = w * np.exp(-alpha * (2*y-1)*(2*pred-1))
        w = w / w.sum()
        learners.append(clf); alphas.append(alpha); errors.append(err); weights_history.append(w.copy())
    return learners, alphas, errors, weights_history

learners_h, alphas_h, errors_h, weights_hist_h = manual_adaboost_collect(X_train_h, y_train_h, best_n if best_n>0 else 10)
plt.figure(); plt.plot(range(1,len(errors_h)+1), errors_h, marker='o'); plt.xlabel('Iteration'); plt.ylabel('Error'); plt.title('Heart manual weak learner error'); plt.savefig('heart_manual_error.png')
plt.figure(); plt.plot(range(1,len(alphas_h)+1), alphas_h, marker='o'); plt.xlabel('Iteration'); plt.ylabel('Alpha'); plt.title('Heart manual alpha'); plt.savefig('heart_manual_alpha.png')
final_weights = weights_hist_h[-1] if len(weights_hist_h)>0 else np.array([])
if final_weights.size>0:
    plt.figure(); plt.hist(final_weights, bins=20); plt.title('Heart final sample weight distribution'); plt.savefig('heart_weights_hist.png')
    high_idx = np.argsort(-final_weights)[:10]
    print("Heart highest weight sample indices", high_idx.tolist())

# Q3 WISDM
lines = []
with open(wisdm_path, 'r', errors='ignore') as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        parts = re.split('[,; ]+', line)
        if len(parts) < 5:
            continue
        user = parts[0]
        activity = parts[1]
        try:
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
        except:
            continue
        lines.append((user, activity, x, y, z))
df_w = pd.DataFrame(lines, columns=['user','activity','x','y','z'])
df_w['act_lower'] = df_w['activity'].str.lower()
def map_act(a):
    if 'jog' in a or 'up' in a:
        return 1
    if 'walk' in a or 'sit' in a or 'stand' in a or 'down' in a:
        return 0
    return np.nan
df_w['label'] = df_w['act_lower'].apply(map_act)
df_w = df_w.dropna(subset=['label'])
X_w = df_w[['x','y','z']].values
y_w = df_w['label'].astype(int).values
Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_w, y_w, test_size=0.3, random_state=42, stratify=y_w)
stump_w = DecisionTreeClassifier(max_depth=1, random_state=0)
stump_w.fit(Xw_train, yw_train)
print("WISDM stump train acc", accuracy_score(yw_train, stump_w.predict(Xw_train)))
print("WISDM stump test acc", accuracy_score(yw_test, stump_w.predict(Xw_test)))
print("WISDM stump confusion\n", confusion_matrix(yw_test, stump_w.predict(Xw_test)))

learners_w, alphas_w, errors_w = manual_adaboost(Xw_train, yw_train, 20)
plt.figure(); plt.plot(range(1,len(errors_w)+1), errors_w, marker='o'); plt.xlabel('Iter'); plt.ylabel('Error'); plt.title('WISDM error'); plt.savefig('wisdm_error.png')
plt.figure(); plt.plot(range(1,len(alphas_w)+1), alphas_w, marker='o'); plt.xlabel('Iter'); plt.ylabel('Alpha'); plt.title('WISDM alpha'); plt.savefig('wisdm_alpha.png')
ytr_pred_w = predict_ensemble(Xw_train, learners_w, alphas_w)
yte_pred_w = predict_ensemble(Xw_test, learners_w, alphas_w)
print("WISDM manual train acc", accuracy_score(yw_train, ytr_pred_w))
print("WISDM manual test acc", accuracy_score(yw_test, yte_pred_w))
print("WISDM manual confusion\n", confusion_matrix(yw_test, yte_pred_w))
adb_w = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=1.0, random_state=0)
adb_w.fit(Xw_train, yw_train)
print("WISDM sklearn train acc", accuracy_score(yw_train, adb_w.predict(Xw_train)))
print("WISDM sklearn test acc", accuracy_score(yw_test, adb_w.predict(Xw_test)))
print("WISDM sklearn confusion\n", confusion_matrix(yw_test, adb_w.predict(Xw_test)))
