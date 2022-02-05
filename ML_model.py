import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# function for deciding if its hard landing or not
def landing(row):
    if row['Vert_accel_landing'] >= 1.2:
        x = 1
    else:
        x = 0
    return x


data = "C:\\Users\\kt733e\\Documents\\Work\\Proj\\Tail652\\Table\\table.csv"

df = pd.read_csv(data)
# defining plot parameters
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
plt.style.use('seaborn-whitegrid')
plt.figure(dpi=100, figsize=(10, 7))

# Visualizing the vertical acceleration distribtion
sns.histplot(x='Vert_accel_landing', data=df, kde=True, binwidth=0.01)

# plotting correlation plot
f, ax = plt.subplots(figsize=(10, 7))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# making flight number column as float
df['flightNum'] = df['flightNum'].astype('float64')

# dropping object columns as they don't drive vertical acceleration
col = [x for x in df.columns if df[x].dtype == 'object']
data = df.drop(col, axis=1)


data['Hard_ldg'] = data.apply(landing, axis=1)
data = data.set_index('flightNum')

# making new feature after conbining 2 features
data['ratio1'] = data['IVV_8s_before']/data['GS_8s_before']

x = data.drop(['Hard_ldg', 'Vert_accel_landing', 'Vert_accel_touchdown', 'ldg_apt_id'], axis=1)
y = data['Hard_ldg']

# spliting data into train and test samples
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2, shuffle=True, random_state=0)

# preprocessing the data to make normalise the samples
scaler = StandardScaler()
x_train_norm = scaler.fit_transform(train_x)
x_test_norm = scaler.fit_transform(test_x)

# classification model of Linear SVC with class_weight as per ratio of hrad landings in data
clf = LinearSVC(dual=False, class_weight={0: 1, 1: 4}, tol=1e-4)
clf.fit(x_train_norm, train_y)
pred_y2 = clf.predict(x_test_norm)
cm = metrics.confusion_matrix(test_y, pred_y2)
print('Linear SVC:\n Confusion Matrix:\n', cm)

cr = metrics.classification_report(test_y, pred_y2)
print('Classification Report:\n', cr)

f_score = metrics.f1_score(test_y, pred_y2)
print('f1 Score:\n', f_score)

acc = metrics.accuracy_score(test_y, pred_y2, normalize=True, sample_weight=None)
print('Accuracy:\n', acc)

# getting feature importance and plotting it
imp = clf.coef_
sd = pd.DataFrame({'features': train_x.columns, 'imp': pd.Series(imp[0])})
plt.barh(range(len(sd.index)), sd.imp, align='center')
plt.yticks(range(len(sd.index)), sd.features)

plt.show()
