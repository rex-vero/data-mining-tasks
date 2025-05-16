# داده کاوی

## دیتاست 1000 سرطانی عمانی
**این پروژه با Jupyter Notebook نوشته شده است.**

---

### Task 1: نمایش داده

1. **خواندن کتابخانه های مورد نیاز**
```python
import pandas as pd
import matplotlib.pyplot as plt
```

2. **خواندن دیتاست**
```python
ds = pd.read_csv('_cancer_dataset_uae.csv')
```

3. **انتخاب 1000 نفر از بین کل دیتاست**
```python
data = ds.head(1000)
```

4. **ستون Age برای داده کاوی انتخاب شد**
```python
dd = data.Age
```

5. **Five-Number Summary**
```python
five_number_summary = dd.describe()
```
شامل: minimum, maximum, Q1, median (Q2), and Q3.

6. **Boxplot of Age**
```python
plt.figure(figsize=(16, 8))
plt.grid(True)
plt.title('Age Boxplot')
plt.boxplot(dd)
```

7. **حساب کردن IQR, Min Range و Max Range**
```python
q1 = dd.quantile(0.25)
q3 = dd.quantile(0.75)
IQR = q3 - q1
minimum = q1 - 1.5 * IQR
maximum = q3 + 1.5 * IQR
```

8. **بقیه چارت ها جهت نمایش**
- **هیستوگرام**
```python
_, ax = plt.subplots()
ax.hist(dd, bins=8, linewidth=1, edgecolor="white")
```

- **ECDF Plot**
```python
_, ax = plt.subplots()
ax.ecdf(dd)
```

- **Step Plot**
```python
_, ax = plt.subplots()
ax.stairs(dd, linewidth=0.5)
```

در قدم اولیه داده را انتخاب کرده، ستون Age را به عنوان داده اصلی انتخاب کرده و سپس با پیدا کردن five_number_summary و IQR باکس پلا و بقیه نمودار هارا کشیده و نمایش می دهیم.
---

### Task 2: نمایش داده های گمشده و نرمال سازی

1. **چک کردن داده های گمشده**
```python
data.isnull().sum()
```

2. **کپی گرفتن از داده و جایگذاری داده های گمشده با کلمه unknown**
```python
data = data.copy()
data.fillna("unknown", inplace=True)
```

3. **تایید جایگذاری داده های گمشده**
```python
data.isnull().sum()
```

4. **متود های نرمال سازی روی ستون Age**

- **Z-Score**
```python
dd_zscore = (dd - dd.mean()) / dd.std()
```

- **Min-Max**
```python
dd_minmax = (dd - dd.min()) / (dd.max() - dd.min())
```

داده های گمشده دیتاست خود را پیدا کرده و جایش "unknown" میگذاریم سپس بعد از نمایش عادی داده ها آن را با دو روش zscore و minmax نرمال سازی میکنیم.
---
### Task 3: نمایش خوشه بندی

1. **خواندن کتابخانه های مورد نیاز**
```python
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
```

2. **تنظیم تعداد خوشه ها**

* KMeans

```python
kmeans = KMeans(n_clusters=4)
```
* شعاع و تعداد خوشه برای DBSCAN

```python
dbscan = DBSCAN(eps=1, min_samples=1)
```

3. **درنظر گرفتن ستون جدید تحت عنوان نوع سرطان**
```python
dt=data.Cancer_Type
```
4. **رسم خوشه**
* KMeans

```python
sns.scatterplot(ax=axes[0], x=data.Age, y=dt, hue=k_labels, palette='Set2', s=70)
axes[0].set_title("KMeans Clustering")
axes[0].set_yticks(dt)
```
* DBSCAN
```python
sns.scatterplot(ax=axes[1], x=data.Age, y=dt, hue=db_labels, palette='Set2', s=70)
axes[1].set_title("DBSCAN Clustering")
axes[1].set_yticks(dt)
plt.show()
```

5. **پیدا کردن مرکز هر خوشه**
```python
kmeans.cluster_centers_.round().astype(int)
```

6. **نمایش تعداد و مقادیر هر خوشه**
* KMeans

```python
for i in range(kmeans.n_clusters):
    ages = data[data['K-means'] == i]['Age'].round().astype(int)
    print(f"\nCluster: {i} (Length: {len(ages)}):")
    print(ages.values)
```
* DBSCAN

```python
for label in sorted(data['DBSCAN'].unique()):
    if label == -1:
        print(f"\nNoise (label = -1):")
    else:
        print(f"\nDBSCAN Cluster: {label}:")
    ages = data[data['DBSCAN'] == label]['Age'].values
    print(ages)
```

بعد از تعریف کتابخانه های مور نیاز، با دو روش خوشه بندی یعنی K-Means و DBSCAN میتوانیم تعداد خوشه ها و حتی مقایر و مرکز هر خوشه را مشاهده کرد.
---
### Task 4: نمایش طبقه بندی

1. **خواندن کتابخانه های مورد نیاز**
```python
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
```

2. **تنظیم لیبل**

```python
le = LabelEncoder()
data['Target'] = le.fit_transform(data['Cancer_Stage'])
x = data[['Age']]
y = data['Target']
```
3. **تقسیم داده به آموزش و تست**
```python
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=42)
```
4. **تنظیم داده جهت یادگیری و برچسب زنی**
- درخت تصمیم گیری
```python
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree_pred = tree.predict(x_test)
```
- بیزین ساده
```python
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
```
5. **گزارش عملکرد**
* Classification
```python
classification_report(y_test, tree_pred, target_names = le.classes_) ##درخت تصمیم گیری
classification_report(y_test, nb_pred, target_names = le.classes_) ##بیزین ساده
```
* Confusion Matrix
```python
confusion_matrix(y_test, tree_pred) ##درخت تصمیم گیری
confusion_matrix(y_test, nb_pred) ##بیزین ساده
```
6. **رسم نمودار ها**
- درخت تصمیم گیری
```python
fig=plt.figure(figsize=(24,12))
plot_tree(tree,feature_names=['Age'],class_names=le.classes_,filled=True,rounded=True,fontsize=5)
plt.title("Decision Tree Classifier",fontsize=20)
plt.show()
```
- بیزین ساده
```python
plt.figure(figsize = (10, 6))
for label in sorted(y.unique()):
    sns.kdeplot(x_train[y_train == label]['Age'], fill = True, label = le.inverse_transform([label])[0])
plt.title("Naive Bayes - Density of Age per Class")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
```
بعد از اینکه ستون Cancer_Stage را کلس خود قرار دادیم آن را با ستون Age مقایسه میکنیم، سپس داده را به آموزش و آزمون تقسیم میکنیم و با آن درخت تصمیم گیری را رسم نموده، گزارش عملکرد گرفته و درخت را نمایش میدهیم، همین روند نیز برای بیزین ساده نیز انجام میگردد.
---