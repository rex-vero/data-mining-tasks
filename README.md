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
fig, ax = plt.subplots()
ax.hist(dd, bins=8, linewidth=1, edgecolor="white")
```

- **ECDF Plot**
```python
fig, ax = plt.subplots()
ax.ecdf(dd)
```

- **Step Plot**
```python
fig, ax = plt.subplots()
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
اول برای K means

```python
kmeans = KMeans(n_clusters=4)
```
بعد برای DBSCAN که شعاع و حداقل خوشه را نشان میدهیم

```python
dbscan = DBSCAN(eps=1, min_samples=1)
```

3. **درنظر گرفتن ستون جدید تحت عنوان نوع سرطان**
```python
dt=data.Cancer_Type
```
4. **رسم خوشه**
رسم kmeans با ستون عمودی تایپ سرطان

```python
sns.scatterplot(ax=axes[0], x=data.Age, y=dt, hue=k_labels, palette='Set2', s=70)
axes[0].set_title("KMeans Clustering")
axes[0].set_yticks(dt)
```
رسم dbscan با ستون عمودی تایپ سرطان

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
kmeans

```python
for i in range(kmeans.n_clusters):
    ages = data[data['K-means'] == i]['Age'].round().astype(int)
    print(f"\nCluster: {i} (Length: {len(ages)}):")
    print(ages.values)
```
dbscan

```python
for label in sorted(data['DBSCAN'].unique()):
    if label == -1:
        print(f"\nNoise (label = -1):")
    else:
        print(f"\nDBSCAN Cluster: {label}:")
    ages = data[data['DBSCAN'] == label]['Age'].values
    print(ages)
```

بعد از تعریف کتابخانه های مور نیاز، با دو روش خوشه بندی یعنی K-Means و DBSCAN میتوانیم تعداد خوشه ها و حتی مقایر و مرکز هر خوشه را مشاهده کرد
---
### Task 4: نمایش طبقه بندی