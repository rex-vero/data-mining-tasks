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

---
### Task 3: نمایش خوشه بندی

1. **خواندن کتابخانه های مورد نیاز**
```python
from sklearn.cluster import KMeans
import seaborn as sns
```

2. **تنظیم تعداد خوشه ها**
```python
kmeans = KMeans(n_clusters=4)
```

3. **رسم خوشه**
```python
plt.figure(figsize=(10, 3))
sns.scatterplot(x=data['Age'], y=[0]*len(data), hue=labels, palette='Set2', s=100)
plt.title("K-Means")
plt.yticks([])
plt.xlabel("Ages")
plt.show()
```

4. **پیدا کردن مرکز هر خوشه**
```python
kmeans.cluster_centers_.round().astype(int)
```

5. **نمایش تعداد و مقادیر هر خوشه**
```python
for i in range(kmeans.n_clusters):
    ages = data[data['K-means'] == i]['Age'].round().astype(int)
    print(f"\nCluster: {i} (Length: {len(ages)}):")
    print(ages.values)
```