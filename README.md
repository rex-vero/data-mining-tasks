# Data Mining

## Dataset: 1000 Omani Cancer Cases
**This project is written using Jupyter Notebook.**

---

### Task 1: Data Overview and Visualization

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

### Task 2: Missing Data Handling and Normalization

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