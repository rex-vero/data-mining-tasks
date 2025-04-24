# داده کاوی
## دیتاست 1000 سرطانی عمانی
**این پروژه با JUPYTER NOTEBOOK نوشته شده است**
### task 1
ابتدا با خط های
import pandas as pd
import matplotlib.pyplot as plt
دسترسی خود را به کتابخانه های مورد نیاز پروژه چک کرده سپس با دستور
ds = pd.read_csv('_cancer_dataset_uae.csv')
یک دیتاست که به صورت اکسل سیو شده است را در متغیری به نام ds ذخیره سازی نمونده و پس از آن با دستور
data = ds.head(1000)
هزارتا از موجودیت های داده را انتخاب نمونده و چاپ کرده، سپس با دستور
dd = data.Age
فقط ستون سن را اننتخاب نموده و با دستور
five_number_summary = dd.describe()
از five_number_summary پرینت گرفته و مقادیری همچون min, max, q1, q2 و q3 را به دست آورده.
سپس برای نمایش boxplot ستون Age از دستور زیر استفاده میکنیم
plt.figure(figsize=(16, 8))
plt.grid(True)
plt.title('Age boxplot')
plt.boxplot(dd)
همچنین برای اطمینان بیشتر از مقادیر و دقت جدول هم IQR و هم min range و max range با دستور زیر به دست آورده
q1 = dd.quantile(0.25)
q3 = dd.quantile(0.75)
IQR = q3 - q1
minimum = q1 - 1.5 * IQR
maximum = q3 + 1.5 * IQR
همچنین برای کشیدن انواع چارت های دیگر از دستورای زیر استفاده میکنیم
fig, ax = plt.subplots()
ax.hist(dd, bins=8, linewidth=1, edgecolor="white")

fig, ax = plt.subplots()
ax.ecdf(dd)

fig, ax = plt.subplots()
ax.stairs(dd, linewidth=0.5)
### task 2
حال با دستور
data.isnull().sum()
تعداد داده های گمشده جدول هزارتایی داده های خود را به دست آورده، برای درست کردن این موضوع ابتدا از دیتا یک کپی گرفته سپس داده های گمشده را با کلمه unknown جایگذاری میکنیم که به صورت مراحل زیر پیش میرویم
data = data.copy()
data.fillna("unknown", inplace=True)
درنهایت با چاپ دوباره دستور
data.isnull().sum()
متوجه میشویم که هیچ داده گمشده ای وجود ندارد.
برای نرمال سازی به دو روش Z-Score و MinMax ابتدا برای مشاهده مقادیر اولیه جدول Age آن را که تحت عنوان متغیر dd تعریف شده است پرینت میکنیم سپس برای نمایش مقادیر نرمال سازی شده از دستورات زیر استفاده میکنیم.
dd_zscore = (dd - dd.mean()) / dd.std()
dd_minmax = (dd - dd.min()) / (dd.max() - dd.min())