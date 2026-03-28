# تکلیف: یکپارچه‌سازی داده‌های ورود و رمز عبور (Pandas & NumPy)

**هدف پروژه:** پیاده‌سازی تابعی برای ترکیب داده‌های یک **DataFrame** و یک **آرایه NumPy**، انجام اصلاحات درجا (In-place) و ذخیره خروجی نهایی.

---

## صورت مسئله و ورودی‌ها

یک شرکت داده‌های کاربران خود را در دو ظرف (Container) مجزا ذخیره کرده است:
1.  **ورودی اول (`id_name_verified`):** یک DataFrame (خوانده شده از `data.csv`) شامل ستون‌های: `Id` ،`Login` و `Verified`.
2.  **ورودی دوم (`id_password`):** یک آرایه دو‌بعدی NumPy (خوانده شده از `passwords.npy`) که هر سطر آن شامل دو مقدار است: `Id` و `Password`.

**نکته کلیدی:** در هر دو منبع، ردیف‌هایی که در یک شاخص (Index) قرار دارند، دارای `Id` یکسان هستند.

---

## الزامات زیروظایف (Subtasks)

1.  **حذف ستون اضافی:** ستون `Verified` باید از DataFrame حذف شود.
2.  **افزودن رمز عبور:** ستون `Password` از آرایه NumPy استخراج شده و به عنوان آخرین ستون به DataFrame اضافه شود.
3.  **اصلاح درجا (In-place):** تغییرات باید مستقیماً روی DataFrame اصلی اعمال شوند (تابع نباید مقدار جدیدی بازگرداند).
4.  **ذخیره‌سازی:** نتیجه نهایی باید در فایل `output.csv` ذخیره شود.

---

## پیاده‌سازی کد (Solution)

در اینجا کد کامل برای حل این مسئله آورده شده است:

```python
import pandas as pd
import numpy as np

def login_table(id_name_verified, id_password):
    """
    اصلاح درجا (In-place) دیتافریم برای حذف تاییدیه و اضافه کردن رمز عبور
    """
    # ۱. حذف ستون Verified به صورت درجا
    id_name_verified.drop(columns=['Verified'], inplace=True)
    
    # ۲. اضافه کردن ستون Password از ستون دوم آرایه NumPy (اندیس 1)
    # با توجه به اینکه ترتیب ردیف‌ها یکسان است، مستقیماً جایگذاری می‌کنیم
    id_name_verified['Password'] = id_password[:, 1]

# --- بخش اجرایی اسکریپت ---

try:
    # بارگذاری داده‌ها
    id_name_verified = pd.read_csv("data.csv")
    id_password = np.load("passwords.npy")

    # فراخوانی تابع برای اعمال تغییرات
    login_table(id_name_verified, id_password)

    # ذخیره نتیجه نهایی در فایل CSV
    id_name_verified.to_csv("output.csv", index=False)
    print("عملیات با موفقیت انجام شد و فایل output.csv تولید گردید.")

except FileNotFoundError:
    print("خطا: فایل‌های ورودی (data.csv یا passwords.npy) یافت نشدند.")
except Exception as e:
    print(f"یک خطای غیرمنتظره رخ داد: {e}")
```