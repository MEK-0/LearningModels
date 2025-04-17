import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox

# --- Veri seti  ---
data = {
    'Tarih': ['2024-01-05', '2024-01-06', '2024-01-08', '2024-01-09', '2024-01-10',
              '2024-01-12', '2024-01-14', '2024-01-15', '2024-01-17', '2024-01-18'],
    'Urun': ['Lavanta', 'Kahve', 'Vanilya', 'Lavanta', 'Vanilya',
             'Kahve', 'Lavanta', 'Kahve', 'Vanilya', 'Lavanta'],
    'Adet': [12, 7, 15, 10, 5, 11, 14, 9, 12, 8],
    'Birim_Fiyat': [40, 45, 38, 40, 38, 45, 40, 45, 38, 40]
}
df = pd.DataFrame(data)
df['Toplam_Gelir'] = df['Adet'] * df['Birim_Fiyat']

# --- Model ---
X = df[['Adet']]
y = df[['Toplam_Gelir']]
model = LinearRegression()
model.fit(X, y)

# --- Tkinter arayüzü ---
def tahmin_et():
    try:
        adet = int(entry.get())
        test_input = np.array([[adet]])
        tahmini_gelir = model.predict(test_input)[0][0]
        messagebox.showinfo("Tahmin Sonucu", f"{adet} adet satıştan tahmini gelir: {tahmini_gelir:.2f} TL")
    except ValueError:
        messagebox.showerror("Hata", "Lütfen geçerli bir sayı giriniz.")


pencere = tk.Tk()
pencere.title("Satış Tahmini Aracı")
pencere.geometry("400x200")


etiket = tk.Label(pencere, text="Tahmini satış adedini giriniz:")
etiket.pack(pady=10)

entry = tk.Entry(pencere, font=("Arial", 14))
entry.pack(pady=5)


buton = tk.Button(pencere, text="Tahmini Geliri Hesapla", command=tahmin_et)
buton.pack(pady=20)


pencere.mainloop()
