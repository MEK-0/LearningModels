import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk

# Veri Seti Yolu
data_path = '/Users/mahmutesat/Desktop/Machine-learning/SATILIK_EV1.xlsx'
veri = pd.read_excel(data_path, engine='openpyxl')

# Bağımsız değişkenler ve hedef değişken
x = veri[['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı']]
y = veri[['Fiyat']]

# Eğitim ve test verisi ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Model oluşturma ve eğitim
model = LinearRegression()
model.fit(x_train, y_train)

# Modelin katsayılarını dataframe'e dönüştürme
coeff_df = pd.DataFrame(model.coef_.T, x.columns, columns=['Öznitelik_Katsayıları'])
print(coeff_df)

root = tk.Tk()
root.title("Ev Fiyatı Tahmin Uygulaması")

canvas1 = tk.Canvas(root, width=400, height=400)
canvas1.pack()

# Kullanıcıdan girdi almak için etiketler ve giriş kutuları
label1 = tk.Label(root, text='Oda Sayısı')
canvas1.create_window(65, 100, window=label1)
entry1 = tk.Entry(root)
canvas1.create_window(200, 100, window=entry1)

label2 = tk.Label(root, text='Net M2')
canvas1.create_window(75, 120, window=label2)
entry2 = tk.Entry(root)
canvas1.create_window(200, 120, window=entry2)

label3 = tk.Label(root, text='Katı :')
canvas1.create_window(80, 140, window=label3)
entry3 = tk.Entry(root)
canvas1.create_window(200, 140, window=entry3)

label4 = tk.Label(root, text='Yaşı :')
canvas1.create_window(80, 160, window=label4)
entry4 = tk.Entry(root)
canvas1.create_window(200, 160, window=entry4)

# Sonucu gösterecek etiket
label_Prediction = tk.Label(root, text="Tahmin Edilen Fiyat: ", bg='lawngreen')
canvas1.create_window(200, 220, window=label_Prediction)


# Tahmin fonksiyonu
def values():
    try:
        # Girişleri al
        Oda_Sayısı = float(entry1.get())
        Net_M2 = float(entry2.get())
        Kat = int(entry3.get())
        Yas = int(entry4.get())

        # Model tahminini yap
        predicted_price = model.predict([[Oda_Sayısı, Net_M2, Kat, Yas]])[0][0]

        # Sonucu tkinter etiketinde göster
        result_text = f'Evin Tahmin Edilen Fiyatı (₺): {predicted_price:,.2f}'
        label_Prediction.config(text=result_text)

    except ValueError:
        label_Prediction.config(text="Lütfen geçerli sayılar giriniz!")


# Tahmin butonu
button1 = tk.Button(root, text='Evin Tahmin Fiyatını Hesapla', command=values, bg='azure')
canvas1.create_window(200, 190, window=button1)

# Ana döngü
root.mainloop()