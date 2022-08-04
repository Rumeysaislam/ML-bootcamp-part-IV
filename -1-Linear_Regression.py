######################################################
# Sales Prediction with LINEER REGRESSION / Lineer Reg. ile Satis Tahmin Modeli
# Reklam harcamalarina iliskin ne kadar satis yapildigini elde ediliyor.

# Yapacagimiz sey;
# Once simple lineer reg. yani iki degiskenli basit bir reg. modeli kurmak,
# Daha sonrasinda veri setinde bulunan 5 degiskenin tamamiyla bir model kurmak.
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x) # Virgulden sonra iki basamak goster ayari

# M.L konusunda en fazla kullanilan kutuphanelerden; sklearn kutuphanesinin cesitli modullerinden bir kaç metod cagiriyoruz;
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score



# Simple Linear Regression with OLS Using Scikit-Learn;

df = pd.read_csv("datasets/advertising.csv)
df.shape
# TV, radio, newspaper bagimsiz; sales bagimli degiskenimiz.

X = df[["TV"]]
y = df[["sales"]]
# Bu iki degisken arasinda var oldugunu dusundugumuz dogrusal iliskiyi once modelleyecegiz,
# Sonrasinda bu model denklemini bir grafik yardimiyla degerlendirecegiz.




# Model;

# "LinearRegression()" metodu ile modeli kuruyoruz;
reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV (Tahmini degerimiz = sabit + agırlık * bagimsiz degisken)

# Ilgili nesneleri inceleyelim;
# sabit (b - bias - intercep)
reg_model.intercept_[0]

# tv'nin katsayısı (w1 - coefficient )
reg_model.coef_[0][0] # Degiskenin agirligini bulmus olduk.

# Model kurdugumuzda, elimizde sabit ve agirlik degerleri olmus olacak.




# Tahmin;

# 150 birimlik TV harcamasi olsa ne kadar satis olmasi beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcamasi olsa ne kadar satis olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T

# TV'nin max degeri 296, ben 500 girdim;
# verinin oruntusunu ogrendim, gozlenmemis bir deger bile olsa bunu ogrendigim modele sorabiliriz.
# Satisin max degeri=27 iken 30 'u elde ettik. Modellemenin faydalari :D



# Modelin Gorsellestirilmesi

# Seaborn icerisinden regresyon grafigi olusturmak icin "regplot" metodunu kullaniyoruz;
# x: Bagimsiz, y: Bagimli degisken
# "scatter_kws={'color': 'b', 's': 9}" Gercek degerlere iliskin ayarlar
# "ci=False"; Guven araligini eklememeyi sectik.
# color="r" : Reg. cizgisinin renginin ne olacagi

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

# Basligi "DINAMIK" olarak belirliyoruz; Sales formulunde virgulden sonra iki basamak olsun bilgisini girdik.
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)                                          # y-ekseninde 0'i baslangic noktasi olarak sectik.
plt.show(block=True)

# Iki boyutta iki degiskenli bu modelin gosterimini elde etmis olduk.
# Kirmizi cizgi: Modelimiz; Tahmin denklemimizdir (tahmin edilen degerlerdir).
# Dogrusal iliski oldugunu dusunerek bir model kurduk ve tutarli gorunuyor.




# Tahmin Basarisi;

# MSE
y_pred = reg_model.predict(X) # Girdigim bagimsiz degiskene (x) gore, bagimli degiskeni (y_pred) tahmin etmek istiyoruz.
# mean_squared_error' gercek degerleri ve tahmin edilen degerleri gonderirsek;
print(mean_squared_error(y, y_pred))
# 10.51; Olabilecek min. deger olmasini istiyorum.
y.mean() # Bagimli degisken olan satislarin ortalamasi
y.std() # Bagimli degisken olan satislarin standart sapmasi
# 9-19 arasinda degerler degisiyor gibi, ort. hata (10) bu durumda biraz buyuk gibi :)


# RMSE
# MSE'den gelen ifadenin karakokudur. Karekok fonk: "sqrt"
np.sqrt(mean_squared_error(y, y_pred))
# 3.24


# MAE
mean_absolute_error(y, y_pred)
# 2.54


# R-KARE
# Dogrusal reg. modellerinde modelin basarisina iliskin cok onemli bir metriktir.
# Veri setindeki bagimsiz degiskenlerin, bagimli degiskeni aciklama yuzdesidir.
reg_model.score(X, y)
# Yorum: Bu modelde bagimsiz degikenler, bagimli degiskenin %61'ini aciklayabilmektedir.
# Degisken sayisi arttikca R-Kare sismeye meyillidir. Duzeltilmis R-Kare degerinin de goz onunde bulundurulmasi gerekir.


# Hata metriklerini birbirileri ile kiyaslamak dogru degil.
# Hangisini uyguladiysan, veri setinde yaptigin degisiklikler sonrasi yine onu kullan ve karsilastirmalarini yap. :)

# Dogrusal regresyonlar yuksek tahmin basarili modeller degildir. Gelismis reg. problemlerinde kullanacagimiz modeller; agaca dayali regresyon modelleri olacak.




# Multiple Linear Regression / Coklu Dogrusal Regresyon;

df = pd.read_csv("datasets/advertising.csv")

X = df.drop('sales', axis=1) # drop ile "sales" degiskenini silip; Bag. degiskenlerimi elde ettim.

y = df[["sales"]] # Bagimli degiskenimi sectim.




# Model;
# Veri setini once, train-test diye ayirmaliyiz "train_test_split" metodu ile;
# test_size=0.20 :test seti boyutunu belirttik ve rasgele setler olusturduk.
# random_state=1 : rassaligin ayni olmasini istedigimizde belirtiyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape            # (200'un %20'i; 40 gozlem birimi ve 1 (bag.lı) degisken)
X_test.shape            # (200'un %20'i; 40 gozlem birimi, 3 degisken)

y_train.shape           # (200'un %80'i; 160 gozlem birimi, 1 (bag.lı) degisken )
X_train.shape           # (200'un %80'i; 160 gozlem birimi, 3 degisken )


# train seti ile model kuruyoruz;
reg_model = LinearRegression().fit(X_train, y_train)

# ya da;
# reg_model = LinearRegression()
# reg_model.fit(X_train, y_train)      ;diye gosterebiliriz.


# sabit (b - bias)
reg_model.intercept_[0]

# coefficients (w - weights)
reg_model.coef_[0]




# Tahmin;

# Asagidaki gozlem degerlerine gore satisin beklenen degeri nedir?

# TV: 30
# radio: 10
# newspaper: 40; yeni gozlemlerimiz

# 2.90 : sabit
# 0.0468431 , 0.17854434, 0.00258619 : katsayilar


# MULAKAT SORUSU: Sabiti ve agirlık degerlerini kullanarak model denklemini yaz.

# Model denklemimiz;
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002


2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619 # 6,20

# Fonksiyonel sekilde yazmak icin;
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T                   # Listeyi datframe'e cevirdik ve transpozunu aldik.

reg_model.predict(yeni_veri)                            # Modele yeni veriyi sorduk. 6,20




# Tahmin Başarısını Değerlendirme;

# Train RMSE
y_pred = reg_model.predict(X_train)                    # Train setinin bagimli degiskenini de ayrı sekilde tahmin edebiliyoruz. Modeli trainler uzerinden kurduk.
np.sqrt(mean_squared_error(y_train, y_pred))           # Ort. kare hatasi karekoku
# 1.73 ; Train seti hatamiz.

# Degisken sayisi arttikca, hata degerimiz duser; basari artar.




# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test setindeki hataya da bakilabilir;
# Test RMSE
y_pred = reg_model.predict(X_test)                      # Modele test setinin bagimsiz degiskenlerini sorduk.
np.sqrt(mean_squared_error(y_test, y_pred))             # Bagimsiz degiskenin gercek ve tahmini degerlini girdik.
# 1.41; Test seti hatamiz

# Normalde test hatasi, train hatasindan daha yuksek cikar. Bu durumda guzel bir sonuc elde ettik (1,41 > 1,73).



# Test RKARE
reg_model.score(X_test, y_test)
# Veri setindeki bagimsiz degiskenin, bagimli degiskeni aciklama yuzdesi: ~ %90



# 10 Katlı Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,  # 9 parcasi ile model kurup bir parcasi ila test ettik.
                                 scoring="neg_mean_squared_error")))

# (10 Katli capraz dogrulamayi tum veri seti uzerinden (X, y) yaptık. Cunku veri zaten az)
# "cross_val_score" fonksiyonunun skoru: "neg_mean_squared_error"
# yani negatif ort. hatayi verdiginden cross_val_score'i - ile carptik :D


# Karekok alınca(sqrt), RMSES degerlerini elde etmis olduk. Ortalamasini aldik: 1.69


# Veri setimizin boyutu az oldugundan 10 Katli capraz dogrulama yontemine daha fazla guvenebiliriz.
# Veri setimizin boyutu cok oldugunda fark etmez diyebilirdik.



# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71






# Simple Linear Regression with Gradient Descent from Scratch;

# Basit dogrusal (tek degiskenli) regresyonu kendi fonksiyonlarimizla Gradient Descent kullanarak inceleyecek olursak;


# Cost fonksiyonumuzun amaci MSE'yi hesaplamak;
def cost_function(Y, b, w, X):
    m = len(Y)                          # Gozlem sayisini tuttuk. Cunku butun gozlem birimlerini gezip hatayi hesaplayacagiz.
    sse = 0                             # sum of square error: Hata kareler toplami

    for i in range(0, m):               # Bu uygulama icin m: gozlem sayimiz: 200)
        y_hat = b + w * X[i]            # Sabit ve agirlik degerine gore tahmin edilen y degerleri hesaplandi. ( x[i]: i. bag.siz degiskenin gozlem birimi)
        y = Y[i]                        # i. indexteki gercek degerler
        sse += (y_hat - y) ** 2         # Gercek deger ile tahmini degerin farkinin karesini alip, her gozlem icin sse'ye ekleyerek, guncelliyoruz ve toplam hatayi buluyoruz.

    mse = sse / m                       # Ortalama hatayi bulmak icin
    return mse


# update_weights / Bir tane agirligi gozlem birimlerinde gezdirip update etmek icin;
def update_weights(Y, b, w, X, learning_rate):                  # Argumanlara "learning_rate"i (Ogrenme hizi) de ekledik.
    m = len(Y)                                                  # Gozlem sayisini tuttuk.

    b_deriv_sum = 0                                             # Hesaplayacagimiz kismi turevleri de tutuyoruz.
    w_deriv_sum = 0

    for i in range(0, m):                                       # Tum gozlem birimlerine gidiyoruz.
        y_hat = b + w * X[i]                                    # Tahmin edilen deger
        y = Y[i]                                                # Gercek deger


# Ne yone gidecegimiz kararini vermek icin;
        b_deriv_sum += (y_hat - y)                              # Sabit icin ,tahmini deger ile gercek deger farki alinip toplam ifadesine eklendi.
        w_deriv_sum += (y_hat - y) * X[i]                       # Agirlik icin
# ort. aliyoruz; 1 / m * b/w_deriv_sum
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

# Agirliklari update ettikten sonra tekrar cost_function'a bakabiliriz. :)




# train fonksiyonu / Iterasyon sayisinca islem yapmak icin;
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

# Ilk hatayi raporlamak istersem;
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b                                               # Verilen ilk agirliklari yeniden isimlendirdik.
    w = initial_w

    cost_history = []                                           # Her iterasyonda gozlemledigimiz hatalari listeye atayacagiz.

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)        # Agirliklari ve diger argumanlari ogrenme argumanina gore guncelledik.
        mse = cost_function(Y, b, w, X)                         # 1. Iterasyon dondukten sonra tekrar hataya baktik.
        cost_history.append(mse)                                # append ile mse'yi ekledik.


        if i % 100 == 0:                                        # Her 100'de 1 raporla diyoruz. (Iterasyon sayisi 100'e kalansiz bolunuyorsa 100'un katlaridir.)
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

# 1. iterasyondan sonra dongu devam ediyor... yeni agirliklar ve dolayisiyla yeni hatalar geliyor.
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


# PARAMETRE: Modelin veriyi kullanarak veriden harekerle buldugu degerlerdir; agirlikalar, veri setinden bulunur.
# HIPERPARAMETRE: Veri setinden bulunamayan veri kullanicilar tarafindan ayalranmasi gereken parametrelerdir.


# SORU: Normal denklemler yontemi ile Gradient Descent yontemi arasinda dogrusal reg. acisindan farklar?
# Normal denklemler yontemi ile direkt analitik olarak cozuyoruz.
# Gradient Descent, Optimizasyon yontemi, sürece bagli ggerceklesiyor.
# Gradient Descent'de ayarlanmasi gerekn hiperparametreler vardir. Normal denklemler yonteminde yoktur.


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000


train(Y, initial_b, initial_w, X, learning_rate, num_iters)
# Ciktiyi daha guzel gormek icin;
cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

# Baslangic raporunu ve her 100'de 1 raporlama islemini goruyoruz.
# Her iterasyonda hata orani dusuyor. :)
