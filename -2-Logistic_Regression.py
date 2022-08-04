# Diabetes Prediction with LOGISTIC REGRESSION

# Makine ogrenmesi siniflandirma gorevini bastan sona yapacagiz;

# Is Problemi:

# Ozellikleri belirtildiginde kisilerin diyabet hastasi olup
# olmadiklarini tahmin edebilecek bir makine ögrenmesi
# modeli gelistirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Bobrek Hastaliklari Enstituleri'nde tutulan buyuk veri setinin
# parcasidir. ABD'deki Arizona Eyaleti'nin en buyuk 5. sehri olan Phoenix sehrinde yasayan 21 yas ve uzerinde olan
# Pima Indian kadinlari uzerinde yapilan diyabet arastirmasi icin kullanilan verilerdir. 768 gozlem ve 8 sayisal
# bagimsiz degiskenden olusmaktadir. Hedef degisken "outcome" olarak belirtilmis olup; 1 diyabet test sonucunun
# pozitif olusunu, 0 ise negatif olusunu belirtmektedir.

# Degiskenler
# Pregnancies: Hamilelik sayisi
# Glucose: Glikoz
# BloodPressure: Kan basinci
# SkinThickness: Cilt Kalinligi
# Insulin: Insulin
# BMI: Beden kitle indeksi
# DiabetesPedigreeFunction: Soyumuzdaki kisilere gore diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yil)
# Outcome: Kisinin diyabet olup olmadigi bilgisi. Hastaliga sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis          :Kesifci Veri Analizi
# 2. Data Preprocessing                 :Veri on isleme islemleri
# 3. Model & Prediction                 :Modelleme ve Tahmin
# 4. Model Evaluation                   :Model Basarisini Degerlendirme
# 5. Model Validation: Holdout          :Model Dogrulama Yontemlerini Degerlendirmek
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation   :Yeni Bir Gozlem Birimi icin Tahmin Islemleri


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression # Regresyon modelimiz
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve # Basari metriklerimiz
from sklearn.model_selection import train_test_split, cross_validate

# Kendi tanimladigimiz fonksiyonlar;
# Esik deger hesaplama;
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Esik degeri kullanarak aykiri deger hesaplama;
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Esik degeri kullanarak, bir degiskende aykiri deger varsa, silmeyip belirlenen esik degerlerle degistirme;
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)                      # Butun sutunlari goster
pd.set_option('display.float_format', lambda x: '%.3f' % x)     # Virgulden sonra 3 basamak goster.
pd.set_option('display.width', 500)                             # Konsolda gosterimi genis tut.



# Exploratory Data Analysis / Kesifci Veri Analizi (EDA);

df = pd.read_csv("datasets/diabetes.csv")
df.head()                                                       # 8 tane bagimsiz, 1 tane bagimli degisken (Outcome)
df.shape




# Target'ın Analizi / Bagimli Degiskenin Analizi;

df["Outcome"].value_counts()                                    # sinif sayisina baktik.
# 1 ve 0'lardan olusan kategorik degisken.

# Bagimli degiskenin sinif bilgileri (frekanslarini) sutun grafigi seklinde gormek istersek;
sns.countplot(x="Outcome", data=df)
plt.show(block=True)

# Sinif frekanslarinin oranini bulmak icin;
100 * df["Outcome"].value_counts() / len(df)                    # Hangi siniftan hangi oranda var, bilgisini elde ettik.




# Feature'ların Analizi / Bagimsiz Degiskenlerin Analizi;

df.head()
df.describe().T
# "describe()" sadece sayisal degiskenlerin durumunu ozetler.

# Gorsel olarak gormek istersek;
# (Sayisal degiskenleri gorsellestirmek icin, kutu grafik ve histogram kullanilir.)
# Hist.: Ilgili sayisal degiskenin degerlerini belirli araliklarda ne kadar gozlenme frekansi var onu gosterir.
# Kutu Graf.: Ilgili sayisal degiskenin degerlerini kucukten buyuge siraladiktan sonra degiskenin dagilimi ile ilgili bilgi verir.

# Yas degiskeninin histogrami;
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show(block=True)


# Bu islemi her degisken icin tekrarlamak istemiyoruz;

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

# Butun sayisal degiskenler icin gorsellestirme islemi gerceklesti.
# Bagimli ve bagimsiz degiskenlerimiz sayisal oldugu icin tum degiskenler icin yapti.

# Outcome; Bagimli degiskenimi disarida birakmak istiyorum;
cols = [col for col in df.columns if "Outcome" not in col]


for col in cols:
    plot_numerical_col(df, col)

df.describe().T




# Target vs Features / Bag.li ve Bag.siz degiskenleri birlikte degerlendirirsek;

# Target'a gore groupby'a alip sayisal olan degiskenlerin ortalamasini aliyoruz;
df.groupby("Outcome").agg({"Pregnancies": "mean"})

# Fonksiyonlastirirsak;
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)




# Data Preprocessing (Veri On Isleme);
df.shape
df.head()

df.isnull().sum()                               # Ilgili butun degiskendeki butun eksik degerler.
# Herhangi bir eksik deger yok.

df.describe().T
# Bazi degerlerin 0 olmasi anlamsiz:). Demek ki bazi eksik degerler sifirlarla degistirilmis.
# Su an icin eksik deger yokmus gibi devam edecegiz.

# Esik degerlere gore aykiri degerleri yakaliyoruz;
for col in cols:
    print(col, check_outlier(df, col))
# Sadece "Insulin" degiskeninde aykiri deger cikti.

# Aykiri degerler yerine Insulin icin hesaplanan esik degerleri atanir;
replace_with_thresholds(df, "Insulin")

for col in cols:
    print(col, check_outlier(df, col))
# Butun degiskenlerdeki aykirilik durumunu cozmus olduk.


# Degiskenleri olceklendirmek (Scale) etmek icin;

# "RobustScaler()": Butun gozlem biriminin degerlerinden medyani cikarip, range degerine boluyor.
# StandartScaler'dan farki; aykiri degerlerden etkilenmiyor olmasi.Cunku ortalama degil, medyan cikariliyor. :)
# ve Standartscaler gibi standart sapmaya bolmuyor; range degerine boluyor.

# (Medyan, aykiri degerlerden etkilenmez ama ortalama etkilenir.)

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


# Neden standartlastirma islemi yapiyoruz;
# 1) Modellerin degiskenlere esit yaklasmasini saglamak icin.
# (Degerleri daha buyuk olan degiskenin daha kucuk olana ustunlugu olmadigini ifade etmemiz gerekiyor.)
# 2) Kullanilan parametre tahmin yontemlerinin daha hizli ve daha dogru tahminlerde bulunmasi icin.





# Model & Prediction

# Amacimizİ Kisilerin ozellikleri verildiginde diyabet olup olmadiklarini tahmin etmek;

y = df["Outcome"]                                   # Bagimli degisken

X = df.drop(["Outcome"], axis=1)                    # Bagimsiz degiskenleri tanimlamak icin bagimli degiskenimizi cikardik.

# Ikisi arasindaki iliskiyi modelliyoruz;
log_model = LogisticRegression().fit(X, y)          # Logistik Reg. modelimizi kurduk.


log_model.intercept_                                # Modelin sabiti
log_model.coef_                                     # Bagimsiz degiskenlerin agirlik degerleri

# Bu modeli kullanarak tahmin etme islemini gerceklestirelim;
y_pred = log_model.predict(X)                       # Bagimsiz degiskenlerden bagimli degiskeni tahmin etti.

y_pred[0:10]                                        # Tahmin degerleri

y[0:10]                                             # Gercek degerler





# Model & Prediction / Model ve Tahmin;

y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)
y_pred[0:10]

y[0:10]





# Model Evaluation / Model Basari Degerlendirme;

# Karmasiklik matrisini isi haritasi kullanarak gorsellestirmek icin;
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)            # Numerik karsiliklar
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show(block=True)

plot_confusion_matrix(y, y_pred)

# Bu hesaplamalarin otomatik yapildigi fonksiyonumuz;
print(classification_report(y, y_pred))
# supoort: Ilgili siniflarin frekanslari
# 1 sinifina gore basari metriklerini degerlendiriyoruz (Ana odak noktamiz).
# Gozlem birimi sayilarina gore aritmetik ort.(macro avg) ve agirlikli ort.(weighted avg)


# Accuracy: 0.78 : (156 + 446) / (446 + 54 + 156 + 112) = Dogru siniflandirma orani
# Precision: 0.74 : 156 / (156 + 54) = 1 olarak yaptigimiz tahminler ne kadar basarili?
# Recall: 0.58 : 156 / (156 + 112) = 1 olanlari ne kadar basarili tahmin ettik?
# F1-score: 0.65

# ROC AUC
# (Farkli classification_threshold degerlerine gore olusabilecek basarilimiza gore genel bir metrik)

# Metrigi hesaplamak icin bagimli degiskenin 1 sinifinin gerceklesme olasiliginin tahmin degerlerine ihtiyacimiz var;
y_prob = log_model.predict_proba(X)[:, 1]                                      # 1 sinifinin gerceklesme olasiliklari
roc_auc_score(y, y_prob)
# 0.83939 ; auc skorumuz :)

# Modeli tum veri seti uzerinden (Modelin ogrenildigi veri uzerinde) test ettik.
# Ayni veri uzerinden hem modeli kurup hem de modeli test ettigimden; dogrulanmaya ihtiyacim var.




# Model Validation/ Model Dogrulama : Holdout;

# Veri setini train ve test olarak ikiye boluyoruz;

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)
# random_state ile rastgeleligi bir yere sabitliyorum.

# Train setine modeli kuruyoruz;
log_model = LogisticRegression().fit(X_train, y_train)

# Test setini modele soruyoruz;
y_pred = log_model.predict(X_test)                          # Tahmin degerleri
y_prob = log_model.predict_proba(X_test)[:, 1]              # AUC hesabi yapabilmek icin 1 sinifina ait olma olasiliklarini buluyoruz.
# Bagimsiz degisken degerlerini gonderdik buna karsilik olasilik degerlerini elde ettik.

# Basarimizi degerlendirelim;
print(classification_report(y_test, y_pred))

# y_test: Test setindeki y degerleri
# y_pred: Tahmini y degerleri

# Onceki degerlerimiz;
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Simdiki degerlerimiz;
# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

# Onceki ile cok buyuk farkliliklar yok fakat model gormedigi verilere dokununca tahmin degerleri degisti.
# Daha basarisiz gibi duruyor. Anliyorum ki; Bir model dogrulama islemi gercekten gerekiyor.


# ROC  CURVE cizdirirsek;
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show(block=True)

# AUC
roc_auc_score(y_test, y_prob)

# Onceki AUC degerim; 0.83
# Simdiki AUC degerim; 0.87





# Rastgelelige gore (degerler kucuk de olsa degisiyor) olusan olasi farklilikların goz onunde bulundurulmasi icin;

# Model Validation: K-Fold (K-Katli) Cross Validation;

# Veri setin 10 parcaya bolunur. 9 tanesi ile model 1 tanesi ile test yapilir.
# Bu islem farkli kombinasyonlar icin 10 kez tekrarlanir ve butun test hatalarinin (validasyon hatalarinin) ortalamasi alinir.
# Boylece model, veri setinin farkli parcalari ile moddellenip; farkli parcalariyla test edilmis olacagindan;
# olasi bazi senaryolari da barindirir.

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)            # Butun veriyi kullanarak bu islemi gerceklestirdik.
# Bu noktada su iki yol tercih edilebilir;
# 1) Veri seti bol ise; veri setini train-test diye ikiye ayirip, train setine 10 katli validasyon yapip,
# en sonda test seti perfonmansi incelenebilir.
# 2) Veri seti bol degilse yani cok genis bir orneklemimiz yoksa; butun veriyi kullanarak da capraz dogrulama islemini gerceklestirebiliriz.

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,                     # 5 Katlı Cross Validation yapacagiz.
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# "cross_validate" : Birden fazla basari metrigimizi hesaplamamizi saglar.

# Butun veriyle yaptigimiz ama model dogrulama yontemlerine geldigimiz kisim;
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Model dogrulama yontemine gelip holdout yontemini kullandigimiz kisim;
# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


# K-Katli Capraz Dogrulama yaptiktan sonra elde ettigimiz degerler; (Daha dogru sonuclar :))
cv_results['test_accuracy']                 # Test skorlarimiz; Kombinasyonlar sonucu ortalama accurcy'ler
cv_results['test_accuracy'].mean()          # Ortalamasini alarak, basari metrigimi elde etmis olduk.
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

# Dengeli veri setimiz olsaydi sadece accuracy'e bakmak yeterdi.
# Dengeli degilse accuracy' e ek F1 skora bakmak da yarar var. :)




# Prediction for A New Observation;

X.columns # Bagimsiz Degiskenlerimiz

# X'in icerisinde rastgele bir kullanici seciyoruz;
random_user = X.sample(1, random_state=45)
log_model.predict(random_user) # Bu rastgele kullanici icin tahminimiz
