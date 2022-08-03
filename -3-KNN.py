################################################
# KNN
################################################

# 1. Exploratory Data Analysis                   / Veriyi tanima asamasi
# 2. Data Preprocessing & Feature Engineering    / Eksiklikler, aykirliklik gideriliz, duzenlemeler yapilir ve yeni degiskenler uretilir.
# 3. Modeling & Prediction                       / Model ve tahmin yapma asamasi
# 4. Model Evaluation                            / Model basarisini olcme asamsi
# 5. Hyperparameter Optimization                 / Knn yonteminin dıssal parametresini (hiperparametresi) optimize etme asamasi
# 6. Final Model                                 / Optimize edilen parametrelerle final modeli kurma asamasi

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



# 1. Exploratory Data Analysis;

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts() # Bag.li degiskenin dagilimi




# 2. Data Preprocessing & Feature Engineering;

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Daha dogru sonuc elde etmek icin elimizdeki bag.siz degiskenleri standartlastime islemine sokuyoruz.
X_scaled = StandardScaler().fit_transform(X)
# Cikti np.array seklinde ama sutun isimlerini de gormek istiyoruz;
X = pd.DataFrame(X_scaled, columns=X.columns)
# Standartlastirilmis sekilde bag.siz degisken degerleri geldi.




# 3. Modeling & Prediction;

# Tum veri seti icin basariyi gormek isteiyoruz;

knn_model = KNeighborsClassifier().fit(X, y)           # Knn modeli fit ederek modeli kurduk.
                                                       # Bag.li deg. ile bag.siz degisken arasindaki iliskiyi ogrenmis olduk.

random_user = X.sample(1, random_state=45)             # Rastgele orneklem seciyoruz.

# Kurmus oldugumuz modele, rastegele orneklemin diabet olup olmadigini soralim;
knn_model.predict(random_user)

 # Bir modeli egitmek ayri bir is, ogrenilenden yola cikarak tahmin etmek ayri bir is. :)



 
# 4. Model Evaluation;

# Confusion matrix için y_pred bulunur:
y_pred = knn_model.predict(X)                          # Knn modelini kullanarak butun gozlem birimleri icin tahmin yapıp sakliyoruz (y_pred).

                                                       # AUC için y_prob bulunur ( Olasilik degerleri uzerinden bulunur).
                                                       # AUC : ROC egrisinin altinda kalan alan

y_prob = knn_model.predict_proba(X)[:, 1]              # 1 sinifina ait olma olasiliklarini sectik
                                                       # Bu olasiliklar uzerinden ROC egrisi skoru hesapliyor olacagiz.

print(classification_report(y, y_pred))                # 1 ve 0 sinifina gore hesaplama islemi yapiyor.
                                                       # 1 sinifina odaklaniyoruz.
                                                       # acc 0.83; Basari durumumuz.
                                                       # f1 0.74
                                                       # AUC
# Dengesiz veri seti varsa accurcy her zaman dogru sonuc vermez. Baska metriklere de bakmak gerekir;
# precision, recall ve harmonik ortalamalari olan f1-score
# precision: 1 sinifina yonelik tahmin ettiklerimizin basarisi
# recall: Gercekte 1 olanlari, 1 olarak tahmin eteme basarimiz
# accurcy = 0,83 ama gercekte bir olanları tahmin etme olasiligimiz o kadar yuksek degil (0,70).
# Baktigimizda tahmin oranlarimiz %70 uzerinde; bu model basarili sayilabilir.


roc_auc_score(y, y_prob)
# 0.90 oldukca yuksek bir deger cikti.

# Simdiye kadar tum veriyle model kurup, modelin basarisini yine ayni veriler uzerinden degerlendirdik.
# Aslinda amacimiz modeli, modelin gormedigi verilerle test etmektir;
# Bunun icin hold-out veya cross validation (hold-out'un dezavantajlarini ortadan kaldirmak icin) kullanilir.


# cross validation kullanarak 5 katli capraz dogrulama hatamizi degerlendirecek olursak;
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# "scorin=[...]": kullanmak istedigim metrikler
# score_time: prediction (tahmin) suresi
# 5 kali yaptigimiz icin veri setini bolup, 4' u ile train 1'i ile test setini kurdu.
# Farkli kombinasyonalara gore cikan (5'er tane) sonuclari goruyoruz.


cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Bu degerler daha guvenilir.
# 0.73
# 0.59
# 0.78

# Tahmin oranlarinin degerlerinin dustugunu goruyoruz.
# Cunku modeli olusturgun veri seti ile modelin basarisi test etmek yanliliga sebep olacagindan basari oranlari yuksek cikar.




# BASARI SKORLARI NASIL ARTIRILABILIR?

# 1. Örnek boyutu artirilabilir.
# 2. Veri ön işleme (islemleri detaylandirilabilir.)
# 3. Özellik mühendisliği (Yeni degiskenler turetilebilir.)
# 4. İlgili algoritma için optimizasyonlar yapılabilir.


# KNN modelin komsuluk sayisi hiperparametresi (dissal parametresi) var, degistirilebilirdir.

knn_model.get_params() # Knn modelin parametrelerini cagirmis olduk.

# Parametre: Modellerin veri icerisinden ogrendigi agirliklardir.
# Hiperparametre: Kullanici tarafindan tanimlanmasi gereken dissal ve veri seti icerisinden ogrenilemeyen parametrelerdir.




# 5. Hyperparameter Optimization;

knn_model = KNeighborsClassifier()
knn_model.get_params()
# On tanimli komsuluk degeri 5. Amacim en uygun komsuluk degerini bulmak;

knn_params = {"n_neighbors": range(2, 50)}                        # Bir sozluk olusturuyoruz. Isimlendirmeye dikkat !
                                                                  # Isımlendirme fonksiyon icerisindekiyle ayni olmali; ""n_neighbors"

# "GridSearchCV" : Programatik olarak komsuluklara gore knn modeli grup uygun olanini bulacak.
# "cv=5" : Hataya 5 katli bak demek istedik.
# "n_jobs=-1" : Islemciler olasi en hizli sekilde kullanilir. Daha hizli sonuclara ulasmak icin. :)
# "verbose=1" : Yapilan sonuclar icin rapor istedigimi belirtiyorum.


knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)
# Yorum;
# 48 tane denenecek parametre var.
# Her parametre icin 5 katli CV yapilacagindan dolayi toplam 240 tane fit etme (model kurma) islemi vardir.



knn_gs_best.best_params_                                         # Buluna uygun komsuluk degerini cagiriyoruz; 17 :)
# On tanimli deger (5) ile değil de komsuluk sayisini 17 yaparsam; final modelin daha basarili olmasini beklerim. :)




# 6. Final Model;

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
# 'n_neighbors': 17 ' yi el ile yazmak yerine otomatik atamak için "**knn_gs_best.best_params_" yazdik. (**)


cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Skor degerlerin, on tanimli degerlerle yapilan hesaplamara gore arttigini goruyoruz. Modelimiz artik daha basarili... :)

random_user = X.sample(1) # Rasgtele hasta secme islemi yaptik.
# Degisken degerlerini onceden standartlastirmistik. :)

knn_final.predict(random_user) 
# Rasgele secilen hastanin diabet olup olmadigini tahmin ettik. :)
