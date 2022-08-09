# Random Forests, GBM, XGBoost, LightGBM, CatBoost

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)                    # Olasi uyarilar icin uyarilari kapattik :)

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]                                                           # Bagimli degsikeni,
X = df.drop(["Outcome"], axis=1)                                            # Bagimsiz degsikeni sectik.





## Random Forests

# Model nesnemizi getiriyoruz;
rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()
# 'max_features': Bolunmelerde goz onune alinacak degisken sayisi, 'n_estimators': Fit edilecek bagimsiz agac sayisi
# 'min_samples_split': Bir dugumun dallanmaya maruz kalip kalmayacagina karar vermek icin kac tane gozlem birimi olmasi gerektigi


cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Hiperparametre optimizasyonu oncesi hatalarimizi elde ettik.

# Hiperparametre setimizi giriyoruz;
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}
# Degerleri on tanimli degerlerin etrafindaki degerler olarak girdik,
# 'max_features' da veri setindeki degisken sayisindan daha buyuk deger yazarsak hata aliriz. :)
# On tanimli degerleri de eklememizin nedeni, calismanın basinda elde edilecek hatadan daha kotu bir hata elde etmemek.
# Hiperparametre optimizasyonu sonrasi hatamizin daha yuksek cikma sebebpleri;
# 1) Rastgelelikle ilgili olabilir,
# 2) On tanimli argumanlarin arama setinde olmamasi


# "GridSearchCV" ile hangi degerin daha iyi old. ariyoruz.
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# 180 tane olasi kombinasyon var; Bunlarin her biri icin 5 tane capraz dogrulama yapilacagindan dolayi toplam 900 tane fit etme islemi var.
# Basit ama tahmin basarisi yuksek modeller kurmaliyiz. :)

# En iyi parametreleri elde etmek icin;
rf_best_grid.best_params_
# Parametreleri set edip, "FINAL MODEL" olusturuyoruz;
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
 # Birden fazla metrige bakarak degerlendime yapilmali :)



# Modeli kurduk, uzerinde neler yapabiliriz;

# Feature importance;
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X, num=5)
# Tum degiskenleri gormek icin;
# plot_importance(rf_final, X)

# Validation Curve;
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

# Max derinlige gore bir degerlendirme yapmak istersek;
val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")
# AUC skorlarina gore grafigi olusturmus olduk.
# Islemin uzun surmesi normal; 500 tane agac fit ediliyor ve capraz dogrulama yontemiyle degerlendiriyoruz. :D

# Dallanma arttikca train setinde skorlari artiyor (ynai basarisi artmis gorunuyor)
# ama test setinde (Validation setinde ayni sey soz konusu degil. Bir noktadan sonra validasyon skoru azaliyor.



# Modeli kurduk, hiperparametre optimizasyonu yaptik simdi gelismis metodlari inceleyecegiz;






### GBM

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

# 'learning_rate': Sabit tahmini artiklarla guncelleme hizi
#  'n_estimators': Tahminci hızı. Buradaki aslinda optimizasyon sayisidir (boost etme sayisi)

# Hiperparametre optimizasyonu oncesi durumlara bakalim;
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7591715474068416
cv_results['test_f1'].mean()
# 0.634
cv_results['test_roc_auc'].mean()
# 0.82548

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
# "learning_rate": Ne kadar kucuk ise train suresi o kadar uzamaktadir ama kucuk olmasi durumunda daha basarili tahminler elde edilmektedir.
# "subsample": Baselearner fit edilecegi zaman kac tane gozlemin oransal olarak goz onunda bulundurulacagini ifade eder.
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametreler;
gbm_best_grid.best_params_

# Final model kuruyoruz;
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()






### XGBoost

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.75265
cv_results['test_f1'].mean()
# 0.631
cv_results['test_roc_auc'].mean()
# 0.7987

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}
# "colsample_bytree": Degiskenlerden alinacak gozlem sayisi
# On tanimli degerleri bu sefer yazmadik hem zamandan kazanmak, hem de uyari almamak icin. Aslinda yazmak gerekiyor. :)
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final modeli kuruyoruz;
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()






### LightGBM (Daha hizli :P)

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Tekrar benzer islem yaparak, nasil bir sonuc elde ederiz;
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Kayda deger bir degisiklik olmadi. Degerlendirmek icin yaptik. :)



# Hiperparametre optimizasyonu sadece n_estimators için.
# LGBM icin en onemli hiperparametre; n_estimators (tahmin sayisi) dir. Tahmin sayisi = Iterasyon sayisi = Boosting sayisi
# Diger parametreler icin uygun degerler bulunduktan sonra, tek basina n_estimators icin parametre deger degisimi yapilabilir. :)
# n_estimators icin deger 15ooo'lere kadar denenmeli !
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)
# colsample_bytree ve learning_rate' yi sabitledik.

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}
# Araliklari hassas girmekte bazi durumlarda avantaj saglar. :)

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Cok bir degisim olmadı. Cunku; veri setim, gozlem degerim, dolayisiyla dallanma az.






### CatBoost

# Catboost'un cirkin bir ciktisi var. "verbose=False" yapmayi unutma. :)
# Catboost'un calisma süresi uzun. :/

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

# Aranacak olan hiperparametre setimizi girelim;
cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Hiperparametre optimizasyonu yapmadigimiz halde LGBM'den daha iyi duruyor. :)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
# "iterations": Agac sayisi, boosting sayisi

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Daha onceki kullandigimiz yontemlerden daha yuksek bir AUC skoru geldi. :)

# Gelismis agac yontemlerini, yontemler uzerinde bitirmis bulunuyoruz. :))






## Feature Importance
# Tum modeller uzerinden degerlendirecek olursak;

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

# LGBM'de ilk uc degisti.
# Modelden modele grafiklerde importance degerlerinin degistigi goruluyor.






## Hyperparameter Optimization with RandomSearchCV (BONUS)

# GreadSearchCV: Verilen bir setin olasi butun kombinasyonlarini dener.
# Butun olasi kombinasyonlari denediginden dolayi daha uzun sürer ama en olasi en kapsama ihtimali daha yuksek.

# RandomSearchCV: Verilecek bir hiperparametre seti icerisinden rastgele secimler yapar ve bu rastgele secimleri arar.
# Daha fazla, daha genis bir hiperparametre adayi arasindan rastgele secim yapar, bu sectikleri uzerinden tek tek deneme yapar.

# GreadSearchCV uzun surdugunden RandomSearchCV tercih edilebilir.
# Soyle yapilabilir; RandomSearchCV ile bulunan optimum deger ve etrafina daha az sayida yeni degerler konularak GreadSearchCV islemi yapilabilir. :)


rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,                              # Denenecek parametre sayisi
                               cv=3,                                    # Uzun surmemesi icin 3 aldik. :)
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)


rf_random.best_params_


rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Uygun parametre degerlerini degil de bu degerleri de kapsayan daha genis araliktan yaptigimiz secim isleminin sonucu bunlar.






## Analyzing Model Complexity with Learning Curves (BONUS)

# Ogrenme egriligi ile model karmasikligini inceleme;

# Daha genis daha farkli parametre setleri vererek, AUC degerine gore ogrenme egrilerini olusturarak;
# model karmasikligini yani overfit olup olmadigimizi degerlendirmek istiyoruz;


# Fonksiyonumu tanimliyoruz;
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

# Bir liste icerisinde parametrelerimi ve bu parametreler icin denenecek olan degerleri giriyoruz;
rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]

# Bos model nesnemi olusturuyoruz;
rf_model = RandomForestClassifier(random_state=17)

# Fonksiyonu kullaniyoruz;
for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]
# "max_depth" icin; Max. derinlige gore train seti bir miktar basarini surdurup, test seti bastan itibaren azalarak devam etmis.
# Max. derinligin elde ettigimiz degerlere gore nerede olmasi gerektigiyle ilgili ufak bir bilgi elde ediyoruz.

# "max_features" icin; train skorunda ciddi bir degisiklik yok; AUC deger hep 1 civarinda...
# Fakat validasyon (test) skorunda cesitli degisiklikler var ama kayda deger bir farklilik yok.

# "min_samples_split" icin; train seti hemen reaksiyon vermis.
# Bolmelerde bulunacak gozlem sayisi arttikca train setinin AUC degeri dusmeye baslamis.
# Fakat validasyon setinin artmaya baslamis.
# Modelin genellenebilirligi test setindeki basariya gore bakildiginde min_samples_split yukseldikce, yukselmis gorunuyor.

# "n_estimators" icin; tahminci sayisi arttikca validasyon skorlarinda bir artma gozlemlenmis gibi gorunuyor.
# Train skorunda bir degisiklik yok.


# Hiperparametre optimizasyonu yaparak, es zamanli sekilde olasi tum hiperparametrelerin o olasi kombinasyonlarinin bir arada gozlenmesi senaryosundaki hatalara gore zaten secimimizi yaptik.
# Bunlari neden getiriyoruz?
# Bu secimlere karsi hangi noktada olabildigimizi bir de ogrenme egrileri uzerinden degerlendirerek bir ek bilgi elde ediyoruz.
