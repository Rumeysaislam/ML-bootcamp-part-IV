################################################
# Decision Tree Classification: CART
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART                                        / CART kurulmasi
# 4. Hyperparameter Optimization with GridSearchCV              / Parametre optimizasyonu
# 5. Final Model
# 6. Feature Importance                                         / Degisken onem duzeyleri
# 7. Analyzing Model Complexity with Learning Curves            / Model karmasikliginin ogrenme egriligiyle incelenmesi
# 8. Visualizing the Decision Tree                              / Karar agacinin gorsellestirilmesi
# 9. Extracting Decision Rules                                  / Karar kurallarinin cikarilmasi
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model                                  / Modeli saklamak ve yuklemek


# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings                                 # Olasi bazi uyarilari ignore etmek icin import ettik.
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz                                 # Excel, python ve sql formunda bazi ciktilar verir.


pd.set_option('display.max_columns', None)      # Tum sutunlari gostermek icin ayar yaptik.
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)


df = pd.read_csv("/home/rumeysa/Desktop/Miuul_summercamp/5.Hafta/machine_learning/machine_learning/datasets/diabetes.csv")

### 1. Exploratory Data Analysis

# GENEL RESİM;

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)



# NUMERİK VE KATEGORIK DEGISKENLERIN YAKALANMASI;

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


## KATEGORIK DEGISKENLERIN ANALIZI;

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, "Outcome")



## NUMERIK DEGISKENLERIN ANALIZI;

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)



## NUMERIK DEGISKENLERIN TARGET GORE ANALIZI;

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)



## KORELASYON;

df.corr()

# Korelasyon Matrisi;
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)



## BASE MODEL KURULUMU;

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)




### 2. Data Preprocessing & Feature Engineering

## EKSIK DEGER ANALIZI;

# Bir insanda Pregnancies ve Outcome disindaki degisken degerleri 0 olamayacagi bilinmektedir.
# Bundan dolayi bu degerlerle ilgili aksiyon karari alinmalidir. 0 olan degerlere NaN atanabilir .
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# Gozlem birimlerinde 0 olan degiskenlerin her birisine gidip 0 iceren gozlem degerlerini NaN ile degistirdik.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])


# Eksik Gozlem Analizi;
df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)


# Eksik Degerlerin Bagımlı Degisken ile İliskisinin İncelenmesi;
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)


# Eksik Degerlerin Doldurulmasi;
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()



## AYKIRI DEGER ANALIZI;

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykiri Deger Analizi ve Baskilama Islemi;
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))




## OZELLIK CIKARIMI;

# Yas degiskenini kategorilere ayırıp yeni yas degiskeni olusturulması
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI 18,5 asagısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 ustu obez
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glukoz degerini kategorik degiskene cevirme
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# # Yas ve beden kitle indeksini bir arada dusunerek kategorik degisken olusturma
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# Yas ve Glikoz degerlerini bir arada dusunerek kategorik degisken olusturma
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# Insulin degeri ile kategorik degisken turetmek;
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

# Kolonların buyultulmesi;
df.columns = [col.upper() for col in df.columns]

df.head()
df.shape



## ENCODING

# Degiskenlerin tiplerine gore ayrilmasi islemi;
cat_cols, num_cols, cat_but_car = grab_col_names(df)



## LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()



## STANDARTLASTIRMA;

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape



## ONE - HOT ENCODING

# cat_cols listesinin guncelleme islemi;
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape
df.head()




### 3. Modeling using CART

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)               # CART modeli kuruldu.

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)                                              # Butun gozlemler icin tahmin edilen degerleri hesapladik.

# AUC için y_prob: ROC egrisi icin AUC hesaplamamiz gerekiyor.
y_prob = cart_model.predict_proba(X)[:, 1]

## Confusion matrix
print(classification_report(y, y_pred))
# Teorik olarak bu kadar dogru bir olcum mumkun degil; rastlantisal bir hata olmasi lazim. (1.00)


## AUC
roc_auc_score(y, y_prob)
# Modelim asiri ogrenme mi yasadi yoksa cok mu basarili (Tum veri seti uzerinden degerlendirdik)?



## Holdout Yontemi ile Basari Degerlendirme;      (Veri setini, train ve test olarak ayiriyoruz.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)


## Train Hatası
y_pred = cart_model.predict(X_train)            # Train setinin bag.siz degiskenlerini sorup, tahmin edilen degerleri aldik.
y_prob = cart_model.predict_proba(X_train)[:, 1]# Train setinin bag.siz degiskenlerini sorup, AUC degerleri icin olasiliklari aldik.
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
# Train setimdeki basarim 1.00 cikti.


## Test Hatası
y_pred = cart_model.predict(X_test)             # Modele, Modelin hic gormedigi test setinin bag.siz degisken degerlerini gonderiyoruz.
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
# Modeli teste soktugumuzda butun metrik deherleri degisti ve basari orani azaldi.
# Test setinde bu kadar hata yapip, train setinde cok basarili olmasi; modelin overfit old. yani asiri ogrendigini gosterir. :)

# (Random state (rastgelelik) degisince de bu basari metrikleri degisir.)

# Olusan soru isaretleri acisindan capraz dogrulama ile model basarisi sorgulama daha iyi olacak.
# Veri seti az oldugunda haldout sendeler. :)



## CV ile Basari Değerlendirme;
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

# Modeli olustururken ".fit(X, y)" yapmasak olurdu ama zaten "cross_validate", fit islemini gormezden gelecek.

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,                                       # 5-Katli capraz dogrulama yapiyoruz.
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7058568882098294
cv_results['test_f1'].mean()
# 0.5710621194523633
cv_results['test_roc_auc'].mean()
# 0.6719440950384347

# Elde ettigimiz en gecerli basarilar bunlardir. :)
# Cunku; Basari degerlendirmesinin yuksek olmasi degil, dogrulama kaygisi barindirmamasidir. :)

# Model basarimizi;
# Yeni gozlemler
# Yeni degiskenler ekleyerek
# Veri on isleme islemlerine dokunarak
# Hipermarametre optimizasyonu yaparak
# Dengesiz veri yaklasimi kullanilarak arttirilabilir.

# Dengesiz veri yaklasimi : Bagimli degiskendeki siniflarin dagililimlarinin cok farkli olmasi durumunda;
# azaltarak arttirarak, rastgele orneklem yontemleri gibi yontemler kullanarak, dengesizligin giderilmeye calisilmasi.




### 4. Hyperparameter Optimization with GridSearchCV
# ( CART yontmei icin hiperparametre optimizasyonu yapip, basariyi arttirmak istiyoruz.)


cart_model.get_params()                              # Mevcut model parametreleri
#  " 'min_samples_split': 2 ": Bolme islemini direkt etkileyen dolayisiyla asiri ogrenmey etkileyen dissal parametre
# 2 tane kalana kadar bolme durumunu ifade ediyor.

# 'max_depth': Asiri ogrenmenin onune gecebilecegimiz diger parametre

# Sozluk olusturarak denenecek parametreleri, on tanimli degerinden baslayacak sekilde gireriz;
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# Uygun paarametre icin arama islemi yapiyoruz;
cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,               # Islemcileri tam performans kullanma ayari
                              verbose=1).fit(X, y)     # "verbose": rapor istedigimizi belirtir.
                                                       # "verbose=2" yapsaydik, daha detayli rapor elde ederdik.
# Iki parametre icin olasi 180 tane parametre var 5 katli capraz dogrulama yapip hata hesaplama isleminde 900 tane fit islemi varmis.


cart_best_grid.best_params_
# 'max_depth': 5, 'min_samples_split': 4

# En iyi degerlere karsilik, en iyi skor;
cart_best_grid.best_score_
# GridSearchCV icerisindeki "scoring" icin on tanimli deger accuracy' dir. scoring=f1, roc_ouc de yapilabilirdik.


# Rasgele gozlem birimi seciyoruz;
random = X.sample(1, random_state=45)
# Degiskenleri standartlastirmadik. Cunku; agac modellerde standartlastirmaya gerek yok. :)

# Arama islemleri sonucunda bulunan en iyi parametreleri veren metod "GridSearchCV" dir; metodu bir modeli kurar (fit eder) zaten.
# Biz bu modeli degil, parametreleri alip kuracagimiz (fit edecegimiz) final model uzerinden devam edecegiz. :)

cart_best_grid.predict(random)





### 5. Final Model

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

# Bir diger en iyi parametreleri modele atama yontemi ("set_params");
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)         # En uygun parametreleri modele set edip, fit ediyoruz.

cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7500806383159324
cv_results['test_f1'].mean()
# 0.614625004082526
cv_results['test_roc_auc'].mean()
# 0.797796645702306
# Butun metriklerde ilerleme elde ettik. :)





### 6. Feature Importance / Degisken Onem Duzeyleri;
# SSE degerimizi azaltan degisken, onemli degiskenimizdir. :)

cart_final.feature_importances_                     # Degiskenlerin onem duzeylerini cagirdik.

# Modeldeki degiskenlerin onem duzeylerini gorsellestiren fonksiyonumuz;
def plot_importance(model, features, num=len(X), save=False):                                           # Argumanlarimiz; model, degiskenler, gorsellestirilecek degisken sayisi, gorseli kaydetme durumu

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})      # feature_imp: degiskenlerin isimlerini ve importance skorlarini (model.feature_importances_) tasiyor.
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])           # feature_imp'i kucukten buyuge siralandiktan sonra boxplot ile gorsellestiriyoruz.
                                                                                                        # 0:num; veri setindeki degisken sayisi kadar gorsellestir bilgisini tasiyor On tanimli degeri :num=len(X) .
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:                                                                                            # Gorseli kaydetmek istedigimiz durum; save=True
        plt.savefig('importances.png')


plot_importance(cart_final, X, num=5)





### 7. Analyzing Model Complexity with Learning Curves / Öğrenme Eğrileriyle Model Karmaşıklığını Analiz Etme

# Overfit'e dustugumuzu, train seti ile test setinin hata farklarinin ayrismaya basladigi noktaya bakarak anlariz.
# Bu durumun onune gecmek icin model karmasikligi azaltilir. Optimum noktada dururuz.
# Model karmasikligina sebep olan parametre modelden modele gore degisebilir.


# valisation_curve metodunu kullanarak sectigimiz parametreye gore numerik ciktilar elde edip ogrenme egrilerini gormek gorsellestirecegiz;
train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),                # Deneyecegi derinlik araligini verdik.
                                           scoring="roc_auc",                       # Hangi metrige gore raporlayacagini soyledik.
                                           cv=10)
# Test ve train skorlari kaydedildi. max_depth' in 10 tane farkli degeri oldugundan; 10 tane array goruyoruz.
# Array'lerin icindekiler, 10 katli capraz dogrulama sonuclari. 9 parca ile kurup 1 aile test ediyoruz.


# Ilgili parametre degeri icin ortalama cross validation AUC degeri hesaplanir.
mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)
# Ortalamayi almamiza ragmen 10 tane deger olmasinin sebebi; her bir parametre icin ortalamanin ayri alinmis olmasi.


# Train seti hatasi ile test seti hatasi gorsellestirilir ve ayrim noktalari uzerinden karar verilir;
plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,                 # Train-Test ile Train-Validation ifadeleri birbirine denktir. Validation islemi yaptigimiz icin bu ifadeyi kullandik.
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show(block=True)

# Train basarisi artmaya devam ederken, test basarisi 3'ten sonra azalmaya basliyor.
# Modeldeki dallanma arttikca genellenebilirlik ozelligi kaybedilmeye baslanmis.

# Yorum : max_depth icin en uygun parametre degerini 5 bulmustuk. 3 mü olmaliydi?
# Model karmasikliginda degiskeni tek basina degerlendirdigimiz icin bu sekilde farklilik oldu. Yorumlayabilmek, degerlendirebilmek icin (Bulunan en uygun parametreyyle farkina) bakiyoruz.
# Bu grafige bakarak hiperparametre degerimiz üc olsun diyemeyiz.
# Zaten en uygun parametreyi eş anli olarak (Tum parametrelerin degerleri goz onune alinarak) bulduk.



# Yaptigimiz islemleri fonksiyonlastirmak istersek;
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")           # Dinamik sekilde model ismini aliyoruz.
    plt.xlabel(f"Number of {param_name}")                               # Eksen isimlerini de manuel girmek yerine dinamik olarak aliyoruz.
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

# scoring icin on tanimli degeri degistirirsek;
val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")


# Birden fazla hiperpametre seti oldugunda gorsellestirme islemini pratik sekilde yapmak icin;
cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])





### 8. Visualizing the Decision Tree / Karar Agacini Gorsellestirme

# conda install -c anaconda graphviz
import graphviz

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)                                  # Kurdugumuz modelin png dosyasi ciktisi alacagiz.


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")
# Gorselde 5 seviye oldugunu goruyoruz.

cart_final.get_params()
# Parametreleri cagirdigimizda max_depth degerinin:5 oldugunu. Dallanma sayisinin 5 oldugunu gorduk.

# max_depth degeri girmezsek ya da on tanimli degeri None ile birakirsak max_samples_split=2'ye kadar (On tanimli deger:2 gozlem kalana kadar) dallanmaya devam eder. :)





### 9. Extracting Decision Rules / Karar Kurallari Cikarimi
# Karar agazlari belirli karar kurallari uretiyor. Gorsel olarak gorduk, teorik olarak nasil görecegimiz;

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)
# Karar kurallarini console'da gozlemledik.





### 10. Extracting Python Codes of Decision Rules / Karar Kurallarinin Python Kodlarini Cikartma

# Bir agac yonteminin karar kurallarini cikarip, bir sql (ya da python) sorgusu haline getirebilirsek modeli canliya alma (deployment) islemini gerceklestirmis oluruz. :)

# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1

import sklearn
# Versiyon kontrolumuzu yapmak icin;
sklearn.__version__


print(skompile(cart_final.predict).to('python/code'))            # Python kodlarini cikariyoruz.

# Alinan cikti: Gorsel tekniklerle elde ettigimiz karar agacimizin fonksiyonlastirilabilecek olan karar kurallaridir.
# Fonksiyon tanimlayip degerler sordugumuzda tahminler elde edecilecegiz.


# pip install sqlalchemy
print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))     # Sqlite kodlarini cikariyoruz.


print(skompile(cart_final.predict).to('excel'))                 # Excel kodlarini cikariyoruz.





### 11. Prediction using Python Codes / Python Kodlari ile Tahmin Etme

def predict_with_rules(x):
    return (((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else 0 if x[6] <=
    0.5005000084638596 else 0) if x[5] <= 45.39999961853027 else 1 if x[2] <=
    99.0 else 0) if x[7] <= 28.5 else (1 if x[5] <= 9.649999618530273 else
    0) if x[5] <= 26.350000381469727 else (1 if x[1] <= 28.5 else 0) if x[1
    ] <= 99.5 else 0 if x[6] <= 0.5609999895095825 else 1) if x[1] <= 127.5
     else (((0 if x[5] <= 28.149999618530273 else 1) if x[4] <= 132.5 else
    0) if x[1] <= 145.5 else 0 if x[7] <= 25.5 else 1 if x[7] <= 61.0 else
    0) if x[5] <= 29.949999809265137 else ((1 if x[2] <= 61.0 else 0) if x[
    7] <= 30.5 else 1 if x[6] <= 0.4294999986886978 else 1) if x[1] <=
    157.5 else (1 if x[6] <= 0.3004999905824661 else 1) if x[4] <= 629.5 else 0
    )                                           # Yukarida elde ettigimiz python kodunu bir fonksiyon ile tanimlamis olduk.

X.columns                                       # Degisken isimleri

x = [12, 13, 20, 23, 4, 55, 12, 7]              # Yeni gelen hastanin bilgileri bunlar olsun dedik.

predict_with_rules(x)                           # Hasta bilgilerini girdik ve tahmin sonucu geldi.

x = [6, 148, 70, 35, 0, 30, 0.62, 50]

predict_with_rules(x)





### 12. Saving and Loading Model
#Kurdugumuz modeli daha sonra kullanmak istiyoruz;

joblib.dump(cart_final, "cart_final.pkl")               # Dosyayi pkl formatta "dump" ile kaydediyoruz.

cart_model_from_disc = joblib.load("/home/rumeysa/PycharmProjects/pythonProject/CART/cart_final.pkl")
# Daha once kaydettigimiz modeli tekrar yukleyip yeni isim verdik.


x = [12, 13, 20, 23, 4, 55, 12, 7]

# "predict" metodunu kullandigimizdan dolayi dataframe'e cevirmemiz gerekiyor.
cart_model_from_disc.predict(pd.DataFrame(x).T)         # DataFrame'in transpozunu aldik.




# Basit, hizli el degistiren, veri tabanina yakin cozumler her zaman avantajlidir. :)
# Mumkun oldugunca SQL ve sunucu ortamlarinda kalmaliyiz.
