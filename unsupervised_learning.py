
# Unsupervised Learning / DENETIMSIZ OGRENME

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder


## K-Means

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
# Eyaletleri segmentlere ayirmak; kumelemek istiyoruz;

df.head()
df.isnull().sum()
df.info()                           # Segmentlere ayirmak istedigimiz 50 tane gozlem birimi var.
df.describe().T
# Uzaklik temelli yontemlerde degiskenlerin standartlastirilmasi onem tasiyor. :)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]
# fit_transform sonrasi numpy array oldugundan dolayi head ile bakmadik ilk bes gozleme.

# Model kuruyoruz (ama denetimsiz ogr. olg. yani bagimli deg. olmadigini unutma :) );
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)      # x'imiz bagimli degiskenlerimizi barindiran df
kmeans.get_params()

# n_cluster, disaridan belirlenmesi gereken onemli bir parametre.


kmeans.n_clusters
kmeans.cluster_centers_                                     # Belirlenen dort kumenin, cluster merkezleri (standartlastirilmis degerdekş gozlem birimleri)
kmeans.labels_                                              # Her bir gozlemin KMeans tarafindan belirlenen kume etiketleri (0: ilk kume, 3: son kume)
kmeans.inertia_                                             # En yakın cluster'a olan uzakliklar



# Optimum Kume Sayisinin Belirlenmesi

kmeans = KMeans()               # Bos bir Kmeans nesnesi olusturduk.
ssd = []                        # Bos bir liste olusturduk.
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)           # Araliktaki butun k degerlerini girecek ve fit etme islemi yapacak ve
    ssd.append(kmeans.inertia_)                     # Inertia degerlerini SSD icerisine gonderecek.

# SSD degerlerini yorumlamayi kolaylastirmak icin gorsel olusturuyoruz;
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show(block=True)
# Kume sayisi artikca SSD=SSR sayilari artiyor. Kume sayisi arttikca hata duser.
# Gozlem sayisi kadar kume sayisi secseydik SSD=0 olurdu.
# K-means yontemi kume sayisi onerisi yapar ama kesin sonucu K-means'e bakarak soylemeyiz. :)
# Grafikten bir secim yapsaydik; 5-direklenmenin en fazla old. (egimin) nokta olacagindan onu secerdik. :)


# Daha optimum bir yol denersek;
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
# Optimum kume sayisini kendi belirliyor (5). :)
# Distorcion score = Distance

# Degeri gormek istedigimde;
elbow.elbow_value_                                  # Optimum kume sayimiza ulasmis olduk. Aslinda hiperparametre optimizasyonu yapmis olduk. :)



# Final Cluster'larin Olusturulmasi;

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)  # Butun veriye modeli fit ediyoruz.

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

# Hangi eyalet hangi cluster'da hala bilmiyorum.

clusters_kmeans = kmeans.labels_

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

df["cluster"] = clusters_kmeans                     # Dataframe'e yeni bir degisken ekledik. Yani tuttugumuz labellari df icerisine gonderdik.

df.head()

df["cluster"] = df["cluster"] + 1                   # clusterlardaki 0 ifadesi bizi rahatsiz etmesin diye 1 ekledik. :)
# Hangi eyaletin hangi cluster'da oldugunu artik biliyoruz.

df[df["cluster"]==5]                                # Bes numarali cluster'da hangi eyalerin olduguna baktik.

df.groupby("cluster").agg(["count","mean","median"])# Bir cluster'daki gozlem sayisi, ortalama ve medyanlarini gormek istiyorum.

df.to_csv("clusters.csv")                           # csv dosyasi olarak disari cikardik.








## Hierarchical Clustering

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

# Uzaklik temelli yontem kullandigimiz icin veri setini standartlastiriyoruz;
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# Birlestirici bir classting yontemi olan "linkage"; öklid uzakligine gore gozlem birimlerini kumelere ayiriyor.
hc_average = linkage(df, "average")

# "dendrogram" ile kumeleme yapisini gosteren semaya bakariz;
plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show(block=True)


# Grafigi toplulastirmak istersek;
plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")                      # Kac tane gozlem barindirdigi
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show(block=True)

# Hiyerarsik kumeleme yontemi: Gozlem birimlerine genelden bakma ve buna gore cesitli karar noktalarina dokunma sansi verir.



# Kume Sayisini Belirlemek;

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')      # y-eksenine gore cizgi atiyoruz.
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show(block=True)

# Iki tane aday noktaya gore cizgilerimizi cektik.
# Kume sayisina karar verdikten sonra final modeli olusturabiliriz.



# Final Modeli Oluşturmak

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)               # Bes cluster icin bilgileri elde etmis olduk.

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1  # Hiyerarsik kumeleme yonteminin onerdigi siniflari da elde etmis olduk.

df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df["kmeans_cluster_no"] = clusters_kmeans      # Iki farkli yontemden gelen cluster'lari ekledik.








## Principal Component Analysis ( PCA )

df = pd.read_csv("datasets/Hitters.csv")
df.head()

# Kategorik degiskenleri va bag.li degiskeni veri setimizden cikariyoruz;
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)                     # Eksik degerleri kaldirdik.
df.shape

# Amacim olusturdugum veri setinin daha az degiskenle ifade edilmesi;
df = StandardScaler().fit_transform(df)     # Standartlastirdik.

pca = PCA()                                 # Model nesnemizi cagirdik.
pca_fit = pca.fit_transform(df)             # fit edip nesnemizi olusturduk.

# Bilesenlerin basarisi, bilesenlerin acikladigi varyans oranlarina gore belirlenmektedir.

# Bilesenlerin acikladigi varyans (bilgi) oranlari;
pca.explained_variance_ratio_
# Kumulatif (pes pese bilesenlerin acıkladigi) varyanslari hesaplarsak;
np.cumsum(pca.explained_variance_ratio_)
# Inceledigimizde goruyoruz ki, ben 16 bilesenle degilde bir kac tane bilesenle veri setini aciklayacak bilgiye ulasiyorum.



# Optimum Bileşen Sayısı

# En kayda deger degisikligin nerede oldugunu gormek icin;

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show(block=True)

# Goz yordamiyla karar verirsek; iki ya da uc secilebilir. Uc bilesen zaten %80 uzerini aciklamis.Dolaayisiyle uc bileseni secebilirim. ( Yoruma acik. :))
# Veri gorsellestirme yapacaksak zaten iki boyuta indirgemek; iki bileseni secmek zorundayız ! :)


# *** Regresyon problemi ile ilgileniyor olsaydik, coglu dogrusal baglanti problemini gidermek icin degisken sayisi kadar bilesen olusturmayi tercih edebiliriz.
# Boylece veri seti icerisindeki bilginin tamami korunmus olunur ama degiskenler birbirinden bagimsiz olur,
# Yuksek korelasyon problemine, coklu dogrusal baglanti problemine sagip olmaz.



# Final PCA'in Oluşturulması

pca = PCA(n_components=3)                   # Bilesen sayisini girdik.
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_               # Aciklanan varyans oranlarina tekrar bakiyoruz. ( Her bilesenin kendi aciklama varyans orani)
np.cumsum(pca.explained_variance_ratio_)    # Bilesenlerin kumulatif olarak varyans oranlarina baktik.








## BONUS: Principal Component Regression

# Diyelim ki; Hitters veri seti dogrusal bir model ile modellenmek istiyor ve degiskenler arasinda coklu dogrusal baglanti problemi var.
# Degiskenler arasindaa yuksek korelasyon varsa bu cesitli sorunlara neden olur; bunu istemiyoruz.

df = pd.read_csv("datasets/Hitters.csv")
df.shape            # 322 tane gozlem birimi var.

len(pca_fit)        # 322 tane, yani gozlem birimleri yerinde.

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col] # Numerik degiskenleri sectik.
len(num_cols)                                                                           # Numerik degiskenlerin sayisina baktik.

others = [col for col in df.columns if col not in num_cols]                             # num_cols disinda kalan degiskenleri de cagirdik.


# pca_fit'deki uc bileseni okunabilirlik acisindan dataframe'e ceviriyoruz;
pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()                             # 16 sayisal degiskeni 3 tane degiskene indirgedik.

df[others].head()                                                                       # Bag.li deg. ve kategorik degiskenler.

# Sayisal degiskenleri secersem ve onlari bilesenlerce ifade edebilirsek,
# bunun uzerine bir regresyon modeli kurarim ve sonucu elde ederiz.
# Yani bilesenleri degisken olarak kullanabiliyoruz.

# Iki veri setini bir araya getirirsek
final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1)                                              # axis=1; yan yana koy, liste icersinde iki df'i ver dedik.
final_df.head()                                                                         # 16 degiskenin 3'e (aralarinda korelasyony yok) indigini gozlemledik.
# Bilesen indirgemesini tamamlamis olduk.


# Regresyon modelini kuralim;
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Kategorik degiskenlerim var (hepsinin sinif sayisi 2) label enc, one hot enc. ya da get_dummys'i kullanabiliriz.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)                                                           # NaN degerlerden kurtulduk. :)

# Bagimli ve bagimsiz degiskenleri secersek;
y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

# Bos model nesnemizi cagiririz;
lm = LinearRegression()                                                                 # Daha sonra kullanmayacagimiz icin fit etmedik. Direkt hatasina bakacagiz.
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))     # 5 katli capraz dogrulama yontemi ile  RMSE degerine baktik.
# Hata oraninin uygunlugunu kontrol etmek icin bagimli degiskenin ortalamasi ile kiyaslanir;
y.mean()                                                                                        # Hata degeri cok dusuk de degil, cok yuksek de degil.
# Degisken sayisini indirgedimiz halde (bilgi kaybini goze aldik) basarimiz cok kotu degil. :)



# Oylesine Decision Tree Regressor'a bakmak istersek;
cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))
# Hata degeri regresyon modelinden daha kotu gibi gorunuyor.

# Hiperparametre optimizasyonu yapalim;
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

# Hiperparametre optimizasyonu sonucu hata degerine bakarsak;
rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))
# Hata degeri azalmis oldu. : )








# BONUS: PCA ile Cok Boyutlu Veriyi 2 Boyutta Gorsellestirme

### Breast Cancer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/breast_cancer.csv")

# diagnosis; iyi huylu, kotu huylu olma durumunu ifade eder.

y = df["diagnosis"]                                     # Bagimli degiskenimiz
X = df.drop(["diagnosis", "id"], axis=1)                # Bagimsiz degiskenlerimiz



def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)                               # Bagimsiz degiskenler standartlastirilir,
    pca = PCA(n_components=2)                                           # PCA hesabi yapilir,
    pca_fit = pca.fit_transform(X)                                      # Degisken degerlerini donusturme yani bilesenleri cikarma,
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])         # Bilesenleri df'e cevirme,
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)             # df'i bagimli deg. ile concat
    return final_df

pca_df = create_pca_df(X, y)                            # Fonksiyonu cagirdigimizda veri setinin iki bilesene indirgendigini gorduk.

# Bu iki bileseni gorsellestirmek istersek (bag.li degiskendeki sinif sayisina gore colors degerleri yazilir normalde   !);
def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())                  # Essiz target sinifini verir.
    colors = random.sample(['r', 'b', "g", "y"], len(targets))  # Essiz sinif sayisi kadar rastgele renk sectik.

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show(block=True)

plot_pca(pca_df, "diagnosis")                       # Siniflarin dagilimlari iki eksen uzerinde gorsellestirilmis oldu.
# *** Fonksiyona gonderilecek X'in sayisal degiskenlerden olusmasi lazim! Bagimsiz degiskenlerin icerisinde kategorik degisken olmamasi lazim!





### "Iris" Veri setine uygularsak;
# Ciceklerin tac yaprak ve canak yaprak bilgilerini iceren veri seti;
# 4 tane bagimsiz degisken var ama 3 tane sinifi olan bagimli degisken var; siniflandirme problemi...

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")





### "Diabetes" veri setine uygularsak;

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")