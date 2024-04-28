""" Makine Öğrenmesi ile Maaş Tahmini """

## İş Problemi
##### Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan
# beyzbol oyuncularının maaş tahminleri için bir makine öğrenmesi modeli geliştiriniz.

############
##### Veri Seti Hikayesi
""" Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan
StatLib kütüphanesinden alınmıştır. Veri seti 1988 ASA Grafik Bölümü
Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri
orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve
kariyer istatistikleri, Collier Books, Macmillan Publishing Company,
New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi
Güncellemesinden elde edilmiştir. """


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı (Batter)
# Hits: 1986-1987 sezonundaki isabet sayısı (Pitcher)
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı (Takıma sağlayacağı en büyük yarar sayı turudur.)
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı (Takıma sağlayacağı en büyük yarar sayı turudur.)
# Walks: Karşı oyuncuya yaptırılan hata sayısı (Atıcı pozisyonunda)
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı (Pitcher)
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı (Takıma sağlayacağı en büyük yarar sayı turudur.)
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı (Sayı turunda ne kadar koşucu varsa o kadar sayı kazandırır)
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı ((Pitcher)--Atıcı pozisyonunda)
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

"""Beyzbol takımları sahada 9 kişi ile yer alırlar. Oyunda savunma takımı atıcısı ile topu fırlatırken 
hücum takımında da elinde sopa ile dikilen vurucu, yani batter vardır. 
Takım oyuncularının hücumda hangi sırada topa vurmak üzere gelecekleri maçtan önce belirlenir. 
 "batting order" denir. Bu batting order çok önemlidir; hem taktiksel olarak hem de kaprissel olarak."""

"""Eğer top saha dışına çıkarsa buna "home run" denir. Home run'da diğer runner'lar da otomatikman run yapacağından
 tek seferde takım 4 run kazanabilir. Bu olağanüstü şeye "grand slam" denir. 
 In addition, as an extra info, The Grand Slam in tennis is the achievement of winning all four major championships in one discipline in a calendar year."""

############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import numpy as np
import warnings
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)

df = pd.read_csv("hitters.csv")

#############################################
# 1. Genel Resim

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

def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
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

#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################
#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################

""" cat_summary'yi target_summary_with_cat'e ekleyerek tek fonk. haline getirdim. """
def target_summary_with_cat(dataframe, target, categorical_col, plot=False):

    print(pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe),
                        f"{target}_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
    print("##########################################")

    if plot:
        sbn.countplot(x=dataframe[categorical_col], data=dataframe)
        plt.show()

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


df.loc[df["League"]==df["NewLeague"],["League","NewLeague"]].value_counts() #########1 Önemli 1##########
df["League"].value_counts() ## Sezon boyunca oynadığı lig
df["NewLeague"].value_counts() # Sezon başında bulunduğu lig
df.loc[~(df["League"]==df["NewLeague"]),["League","NewLeague"]] #########2 Önemli 2##########
"""
df.loc[~(df["League"] == df["NewLeague"]),["League", "NewLeague"]].index ###3### sezon başı takımı ile sezona devam takımı farklı olanlar
df[df["League"] != df["NewLeague"]].index  #### üsttekinin daha basite indirgenmişi
"""
type(df.loc[~(df["League"]==df["NewLeague"]),["League","NewLeague"]]) # pandas.core.frame.DataFrame
df.loc[~(df["League"]==df["NewLeague"]),["League","NewLeague","Salary"]]["Salary"].mean()
df.loc[~(df["League"] == df["NewLeague"])]["NewLeague"] ##4 series hali 4##


target_summary_with_cat(df, "Salary", "NewLeague")
df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="N"), ["Salary"]].mean() ## Sezona N de başlayıp A da devam edenler
len(df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="N"), ["Salary"]]) # 9
df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]].mean() ## Sezona A da başlayıp N de devam edenler
len(df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]]) # 10
target_summary_with_cat(df, "Salary", "League")
df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="N"), ["Salary"]].mean() ## Sezona N de başlayıp N de devam edenler
len(df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="N"), ["Salary"]]) # 137
df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]].mean() ## Sezona A da başlayıp A da devam edenler
len(df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]]) # 166


"""" 1. Sınıfımız ********** N ----> A **************** Tek bir gözlem için neler dağları delmediğimiz kaldı! 💖 ************************************** """
""" #### 9 taneden sadece birinin Salary'si Nan, bu yüzden bunu modelle doldurup devam edelim. """
df_N_to_A = df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="N"), :] # N ----> A
df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="N"), ["Salary"]].mean() ### 558.125
##### catcols ları label encoder dan geçirelim:
for col in cat_cols:
    lab_enc = LabelEncoder()
    df_N_to_A[col] = lab_enc.fit_transform(df_N_to_A[col])

#### Robust Scaler uygulayalım
rob_scaler = RobustScaler()
num_cols = [col for col in num_cols if col not in "Salary"]
df_N_to_A[num_cols] = rob_scaler.fit_transform(df_N_to_A[num_cols])

df_N_to_A.shape
index_138_df = df_N_to_A.iloc[df_N_to_A.index==138, :] #########3 Önemli 3######### Maaşı bilinmeyen Beyzbolcunun özellikleri
index_138_df.drop("Salary", axis=1, inplace=True) ## Salary'si tahmin edileceği için Salary değişkeni çıkarılır
df_N_to_A.drop(138, axis=0, inplace=True) ## trainde tahmin edeceğimiz gözlem olmamalı
X_N_to_A = df_N_to_A.drop("Salary", axis=1) ## bağımsız değişkenler
df_N_to_A.shape # (8, 20)
X_N_to_A.isnull().sum()
y_N_to_A = df_N_to_A["Salary"] ## bağımlı değişken
### modelimiz ile train edelim:
model_N_to_A = KNeighborsRegressor().fit(X_N_to_A, y_N_to_A)
""" ### Gözlemimizin Salary tahmini:"""
model_N_to_A.predict(index_138_df)[0] ## 388
""" ### Değeri df teki yerine atamamız lazım! """

""" KNN vs LGBM ---> Üstte görüldüğü üzere KNN'i seçtik.
rmse = np.mean(np.sqrt(-cross_val_score(mod, X_N_to_A, y_N_to_A, cv=3, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)} ('KNN') ") ##3 RMSE: 385.7402 ('LGBM')  --2 RMSE: 596.6178 ('LGBM') --4 RMSE: 437.4784 ('LGBM') 
###4 RMSE: 447.244 ('KNN') --2 NAN ---  3 RMSE: 409.1158 ('KNN') """

# Find the index of the column
column_index = df.columns.get_loc("Salary")
df.iloc[df.index==138, column_index] = model_N_to_A.predict(index_138_df)[0]
"""" ********************************** Çok Sevdim 💖 ********************************************* """

"""" 2. Sınıfımız ********** A ----> N **************** Tek bir gözlem için neler dağları delmediğimiz kaldı! 💖 ************************************** """
df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]].mean() ## A ---> N
len(df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]]) # 10
df_A_to_N = df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="N"), :]
df_A_to_N.isnull().sum() # 0
"""" ********************************** ********************************************* """

"""" 3. Sınıfımız ********** N ----> N **************** Tek bir gözlem için neler dağları delmediğimiz kaldı! 💖 ************************************** """
df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="N"), ["Salary"]].mean() ##  N --- > N
len(df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="N"), ["Salary"]]) # 137
df_N_to_N = df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="N"), :]

"""" 4. Sınıfımız ********** A ----> A **************** Tek bir gözlem için neler dağları delmediğimiz kaldı! 💖 ************************************** """
df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]].mean() ## A --- > A
len(df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="A"), ["Salary"]]) # 166
df_A_to_A = df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="A"), :]

"""" ********************************** ********************************************* """

""" Artık değişkenimizi oluşturabiliriz. """
#### Yeni değişken oluşturalım:
df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="N"), "NEW_CLASSIFY_PLAYER"] = 3 # N --> A
df.loc[(df["League"] != df["NewLeague"]) & (df["NewLeague"]=="A"), "NEW_CLASSIFY_PLAYER"] = 2 # A --> N
df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="N"), "NEW_CLASSIFY_PLAYER"] = 1 # N --> N
df.loc[(df["League"] == df["NewLeague"]) & (df["NewLeague"]=="A"), "NEW_CLASSIFY_PLAYER"] = 4 # A --> A
df["NEW_CLASSIFY_PLAYER"].value_counts() # object
df["NEW_CLASSIFY_PLAYER"].info() ### float

df.loc[:,["NEW_CLASSIFY_PLAYER","Salary"]].head(10)
df["NEW_CLASSIFY_PLAYER"] * df["Salary"] ## type'ı float fakat cat_cols larda olması gerek ve de öyle zaten.
###### float old için dört işlem ile üzerinden değişken üretilebilir.

####### Değişken oluşturmanın farklı bir yöntemi
""" df["NEW_changing_league"] = np.where((df["League"] != df["NewLeague"]) & (df["NewLeague"]=="N"), "BETTER", "WORSE")
 *** "&" yerine "and" kullanmam sonucu hata almıştım. Chatten gelen cevap:
*** It looks like you're trying to use the and logical operator within the np.where function. 
However, when working with pandas Series, you should use the bitwise "&" operator for element-wise logical AND."""

#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs() ### üzerimizdeki negatifleri positive çevirelim. Subliminallik içermez!:))
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    ### corr_matrix'in shape'ine göre "ones" ile değerleri 1 olan matrix, devamında altkısmını nan yapan "triu" 'yu uygula.
    ### Son olarakta sadece 1 olan yerleri cor_matrix değerlerimizle değiştir.
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    drop_index = [upper_triangle_matrix.columns.get_loc(col) for col in drop_list]
    if plot:
        sbn.set(rc={'figure.figsize': (15, 15)})
        sbn.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list, dataframe.iloc[:, drop_index]

high_correlated_cols(df, plot=True)


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)


#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

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

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols: ### aykırı değer yok zaten
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


#############################################
# 2. Missing Values (Eksik Değerler)
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df) ## sadece Salaryde; olması gerek!

"""df.dropna(inplace=True)"""

corr1 = df[num_cols].corr()
# Korelasyonların gösterilmesi
sbn.set()
sbn.heatmap(corr1, cmap="RdBu")
plt.show()
""" * ** *** **** ***** ****** *******"""
# Calculate correlation with other variables
correlations = df.corrwith(df["Salary"])
# Create a DataFrame from correlations
correlation_df = pd.DataFrame(correlations, columns=['Correlation'])
# Create a heatmap using seaborn
plt.figure(figsize=(16, 12))
sbn.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title(f'Correlation Heatmap for {"Salary"}')
plt.show()
""" * ** *** **** Süper! Sonuçlar yatay bir şekilde değerleriyle birlikte geldi. ***** ****** *******"""
neg_corr_cols = [ind for ind in list(correlation_df.index) if correlation_df.loc[ind, "Correlation"]<0]
pos_corr_cols = [ind for ind in list(correlation_df.index) if correlation_df.loc[ind, "Correlation"]>0]
type(neg_corr_cols)

correlation_df.loc[neg_corr_cols, "Correlation"].sort_values(ascending=False)
correlation_df.loc[pos_corr_cols, "Correlation"].sort_values(ascending=False)
""" * ** *** **** ***** ****** ******* ******** ********* ********** """
df.head()
df.describe().T


#############################################
# 3. Feature Extraction (Özellik Çıkarımı)
#############################################
df.columns = [col.upper() for col in df.columns]

### Daha önce oluşturduğumuz değişken üzerinden yürümeye çalışalım:
### 'CATBAT', 'CHITS', 'CHMRUN', 'CRUNS', 'CRBI' corr with Salery > 0.5
df["NEW_CATBAT_CLASSIFY_PLAYER"] = df["CATBAT"] + df["NEW_CLASSIFY_PLAYER"]
df["NEW_CHITS_CLASSIFY_PLAYER"] = df["CHITS"] - df["NEW_CLASSIFY_PLAYER"]
df["NEW_CHMRUN_CLASSIFY_PLAYER"] = df["NEW_CLASSIFY_PLAYER"] / df["CHMRUN"]
df["NEW_CRUNS_CLASSIFY_PLAYER"] = df["CRUNS"] + df["NEW_CLASSIFY_PLAYER"]
df["NEW_CRBI_CLASSIFY_PLAYER"] = df["CRBI"] + df["NEW_CLASSIFY_PLAYER"]

df[['CATBAT', 'CHITS', 'CHMRUN', 'CRUNS', 'CRBI']].corrwith(df["SALARY"])
NEW_cols = [col for col in df.columns if "NEW_" in col]
df[NEW_cols].corrwith(df["SALARY"])

df["NEW_COMPLEX"] = df['CATBAT'] + df['CHITS'] + df['CHMRUN'] + df['CRUNS'] + df['CRBI']
#### Yıllık ortalamaya göre bu yılki istatistiklerin önemi
df["NEW_ATBAT_IMP"] = df["ATBAT"]  / (df["CATBAT"]  / df["YEARS"])
df["NEW_HITS_IMP"] = df["HITS"] / (df["CHITS"]   / df["YEARS"]) # 86-87 vuruş sayısı / yıllık ortalama

df["NEW_HMRUN_IMP"] = df["HMRUN"] / (df["CHMRUN"]  / df["YEARS"])
df["NEW_RUNS_IMP"] = df["RUNS"] / (df["CRUNS"]  / df["YEARS"])
df["NEW_RBI_IMP"] = df["RBI"] / (df["CRBI"]   / df["YEARS"])
df["NEW_WALKS_IMP"] = df["WALKS"] / (df["CWALKS"] / df["YEARS"])
df[["NEW_ATBAT_IMP", "NEW_HITS_IMP", "NEW_HMRUN_IMP", "NEW_RUNS_IMP","NEW_RBI_IMP","NEW_WALKS_IMP"]].corrwith(df["SALARY"])

df["NEW_HITS_O_ATBAT"] = df["HITS"] / df["ATBAT"]
df["NEW_HITS_O_IMP"] = df["NEW_HITS_O_ATBAT"] * df['HMRUN']
df["NEW_HMRUN_O_ATBAT"] = df["HMRUN"] /df["ATBAT"]
df["NEW_TOT_EFF"] = df["RUNS"] + df["ASSISTS"] + df["WALKS"] - df['ERRORS']
df["NEW_RBI_WALKS"]  = df["RBI"] + df["WALKS"]
df["NEW_WALKS_E_ERRORS"] = df["WALKS"] - df['ERRORS']
df["NEW_CHITS_O_CATBAT"] = df["CHITS"] / df['CATBAT']
df["NEW_CHITS_O_IMP"] = df['CHMRUN'] * df["NEW_CHITS_O_CATBAT"]
df["NEW_CHITS_O_CRUNS"] = df["CHITS"] / df["CRUNS"]
df["NEW_CRBI_T_CWALKS"] = df["CRBI"] + df["CWALKS"]
df["NEW_CRBI_T_CWALKS"] = df["CRBI"] + df["CWALKS"]
df["NEW_ASSISTS_E_ERRORS"] = df["ASSISTS"] - df['ERRORS']
df["NEW_CWALKS_O_IMP"] = (df["CWALKS"] - df['ERRORS']) / df["YEARS"]

cat_cols, num_cols, cat_but_car = grab_col_names(df)
pd.set_option("display.max_rows", None)
df[num_cols].corrwith(df["SALARY"])

####### Üsttekilerin oranlarının etkileşimleri
df["NEW_HITS_O_ATBAT_RATE"] = df["NEW_HITS_IMP"] / df["NEW_ATBAT_IMP"]
df["NEW_HITS_O_IMP_RATE"]  = df["NEW_HITS_O_ATBAT_RATE"] * df["NEW_HMRUN_IMP"]
df["NEW_HMRUN_O_ATBAT_IMP"] = df["NEW_HMRUN_IMP"] / df["NEW_ATBAT_IMP"]
df["NEW_RBI_WALKS_SUM"] = df["NEW_RBI_IMP"] + df["NEW_WALKS_IMP"]

df.loc[:,["CHMRUN","HMRUN"]].head(15)

df["NEW_CHITS_O_CATBAT"] = df["CHITS"] / df['CATBAT'] ## BAŞARILI VURUŞ ORANI
df["NEW_CHITS_O_IMP"] = df['CHMRUN'] * df["NEW_CHITS_O_CATBAT"] ## BAŞARILI VURUŞ ORANI * DEĞERLİ VURUŞ SAYISI
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı Hits / AtBat
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı HmRun /AtBat
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı  Runs + Assits + Walks - Errors
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı  RBI + Walks
# Walks: Karşı oyuncuya yaptırılan hata sayısı  Walks - Errors
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı CHits / CAtBat
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı CHmRun * CHits/ CAtBat  ->  CHmRun/4 = en değerli vuruş sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı CHits / CRuns
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı CRBI + CWalks
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı HmRun /AtBat

# 1- (RBI -4*HmRun) / (Hits - HmRun) ### DEĞERLİ VURUŞ SAYI ORANI
df["NEW_WITHOUT_HMRUN"] = df["RBI"] - (4*df["HMRUN"])
df["NEW_WITHOUT_CHMRUN"] = df["CRBI"] - (4*df["CHMRUN"])
df["NEW_IMP_HMRUNRBI_RATE"] = df["NEW_WITHOUT_HMRUN"] / (df["HITS"] - df["HMRUN"])
df["NEW_IMP_HMRUNCRBI_RATE"] = df["NEW_WITHOUT_CHMRUN"] / (df["CHITS"] - df["CHMRUN"])
df["NEW_HMRUN_RBI_RATE"] = 4 * df["HMRUN"] / df["RBI"] ## en değerli vuruşla koşturduğu toplam oyuncu oranı
df["NEW_CHMRUN_CRBI_RATE"] = 4 * df["CHMRUN"] / df["CRBI"] ## en değerli vuruşla koşturduğu toplam oyuncu oranı

df["NEW_CATBAT_CHITS_PLAYER"] = df["NEW_CATBAT_CLASSIFY_PLAYER"] + df["NEW_CHITS_CLASSIFY_PLAYER"]

df["NEW_CHITS_O_HITS_IMP"] = df["NEW_CHITS_O_IMP"] / df["NEW_HITS_IMP"]
df["NEW_HITS_O_ATBAT_IMP"]  = df["NEW_HITS_IMP"] / df["NEW_ATBAT_IMP"]
df["NEW_HMRUN_O_ATBAT_IMP"]  = df["NEW_HMRUN_IMP"] / df["NEW_ATBAT_IMP"]

df["NEW_RBI_WALKS_O_RBI_IMP"]  = df["NEW_RBI_WALKS"] / df["NEW_RBI_IMP"]
df["NEW_CRBI_T_CWALKS_O_RBI_IMP"]  = df["NEW_CRBI_T_CWALKS"] / df["NEW_RBI_IMP"]
df["NEW_CHITS_O_CWALKS_O_IMP"] = df["NEW_CHITS_O_IMP"] + df["NEW_CWALKS_O_IMP"]

cat_cols, num_cols, cat_but_car = grab_col_names(df)
pd.set_option("display.max_rows", None)
df[num_cols].corrwith(df["SALARY"])

correlations = df.corrwith(df["SALARY"])
# Create a DataFrame from correlations
correlation_df = pd.DataFrame(correlations, columns=['Correlation'])
# Create a heatmap using seaborn
plt.figure(figsize=(16, 12))
sbn.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title(f'Correlation Heatmap for {"SALARY"}')
plt.show()


############################################
############ One Hot Encoding
############################################
def one_hot_encoder(dataf, categ_col, dropfirst=False):
    dataf = pd.get_dummies(dataf,categ_col, drop_first=dropfirst)
    return dataf

df.info()
df["NEW_CLASSIFY_PLAYER"].value_counts()
rem_cat_cols = [col for col in cat_cols if col not in "NEW_CLASSIFY_PLAYER"]
df = one_hot_encoder(df, rem_cat_cols, dropfirst=True)

###########################################
############### # 5. Feature Scaling (Özellik Ölçeklendirme)
###########################################

#### we have to fix inf and nan values to be able to apply scaling
df.isnull().sum()
rem_num_cols = [col for col in num_cols if col not in "SALARY"]
df[rem_num_cols].isnull().sum()
for col in rem_num_cols:
    df[col].fillna(0, inplace=True)
    df[col].replace([np.inf, -np.inf], 0, inplace=True) ###### inf değerlerden kurtulmak için

rob_scler = RobustScaler()
df[rem_num_cols] = rob_scler.fit_transform(df[rem_num_cols])
df.head()


##########################################
################ Modeling
##########################################

unknown_Salary_observers = df.loc[(df["SALARY"].isnull()), :].index
len(unknown_Salary_observers)
unknown_Salary_df = df.loc[(df["SALARY"].isnull()), :]
unknown_Salary_df.shape
df = df.drop(unknown_Salary_observers, axis=0)
df.shape
X = df.drop(["SALARY"], axis=1)
y = df["SALARY"]


models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")


################################################
# Random Forests
################################################

rf_model = RandomForestRegressor(random_state=17)

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

################################################
# GBM Model
################################################

gbm_model = GradientBoostingRegressor(random_state=17)

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

################################################
# LightGBM
################################################

lgbm_model = LGBMRegressor(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [300, 500],
                "colsample_bytree": [0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

################################################
# CatBoost
################################################

catboost_model = CatBoostRegressor(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse


######################################################
#  Automated Hyperparameter Optimization
######################################################

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}


lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


regressors = [("RF", RandomForestRegressor(), rf_params),
              ('GBM', GradientBoostingRegressor(), gbm_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
              ("CatBoost", CatBoostRegressor(), catboost_params)]


best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sbn.set(font_scale=1)
    sbn.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


################################################
# Analyzing Model Complexity with Learning Curves
# ################################################

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
    plt.show()


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestRegressor(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1],scoring="neg_mean_absolute_error")

rf_val_params[0][1]



