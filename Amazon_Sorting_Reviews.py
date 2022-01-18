

## RATING PRODUCT & SORTING REVIEWS IN AMAZON

###############
## İŞ PROBLEMİ
###############
# Ürün ratinglerini daha doğru hesaplamaya çalışmak ve ürün yorumlarını daha doğru
# sıralamak.

#####################
## VERİ SETİ HİKAYESİ
#####################
# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı
# puanları ve yorumları vardır



import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df=pd.read_csv("datasets/amazon_review.csv")
df.head()

# GÖREV-1: Average Rating’i güncel yorumlara göre
# hesaplayınız ve var olan average rating ile kıyaslayınız.



def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

df["overall"].mean()

# Average Rating = 4.69
# Var olan Average Rating = 4.58
# Güncel olana göre hesapladığımızda ürünlerin rating ortalamaları daha yüksek çıkıyor.


# GÖREV-2 : Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.

df["total_vote"].sum()

df["helpful_yes"].sum()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df["helpful_no"].sum()

# UP-DOWN Diff Score

def score_up_down_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

score_up_down_diff(6444,1034)

def score_average_rating(helpful_yes, helpful_no):
    return helpful_yes / (helpful_yes+helpful_no)

score_average_rating(6444,1034)

# Hacmi de göz önünde bulundurmak için Wilson Lower Bound Score kullanacağız.

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(6444,1034)


df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)