1)	PROJENIN AMACI
Amac, Scikit-learn kütüphanesinin Olivetti Yüzleri veri kümesi üzerinde yüz tanıma sınıflandırması yapmaktır. Bu sınıflandırma problemi, görüntü işleme ve makine öğrenimi alanlarında sıklıkla karşılaşılan bir uygulama örneğidir. Veri kümesindeki yüz görüntülerini kullanarak, farklı bireylere ait yüzleri tanımayı ve doğru bir şekilde sınıflandırmayı hedefliyoruz.

2)	VERI SETI

a)	Hangi özelliklerden oluşuyor?
Olivetti Yüzleri veri kümesi, 64x64 piksel boyutundaki gri tonlamalı yüz görüntülerinden oluşmaktadır. Her görüntü 4096 (64x64) piksel değerinden meydana gelmektedir.

b)	Kaç örnek bulunuyor?
Veri kümesinde toplam 400 adet yüz görüntüsü bulunmaktadır. Bu görüntüler, 40 farklı bireye ait 10'ar adet görüntüden oluşmaktadır.

c)	Sınıf bilgisi
Veri kümesindeki her bir görüntü, 0 ile 39 arasında değişen sınıf etiketleriyle ilişkilendirilmiştir. Her sınıf, veri kümesinde bulunan 40 farklı bireyi temsil etmektedir.

Özet olarak, Olivetti Yüzleri veri kümesi 400 adet gri tonlamalı yüz görüntüsünden oluşmakta ve 40 farklı bireyin sınıf etiketleriyle ilişkilendirilmiş bulunmaktadır. Bu veri kümesi, görüntü sınıflandırma ve yüz tanıma gibi uygulamalarda sıklıkla kullanılmaktadır.
3)	
Kodda üç farklı sınıflandırıcı kullanılmaktadır: Naive Bayes Sınıflandırıcısı, Çok Katmanlı Algılayıcı (MLP) Sınıflandırıcısı ve Evrişimli Sinir Ağı (CNN) Sınıflandırıcısı.
a) Veri Ön İşleme:

Veri ön işleme tekniklerinden biri olan Standardizasyon (StandardScaler) kullanılmıştır. Bu sayede, görüntü piksel değerleri 0 ortalama ve 1 birim varyansa sahip hale getirilmiştir. Bu işlem, sınıflandırıcıların performansını iyileştirmek için önemlidir.
b)Sınıflandırma Algoritmaları:
Üç farklı sınıflandırıcı (Naive Bayes, MLP ve CNN) uygulanmış ve sonuçları sunulmuştur.

c)Veri Görselleştirme:
Sınıf Dağılımı Grafiği
Sınıflandırma Doğruluğu Grafiği
Her bir pikselin değer dağılımını göstermek için histogram

d)Başarı Ölçütleri:

Naive Bayes Sınıflandırıcısı için, eğitim-test ayırımı (%66-%34) kullanılarak elde edilen sınıflandırma doğruluğu (Accuracy) ve sınıflandırma raporu (Classification Report) sunulmuştur.
MLP ve CNN Sınıflandırıcıları için, eğitim-test ayırımı (%80-%20) kullanılarak elde edilen test doğrulukları (Test Accuracy) sunulmuştur.
