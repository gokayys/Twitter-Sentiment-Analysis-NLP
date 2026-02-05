# -*- coding: utf-8 -*-

"""
Created on Sun Nov 30 15:17:59 2025  
@author: gokay  
"""  
# Dosyanın oluşturulma tarihi ve yazarı bilgisi

import pandas as pd  
# CSV dosyasını okumak ve tablo (DataFrame) işlemleri yapmak için pandas kütüphanesi

import re  
# Metinler üzerinde düzenleme ve temizleme işlemleri yapmak için regex kütüphanesi

import matplotlib.pyplot as plt  
# Grafik çizimleri yapmak için matplotlib

import seaborn as sns  
# Matplotlib üzerine kurulu, daha anlaşılır ve görsel grafikler için seaborn

from sklearn.model_selection import train_test_split  
# Veriyi eğitim ve test olarak ayırmak için kullanılan fonksiyon

from sklearn.feature_extraction.text import TfidfVectorizer  
# Metin verilerini sayısal vektörlere dönüştürmek için TF-IDF yöntemi

from sklearn.metrics import accuracy_score  
# Modelin doğruluk oranını hesaplamak için

from sklearn.metrics import confusion_matrix  
# Gerçek ve tahmin edilen sınıfları karşılaştırmak için hata matrisi

from sklearn.metrics import classification_report  
# Precision, recall ve f1-score değerlerini görmek için

from sklearn.linear_model import LogisticRegression  
# Lojistik Regresyon sınıflandırma modeli

from sklearn.naive_bayes import MultinomialNB  
# Naive Bayes sınıflandırma modeli

from sklearn.svm import LinearSVC  
# Support Vector Machine (SVM) sınıflandırma modeli

from sklearn.tree import DecisionTreeClassifier  
# Karar Ağacı sınıflandırma modeli

from sklearn.ensemble import RandomForestClassifier  
# Rastgele Orman sınıflandırma modeli


print("Dosya yükleniyor...")  
# Program başlıyor

try:  
    # CSV dosyasını okumayı deniyoruz
    df = pd.read_csv(r"veri.csv")  
    # Belirtilen dosya yolundaki CSV dosyasını DataFrame olarak okumasını sağlıyoruz
except FileNotFoundError:  
    # Dosya bulunamazsa bu blok çalışır
    print("DOSYA BULUNAMADI! Lütfen dosya yolunu kontrol et.")  
    # Kullanıcıya hata mesajı verir
    df = pd.DataFrame()  
    # Program hata vermesin diye boş tablo oluşturuyoruz


if not df.empty:  
    # DataFrame boş değilse devam eder
    
    
    def clean_text(text):  
        # Tweet metinlerini temizlemek için fonksiyon tanımlıyoruz
        
        text = str(text).lower()  
        # Metni stringe çevirir ve tüm harfleri küçük yapıyoruz
        
        text = re.sub(
            r"http\S+|www\S+|@\w+|[^\w\s]|\d+",
            '',
            text
        )  
        # Linkleri, kullanıcı etiketlerini, noktalama işaretlerini ve sayıları siliyoruz
        
        return text  
        # Temizlenmiş metni fonksiyonun çıktısı olarak döndürür

    df['clean_text'] = df['tweet'].apply(clean_text)  
    # clean_text fonksiyonunu her tweet’e uygular ve yeni bir sütun oluşturur


    print("\n" + "-"*30)  
    # Konsolda alt satıra geçip 30 tane "-" karakteri basar

    print("KANIT 1: TEMİZLİK KONTROLÜ")  
    # Temizlik işleminin kontrol edildiğini belirtir

    print("-"*30)  
    # Başlık altına ayırıcı çizgi çizer

    print(df[['tweet', 'clean_text']].head())  
    # Orijinal ve temizlenmiş tweetleri ilk 5 satırda gösterir


    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],  
        # Girdi verisi olarak temizlenmiş tweetler
        
        df['class'],  
        # Çıkış verisi olarak sınıf etiketleri
        
        test_size=0.2,  
        # Verinin %20’si test için ayrılır
        
        random_state=123  
        # Her çalıştırmada aynı veri bölünmesi için sabit değer
    )

    tfidf = TfidfVectorizer(
        max_features=5000,  
        # En sık geçen 5000 kelimeyi kullanır
        
        stop_words='english'  
        # Anlam taşımayan İngilizce kelimeleri çıkarır
    )

    X_train_vec = tfidf.fit_transform(X_train)  
    # Eğitim verisini kullanarak TF-IDF vektörlerini oluşturur

    X_test_vec = tfidf.transform(X_test)  
    # Test verisini, eğitimde öğrenilen kelime yapısına göre dönüştürür


    print("\n" + "-"*30)  
    # TF-IDF çıktısı öncesi ayırıcı çizgi

    print("KANIT 2: TF-IDF BOYUTU")  
    # TF-IDF matris boyutunun kontrol edildiğini belirtir

    print("-"*30)

    print(f"Kelime Havuzu (Satır, Kelime): {X_train_vec.shape}")  
    # Oluşan TF-IDF matrisinin satır ve sütun sayısını gösterir


    modeller = {  
        # Denenecek tüm modelleri bir sözlükte toplar
        
        "Lojistik Regresyon": LogisticRegression(max_iter=1000),  
        # Lojistik Regresyon modeli
        
        "Naive Bayes": MultinomialNB(),  
        # Naive Bayes modeli
        
        "Destek Vektör (SVM)": LinearSVC(
            dual=False,  
            # Veri sayısı özellik sayısından fazla olduğu için dual kapatılır
            
            max_iter=1000  
            # Modelin maksimum iterasyon sayısı
        ),
        
        "Karar Ağacı": DecisionTreeClassifier(),  
        # Karar Ağacı modeli
        
        "Rastgele Orman": RandomForestClassifier(n_estimators=50)  
        # 50 ağaçtan oluşan Rastgele Orman modeli
    }

    print("\n" + "="*40)  
    # Modellerin test edileceğini belirten ayırıcı çizgi

    print("MODELLER TEST EDİLİYOR... (Bekleyin)")  
    # Kullanıcıyı işlem süresi konusunda bilgilendirir

    print("="*40)

    en_yuksek_puan = 0  
    # En yüksek doğruluk değerini tutmak için

    kazanan_model_ismi = ""  
    # En iyi modelin adını tutmak için

    kazanan_tahminler = []  
    # En iyi modelin tahminlerini saklamak için


    for isim, model in modeller.items():  
        # Sözlükteki tüm modeller tek tek denenir
        
        model.fit(X_train_vec, y_train)  
        # Model eğitim verisiyle eğitilir
        
        tahmin = model.predict(X_test_vec)  
        # Test verisi ile tahmin yapılır
        
        puan = accuracy_score(y_test, tahmin)  
        # Gerçek etiketlerle karşılaştırılarak doğruluk hesaplanır
        
        print(f"{isim} Başarı Oranı: %{puan*100:.2f}")  
        # Modelin doğruluk oranı ekrana yazdırılır
        
        if puan > en_yuksek_puan:  
            # Eğer bu model daha başarılıysa
            
            en_yuksek_puan = puan  
            # En yüksek skor güncellenir
            
            kazanan_model_ismi = isim  
            # Kazanan modelin adı kaydedilir
            
            kazanan_tahminler = tahmin  
            # Kazanan modelin tahminleri saklanır


    print("\n" + "="*40)  
    # Sonuç bölümüne geçildiğini belirtir

    print("::: SONUÇ TABLOSU :::")  
    # Sonuç başlığı

    print("="*40)

    print(f"KAZANAN MODEL: {kazanan_model_ismi}")  
    # En başarılı modelin adı

    print(f"DOĞRULUK SKORU (Accuracy): %{en_yuksek_puan*100:.2f}")  
    # En yüksek doğruluk oranı


    print("\n--- DETAYLI SINIFLANDIRMA ANALİZİ ---")  
    # Detaylı performans ölçümlerinin başlığı

    print(classification_report(
        y_test,  
        # Gerçek etiketler
        
        kazanan_tahminler,  
        # Model tahminleri
        
        target_names=['Nefret', 'Saldırgan', 'Nötr']  
        # Sınıf isimleri
    ))


    print("\n--- KANIT 3: KAZANAN MODELİN MATRİSİ ---")  
    # Confusion matrix çıktısının başlığı

    print(confusion_matrix(y_test, kazanan_tahminler))  
    # Gerçek ve tahmin edilen sınıfların karşılaştırması


    plt.figure(figsize=(8, 6))  
    # Grafik boyutunu belirler

    cm = confusion_matrix(y_test, kazanan_tahminler)  
    # Confusion matrix tekrar hesaplanır

    sns.heatmap(
        cm,  
        # Çizilecek veri
        
        annot=True,  
        # Hücrelerin içine sayısal değerleri yazar
        
        fmt='d',  
        # Değerleri tam sayı formatında gösterir
        
        cmap='Blues',  
        # Mavi renk skalası kullanır
        
        xticklabels=['Nefret', 'Saldırgan', 'Nötr'],  
        # X ekseni: tahmin edilen sınıflar
        
        yticklabels=['Nefret', 'Saldırgan', 'Nötr']  
        # Y ekseni: gerçek sınıflar
    )

    plt.title(f"Kazanan Model ({kazanan_model_ismi}) Hata Analizi")  
    # Grafiğin başlığını belirler

    plt.ylabel("Gerçek")  
    # Y ekseninin neyi temsil ettiğini belirtir

    plt.xlabel("Tahmin")  
    # X ekseninin neyi temsil ettiğini belirtir

    plt.show()  
    # Grafiği ekranda gösterir

    print("İşlem Tamamlandı.")  
    # Programın sorunsuz bittiğini belirtir

else:
    print("HATA: Dosya okunamadığı için işlem yapılamadı.")  
    # Dosya okunamazsa program bu mesajı verir
