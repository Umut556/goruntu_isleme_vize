import sys
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QVBoxLayout, QWidget, QLabel, QFileDialog, QMdiArea, QMdiSubWindow, QPushButton, QHBoxLayout, QSlider, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QTransform
from PyQt5.QtCore import Qt, QPointF


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dijital Görüntü İşleme Dersi")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.menu_olustur()

        self.bilgi()

        self.mdi_alani = QMdiArea()
        self.layout.addWidget(self.mdi_alani)
        
        self.toolbar = self.addToolBar("Toolbar")
        
        self.sigmoid = {
            "standart": self.standard_sigmoid,
            "yatay_kaydirilmis": self.shifted_sigmoid,
            "egimli": self.sloped_sigmoid,
            "ozel": self.custom_function
        }
        

    def menu_olustur(self):
        menubar = self.menuBar()

        gorev1_menu = menubar.addMenu("Ödev 1: Temel İşlevselliği Oluştur")

        gorev1_eylem = QAction("Görüntü Yükle", self)
        gorev1_eylem.triggered.connect(self.resmi_yukle)
        gorev1_menu.addAction(gorev1_eylem)

        gorev2_menu = menubar.addMenu("Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon")

        gorev2_eylem = QAction("Görüntü Yükle", self)
        gorev2_eylem.triggered.connect(self.resmi_yukle_odev2)
        gorev2_menu.addAction(gorev2_eylem)
        
        vize_odev_menu = menubar.addMenu("Vize Ödev")
        
        vize_odev = QAction("S-Curve", self)
        vize_odev.triggered.connect(self.sigmoid_vize)
        vize_odev_menu.addAction(vize_odev)
        
        vize_odev2 = QAction("Yol Tespiti", self)
        vize_odev2.triggered.connect(self.yol_tespiti_vize)
        vize_odev_menu.addAction(vize_odev2)
        
        vize_odev3 = QAction("Göz Tespiti", self)
        vize_odev3.triggered.connect(self.goz_tespiti_vize)
        vize_odev_menu.addAction(vize_odev3)
        
        vize_odev4 = QAction("Deblur", self)
        vize_odev4.triggered.connect(self.deblur_resim_vize)
        vize_odev_menu.addAction(vize_odev4)
        
        vize_odev5 = QAction("Nesne Sayma ve Özellik Çıkarma", self)
        vize_odev5.triggered.connect(self.nesne_ozellik_vize)
        vize_odev_menu.addAction(vize_odev5)


    def bilgi(self):
        baslik_etiketi = QLabel("Dijital Görüntü İşleme", self)
        baslik_etiketi.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(baslik_etiketi)

        ogrenci_bilgi_etiketi = QLabel("Adı Soyadı: Umut Duran\nÖğrenci Numarası: 211229046", self)
        ogrenci_bilgi_etiketi.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(ogrenci_bilgi_etiketi)

        self.layout.addStretch()

    def resmi_yukle(self):
        dosya_diyalogu = QFileDialog(self)
        dosya_yolu, _ = dosya_diyalogu.getOpenFileName(self, "Görüntü Yükle", "", "Resim Dosyaları (*.jpg *.png)")

        if dosya_yolu:
            alt_pencere = QMdiSubWindow()
            alt_pencere.setWindowTitle("Resim")
            resim_etiketi = QLabel()
            orijinal_pixmap = QPixmap(dosya_yolu)
            resim_etiketi.setPixmap(orijinal_pixmap.scaledToWidth(400))
            alt_pencere.setWidget(resim_etiketi)

            parlaklik_dugmesi = QPushButton("Parlaklık Ayarla", self)
            parlaklik_kaydirici = QSlider(Qt.Horizontal)
            parlaklik_kaydirici.setMinimum(-100)
            parlaklik_kaydirici.setMaximum(100)
            parlaklik_kaydirici.setValue(0)
            parlaklik_kaydirici.valueChanged.connect(lambda deger: self.parlaklik_ayarla(resim_etiketi, orijinal_pixmap.copy(), deger))

            dugme_duzeni = QHBoxLayout()
            dugme_duzeni.addWidget(parlaklik_dugmesi)
            dugme_duzeni.addWidget(parlaklik_kaydirici)

            duzen = QVBoxLayout()
            duzen.addWidget(resim_etiketi)
            duzen.addLayout(dugme_duzeni)

            widget = QWidget()
            widget.setLayout(duzen)
            alt_pencere.setWidget(widget)

            self.mdi_alani.addSubWindow(alt_pencere)
            alt_pencere.show()

    def parlaklik_ayarla(self, resim_etiketi, orijinal_pixmap, deger):
        ayarlanmis_pixmap = orijinal_pixmap.copy()
        ayarlanmis_pixmap.fill(Qt.white)
        cizer = QPainter(ayarlanmis_pixmap)
        cizer.setOpacity((deger + 100) / 200.0)
        cizer.drawPixmap(0, 0, orijinal_pixmap)
        cizer.end()
        resim_etiketi.setPixmap(ayarlanmis_pixmap.scaledToWidth(400))

    def resmi_yukle_odev2(self):
        dosya_diyalogu = QFileDialog(self)
        dosya_yolu, _ = dosya_diyalogu.getOpenFileName(self, "Görüntü Yükle", "", "Resim Dosyaları (*.jpg *.png)")
    
        if dosya_yolu:
            alt_pencere = QMdiSubWindow()
            alt_pencere.setWindowTitle("Resim")
            resim_etiketi = QLabel()
            orijinal_pixmap = QPixmap(dosya_yolu)
            resim_etiketi.setPixmap(orijinal_pixmap.scaledToWidth(400))
            alt_pencere.setWidget(resim_etiketi)
    
            buton_duzeni = QHBoxLayout()
    
            zoom_in_butonu = QPushButton("Yakınlaştır", self)
            zoom_in_butonu.clicked.connect(lambda: self.yakinlastir(resim_etiketi, 1.2))
            buton_duzeni.addWidget(zoom_in_butonu)
    
            zoom_out_butonu = QPushButton("Uzaklaştır", self)
            zoom_out_butonu.clicked.connect(lambda: self.uzaklastir(resim_etiketi, 0.8))
            buton_duzeni.addWidget(zoom_out_butonu)

            rotate_butonu = QPushButton("Resmi Döndür", self)
            rotate_butonu.clicked.connect(lambda: self.dondur(resim_etiketi))
            buton_duzeni.addWidget(rotate_butonu)
    
            buyut_butonu = QPushButton("Resmi Büyüt", self)
            buyut_butonu.clicked.connect(lambda: self.boyutlandir(resim_etiketi, "büyüt"))
            buton_duzeni.addWidget(buyut_butonu)
    
            kucult_butonu = QPushButton("Resmi Küçült", self)
            kucult_butonu.clicked.connect(lambda: self.boyutlandir(resim_etiketi, "küçült"))
            buton_duzeni.addWidget(kucult_butonu)
    
            duzen = QVBoxLayout()
            duzen.addWidget(resim_etiketi)
            duzen.addLayout(buton_duzeni)
    
            widget = QWidget()
            widget.setLayout(duzen)
            alt_pencere.setWidget(widget)

            self.mdi_alani.addSubWindow(alt_pencere)
            alt_pencere.show()
    #Average
    def uzaklastir(self, resim_etiketi, faktor):
        pixmap = resim_etiketi.pixmap()
        if pixmap.isNull():
            return
    
        orijinal_resim = pixmap.toImage()
    
        genislik = orijinal_resim.width()
        yukseklik = orijinal_resim.height()
    
        yeni_genislik = int(genislik * faktor)
        yeni_yukseklik = int(yukseklik * faktor)
    
        yeni_resim = QImage(yeni_genislik, yeni_yukseklik, QImage.Format_ARGB32)
        
        for x in range(yeni_genislik):
            for y in range(yeni_yukseklik):
                src_x = x / faktor
                src_y = y / faktor
    
                x0 = int(src_x)
                y0 = int(src_y)
                x1 = min(x0 + 1, genislik - 1)
                y1 = min(y0 + 1, yukseklik - 1)

                p00 = orijinal_resim.pixelColor(x0, y0)
                p01 = orijinal_resim.pixelColor(x0, y1)
                p10 = orijinal_resim.pixelColor(x1, y0)
                p11 = orijinal_resim.pixelColor(x1, y1)

                red = (p00.red() + p01.red() + p10.red() + p11.red()) // 4
                green = (p00.green() + p01.green() + p10.green() + p11.green()) // 4
                blue = (p00.blue() + p01.blue() + p10.blue() + p11.blue()) // 4
                alpha = (p00.alpha() + p01.alpha() + p10.alpha() + p11.alpha()) // 4
    
                yeni_resim.setPixelColor(x, y, QColor(red, green, blue, alpha))

        yeni_pixmap = QPixmap.fromImage(yeni_resim)
        resim_etiketi.setPixmap(yeni_pixmap)
    
    #Average
    def yakinlastir(self, resim_etiketi, faktor):
        pixmap = resim_etiketi.pixmap()
        if pixmap.isNull():
            return

        orijinal_resim = pixmap.toImage()

        genislik = orijinal_resim.width()
        yukseklik = orijinal_resim.height()

        yeni_genislik = int(genislik * faktor)
        yeni_yukseklik = int(yukseklik * faktor)

        yeni_resim = QImage(yeni_genislik, yeni_yukseklik, QImage.Format_ARGB32)

        for x in range(yeni_genislik):
            for y in range(yeni_yukseklik):
                src_x = x / faktor
                src_y = y / faktor

                x0 = int(src_x)
                y0 = int(src_y)
                x1 = min(x0 + 1, genislik - 1)
                y1 = min(y0 + 1, yukseklik - 1)

                p00 = orijinal_resim.pixelColor(x0, y0)
                p01 = orijinal_resim.pixelColor(x0, y1)
                p10 = orijinal_resim.pixelColor(x1, y0)
                p11 = orijinal_resim.pixelColor(x1, y1)

                red = (p00.red() + p01.red() + p10.red() + p11.red()) // 4
                green = (p00.green() + p01.green() + p10.green() + p11.green()) // 4
                blue = (p00.blue() + p01.blue() + p10.blue() + p11.blue()) // 4
                alpha = (p00.alpha() + p01.alpha() + p10.alpha() + p11.alpha()) // 4
    
                yeni_resim.setPixelColor(x, y, QColor(red, green, blue, alpha))

        yeni_pixmap = QPixmap.fromImage(yeni_resim)
        resim_etiketi.setPixmap(yeni_pixmap)
    
    #Bicubic
    def boyutlandir(self, resim_etiketi, islem):
        yuzde, ok_pressed = QInputDialog.getInt(self, "Boyutlandır", "Boyutlandırma Yüzdesi:", 0, 0, 1000)
    
        if ok_pressed:
            orijinal_pixmap = resim_etiketi.pixmap()
            if orijinal_pixmap.isNull():
                return

            orijinal_resim = orijinal_pixmap.toImage()

            genislik = orijinal_resim.width()
            yukseklik = orijinal_resim.height()

            if islem == "büyüt":
                yeni_genislik = int(genislik * (1 + yuzde / 100))
                yeni_yukseklik = int(yukseklik * (1 + yuzde / 100))
            elif islem == "küçült":
                yeni_genislik = int(genislik * (1 - yuzde / 100))
                yeni_yukseklik = int(yukseklik * (1 - yuzde / 100))

            yeni_resim = QImage(yeni_genislik, yeni_yukseklik, QImage.Format_ARGB32)

            for x in range(yeni_genislik):
                for y in range(yeni_yukseklik):
                    src_x = x / (yeni_genislik / genislik)
                    src_y = y / (yeni_yukseklik / yukseklik)

                    interpolated_color = self.bicubic_interpolation(orijinal_resim, src_x, src_y)
    
                    yeni_resim.setPixelColor(x, y, interpolated_color)

            yeni_pixmap = QPixmap.fromImage(yeni_resim)
            resim_etiketi.setPixmap(yeni_pixmap)
    
    def bicubic_interpolation(self, resim, x, y):
        def cubic_hermite(v0, v1, v2, v3, x):
            c0 = v1
            c1 = 0.5 * (v2 - v0)
            c2 = v0 - 2.5 * v1 + 2 * v2 - 0.5 * v3
            c3 = 0.5 * (v3 - v0) + 1.5 * (v1 - v2)
            return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
    
        x0 = int(x)
        y0 = int(y)
        x1 = x0 + 1
        y1 = y0 + 1

        p00 = QColor(resim.pixel(x0, y0))
        p01 = QColor(resim.pixel(x0, y1))
        p10 = QColor(resim.pixel(x1, y0))
        p11 = QColor(resim.pixel(x1, y1))

        dx = x - x0

        red = cubic_hermite(p00.redF(), p01.redF(), p10.redF(), p11.redF(), dx)
        green = cubic_hermite(p00.greenF(), p01.greenF(), p10.greenF(), p11.greenF(), dx)
        blue = cubic_hermite(p00.blueF(), p01.blueF(), p10.blueF(), p11.blueF(), dx)
        alpha = cubic_hermite(p00.alphaF(), p01.alphaF(), p10.alphaF(), p11.alphaF(), dx)

        interpolated_color = QColor()
        interpolated_color.setRgbF(red, green, blue, alpha)
    
        return interpolated_color
    
    #Bilinear
    def dondur(self, resim_etiketi):
        aci, ok_pressed = QInputDialog.getInt(self, "Resmi Döndür", "Dönüş Açısı (derece):", 0, -360, 360)
        if ok_pressed:
            pixmap = resim_etiketi.pixmap()
            if pixmap.isNull():
                return

            orijinal_resim = pixmap.toImage()

            genislik = orijinal_resim.width()
            yukseklik = orijinal_resim.height()

            merkez = QPointF(genislik / 2, yukseklik / 2)

            donusum_matrisi = QTransform().translate(merkez.x(), merkez.y()).rotate(aci).translate(-merkez.x(), -merkez.y())

            yeni_genislik = genislik
            yeni_yukseklik = yukseklik
            dondurulmus_pixmap = QPixmap(yeni_genislik, yeni_yukseklik)
            dondurulmus_pixmap.fill(Qt.transparent)
            painter = QPainter(dondurulmus_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            for x in range(yeni_genislik):
                for y in range(yeni_yukseklik):
                    dondurulmus_konum = donusum_matrisi.map(QPointF(x, y))

                    if dondurulmus_konum.x() >= 0 and dondurulmus_konum.x() < genislik and dondurulmus_konum.y() >= 0 and dondurulmus_konum.y() < yukseklik:
                        x0 = int(dondurulmus_konum.x())
                        y0 = int(dondurulmus_konum.y())
                        x1 = min(x0 + 1, genislik - 1)
                        y1 = min(y0 + 1, yukseklik - 1)

                        p00 = QColor(orijinal_resim.pixel(x0, y0))
                        p01 = QColor(orijinal_resim.pixel(x0, y1))
                        p10 = QColor(orijinal_resim.pixel(x1, y0))
                        p11 = QColor(orijinal_resim.pixel(x1, y1))

                        dx = dondurulmus_konum.x() - x0
                        dy = dondurulmus_konum.y() - y0
                        red = int((1 - dx) * (1 - dy) * p00.red() + dx * (1 - dy) * p10.red() + (1 - dx) * dy * p01.red() + dx * dy * p11.red())
                        green = int((1 - dx) * (1 - dy) * p00.green() + dx * (1 - dy) * p10.green() + (1 - dx) * dy * p01.green() + dx * dy * p11.green())
                        blue = int((1 - dx) * (1 - dy) * p00.blue() + dx * (1 - dy) * p10.blue() + (1 - dx) * dy * p01.blue() + dx * dy * p11.blue())
                        alpha = int((1 - dx) * (1 - dy) * p00.alpha() + dx * (1 - dy) * p10.alpha() + (1 - dx) * dy * p01.alpha() + dx * dy * p11.alpha())

                        painter.setPen(QColor(red, green, blue, alpha))
                        painter.drawPoint(x, y)
    
            painter.end()
            resim_etiketi.setPixmap(dondurulmus_pixmap)
            
    def sigmoid_vize(self):
        dosya_diyalogu = QFileDialog(self)
        dosya_yolu, _ = dosya_diyalogu.getOpenFileName(self, "Görüntü Yükle", "", "Resim Dosyaları (*.jpg *.png)")
    
        if dosya_yolu:
            orijinal_pixmap = QPixmap(dosya_yolu)
            orijinal_resim = orijinal_pixmap.toImage()
    
            alt_pencere = QMdiSubWindow()
            alt_pencere.setWindowTitle("Resimler")
            alt_pencere_layout = QHBoxLayout()
    
            orijinal_resim_etiketi = QLabel("Orjinal Resim")
            orijinal_resim_etiketi.setAlignment(Qt.AlignCenter)
            orijinal_resim_etiketi.setStyleSheet("font-weight: bold")
            orijinal_pixmap_scaled = orijinal_pixmap.scaled(200, 200, Qt.KeepAspectRatio)
            orijinal_resim_etiketi.setPixmap(orijinal_pixmap_scaled)
            alt_pencere_layout.addWidget(orijinal_resim_etiketi)
    
            for sigmoid_func_name, sigmoid_func in self.sigmoid.items():
                sigmoid_resim = self.apply_sigmoid(orijinal_resim, sigmoid_func)
    
                resim_etiketi = QLabel()
                resim_etiketi.setAlignment(Qt.AlignCenter)
                resim_etiketi.setPixmap(sigmoid_resim)
    
                widget = QWidget()
                layout = QVBoxLayout()
                layout.addWidget(resim_etiketi)
                widget.setLayout(layout)
    
                alt_pencere_layout.addWidget(widget)
    
            alt_pencere_widget = QWidget()
            alt_pencere_widget.setLayout(alt_pencere_layout)
            alt_pencere.setWidget(alt_pencere_widget)
    
            self.mdi_alani.addSubWindow(alt_pencere)
            alt_pencere.show()



    def standard_sigmoid(self, x):
        return 1 / (1 + np.exp(-15 * (x - 0.5)))
    
    def shifted_sigmoid(self, x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))
    
    def sloped_sigmoid(self, x):
        return 1 / (1 + np.exp(-20 * (x - 0.5)))
    
    def custom_function(self, x):
        return 1 / (1 + np.exp(-10 * (x - 0.3)))


    def apply_sigmoid(self, orijinal_resim, sigmoid_func):
        genislik = orijinal_resim.width()
        yukseklik = orijinal_resim.height()
        yeni_resim = QImage(genislik, yukseklik, QImage.Format_ARGB32)
    
        for x in range(genislik):
            for y in range(yukseklik):
                renk = QColor(orijinal_resim.pixel(x, y))
    
                r, g, b, a = renk.red(), renk.green(), renk.blue(), renk.alpha()
                r_norm = r / 255
                g_norm = g / 255
                b_norm = b / 255
    
                r_sigmoid = sigmoid_func(r_norm) * 255
                g_sigmoid = sigmoid_func(g_norm) * 255
                b_sigmoid = sigmoid_func(b_norm) * 255
    
                yeni_renk = QColor(int(r_sigmoid), int(g_sigmoid), int(b_sigmoid), a)
                yeni_resim.setPixelColor(x, y, yeni_renk)
    
        yeni_pixmap = QPixmap.fromImage(yeni_resim)
        return yeni_pixmap
    
                
    def yol_tespiti_vize(self):
        dosya_diyalogu = QFileDialog()
        dosya_yolu, _ = dosya_diyalogu.getOpenFileName(None, "Görüntü Yükle", "", "Resim Dosyaları (*.jpeg *.png *.jpg)")
        
        if dosya_yolu:
            sub_window = QMdiSubWindow()
            sub_window.setWindowTitle("Yüklenen Resim")
            sub_window.setGeometry(100, 100, 400, 400)
        
            resim = cv2.imread(dosya_yolu)
            
            yumusatilmis_resim = cv2.GaussianBlur(resim, (5, 5), 0)
            
            gri_tonlama = cv2.cvtColor(yumusatilmis_resim, cv2.COLOR_BGR2GRAY)
            
            kenarlar = cv2.Canny(gri_tonlama, 50, 150)
            
            tespit_edilen_cizgiler = cv2.HoughLinesP(kenarlar, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if tespit_edilen_cizgiler is not None:
                for cizgi in tespit_edilen_cizgiler:
                    x1, y1, x2, y2 = cizgi[0]
                    cv2.line(resim, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            height, width, channel = resim.shape
            bytesPerLine = 3 * width
            q_img = QImage(resim.data, width, height, bytesPerLine, QImage.Format_RGB888)
            label = QLabel()
            label.setPixmap(QPixmap.fromImage(q_img))
            sub_window.setWidget(label)
        
            self.mdi_alani.addSubWindow(sub_window)
            sub_window.show()

            
    def goz_tespiti_vize(self):
        dosya_diyalogu = QFileDialog()
        dosya_yolu, _ = dosya_diyalogu.getOpenFileName(self, "Görüntü Yükle", "", "Resim Dosyaları (*.jpg *.png *.jpeg *.webp)")
        
        if dosya_yolu:
            sub_window = QMdiSubWindow()
            sub_window.setWindowTitle("Yüklenen Resim")
            sub_window.setGeometry(100, 100, 400, 400)
        
            resim = cv2.imread(dosya_yolu)
            gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        
            dp = 1
            minDist = 50
            param1 = 200
            param2 = 30
            minRadius = 10
            maxRadius = 30
        
            daireler = cv2.HoughCircles(gri_resim, cv2.HOUGH_GRADIENT, dp, minDist,
                                        param1=param1, param2=param2,
                                        minRadius=minRadius, maxRadius=maxRadius)
        
            if daireler is not None:
                daireler = np.uint16(np.around(daireler))
                for daire in daireler[0, :]:
                    merkez_x, merkez_y, yaricap = daire
                    cv2.circle(resim, (merkez_x, merkez_y), yaricap, (0, 255, 0), 2)
        
            height, width, channel = resim.shape
            bytesPerLine = 3 * width
            rgb_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_resim.data, width, height, bytesPerLine, QImage.Format_RGB888)
        
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(QPixmap.fromImage(q_img))
            sub_window.setWidget(label)
        
            self.mdi_alani.addSubWindow(sub_window)
            sub_window.show()

    
    def deblur_gaussian(self, image, kernel_size=(5, 5), sigmaX=10):
        blurred = cv2.GaussianBlur(image, kernel_size, sigmaX)
        deblurred = cv2.addWeighted(image, 2, blurred, -1, 0)
        return deblurred
    
    def deblur_resim_vize(self):
        dosya_diyalogu = QFileDialog()
        dosya_yolu, _ = dosya_diyalogu.getOpenFileName(self, "Görüntü Yükle", "", "Resim Dosyaları (*.jpeg *.png)")
    
        if dosya_yolu:
            resim = cv2.imread(dosya_yolu)
            cv2.imshow("Orijinal Resim", resim)
    
            deblurred_image = self.deblur_gaussian(resim)
    
            cv2.imshow("Duzeltilmis Resim", deblurred_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    def nesne_ozellik_vize(self):
        dosya_diyalogu = QFileDialog()
        dosya_yolu, _ = dosya_diyalogu.getOpenFileName(self, "Görüntü Yükle", "", "Resim Dosyaları (*.jpg *.png)")
        
        if dosya_yolu:
            img = cv2.imread(dosya_yolu)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            lower_green = np.array([0, 100, 0], dtype="uint8")
            upper_green = np.array([50, 255, 50], dtype="uint8")
            green_mask = cv2.inRange(img_rgb, lower_green, upper_green)
            
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            nesne_listesi = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                length = max(w, h)
                width = min(w, h)
                diagonal = np.sqrt(w**2 + h**2)
                energy = np.sum(img[y:y+h, x:x+w]) / (w * h)
                entropy = -1 * np.sum((np.histogram(img[y:y+h, x:x+w], bins=256)[0] / (w * h) + np.finfo(float).eps) * np.log2(np.histogram(img[y:y+h, x:x+w], bins=256)[0] / (w * h) + np.finfo(float).eps))
                mean = np.mean(img[y:y+h, x:x+w])
                median = np.median(img[y:y+h, x:x+w])
                nesne_listesi.append([x + w/2, y + h/2, length, width, diagonal, energy, entropy, mean, median])
            
            df = pd.DataFrame(nesne_listesi, columns=["Center X", "Center Y", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"])
            
            excel_adı = dosya_yolu.split('.')[0] + "_nesne_ozellikleri.xlsx"
            df.to_excel(excel_adı, index=False)
            print("Excel dosyası oluşturuldu:", excel_adı)
            
        else:
            print("Dosya seçilmedi.")
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())