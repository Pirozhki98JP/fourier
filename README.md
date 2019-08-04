# ディジタル信号処理と画像処理　レポート課題


- OpenCV等で画像を取得し，フーリエ変換・逆変換をリアルタイムに行う．

以下のソースコードのように実装した．

``` Python

import numpy as np
import cv2
import scipy.misc
from scipy.fftpack import dct, idct
import sys
%matplotlib inline


class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType    
        self.mouseEvent["flags"] = flags    

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]                

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]                

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]  

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]  

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])

    if __name__ == "__main__":
        #入力画像
        fuji = cv2.imread('Images/fuji.jpg', 0)
        fuji.astype(float)
        fuji_F = dct(dct(fuji, axis=0), axis=1) ## 2D DCT of fuji

        H,W = fuji.shape
        plt.imshow(fuji)
        plt.title("fuji_origin")
        plt.show()

        canvas = np.zeros((H,W))
        visited = np.zeros((H,W))
        #表示するWindow名
        window_name = "dct"

        #画像の表示
        cv2.imshow(window_name, fuji_F)

        #コールバックの設定
        mouseData = mouseParam(window_name)

        while 1:
            cv2.waitKey(20)
            #左クリック
            if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
                #print(mouseData.getPos())
                x, y = mouseData.getPos()

            if x < H and y < W and visited[x, y] == 0:
                print(x,y)
                a = np.zeros((H,W))
                a[x,y] = 1
                base = idct(idct(a, axis=0), axis=1) ## create dct bases
                canvas += fuji_F[x,y] * base ## accumulate
                visited[x, y] = 1

            #右クリックがあったら終了
            elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
                break;

        cv2.destroyAllWindows()            
        print("Finished")

        plt.imshow(canvas)
        plt.title("fuji_rebuilt")
        plt.show()

```
- コードの説明

    - 前半部分は，opencvのウィンドウ上の座標をマウスから取得するために必要なクラスの定義である．参考文献1のサイトのソースコードを参考にした．
        - 今回は，ウィンドウ上の座標の取得だけなので，x座標とy座標を返す関数`getPos()`のみ使用した．

    - 今回使用した画像は，fuji.jpgである．それを以下に示す．
    ![Mt.fuji](fuji.jpg)

    - 今回，関数は，DFT（離散フーリエ変換）ではなく，DCT（離散コサイン変換）を使用した．
        - DCTは，実数列に対して実数列を返す関数であるのに対して，DFTは，実数列に対して複素数列を返すため，扱いづらく，また，今回はDFTでの実装が困難を極めたため，代替案としてDCTでの実装を試みた．

    - まず，DCTにより変換した画像`fuji_F`を生成し，openCVのウィンドウに表示する．

    - 次に，マウスから得られた座標（x,y）を元に，fujiと同じサイズで，(x,y)が１，その他が０となるようなarray型変数`a[]`を作成し，IDCTによる逆変換をかける．

    - さらに，aとfuji_Fを掛け合わせたものを，canvasに足し合わせていく．これは，各画素の正弦波を重ね合わせて，元の画像を復元する動作を表している．

    - ウィンドウを右クリックすることで，whileループを抜けて，復元した画像が表示される．

    - Enterキーが押されると，カメラ画像からの映像の取得が終了する仕組みとなっている．


- ウィンドウをクリックする回数を増やすことで，徐々に変換まえの画像に近づいていく様子が確認できた．

- 次に示すgifは，ウィンドウをクリックすることで得られた復元画像ではないが，forループによって，画像を復元する過程を示した物である．波動関数の合成により，徐々に、元の画像に近く様子が伺える．


![](fuji.gif)

- 実行環境は以下に示す通りである．
    - macOS Mojave version 10.14.3
    - Python 3.7.3
    - openCV version 3.4.2

- 参考文献
    1. [OpenCVで表示した画像にマウスクリックした場所を取得する方法 (Python) - 白猫学生のブログ](http://whitecat-student.hatenablog.com/entry/2016/11/09/225631)
        - openCVのウィンドウをマウスでクリックすることで画像上の座標を取得するための機能を実装する際に利用した．
    2. [DCTでlennaを再構成する](https://algorithm.joho.info/machine-learning/python-scikit-image-rgb2gray/)
        - DCTとIDCTを用いて、画像を変換、復元するためのソースコードの参考にした．

    3. [フーリエ変換 — OpenCV-Python Tutorials 1 documentation](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html)
        - pythonでフーリエ変換と逆変換を行うためには，ソースコードをどのように記述するべきなのかを調べるために利用した．
