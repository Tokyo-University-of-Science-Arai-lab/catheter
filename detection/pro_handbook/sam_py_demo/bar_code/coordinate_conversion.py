import math
import numpy as np
import time

"""
回転行列の計算クラス
"""
class calculate_matrix:
    def __init__(self):
        self.theta = None
        self.Sin = None
        self.Cos= None
        self.x = None
        self.y = None
        self.z = None
        self.rx = None
        self.ry = None
        self.rz = None

    #角度［°］からsinを計算
    def S(self,theta):
        self.Sin = math.sin(np.radians(theta))
        return self.Sin
    
    #角度［°］からcosを計算
    def C(self,theta):
        self.Cos = math.cos(np.radians(theta))
        return self.Cos
    
    #x軸にθ回転したときの回転行列
    def R_x(self,theta):
        self.Sin = self.S(theta)
        self.Cos = self.C(theta)
        R_x = np.zeros((3,3))
        R_x[0]=[1,0,0]
        R_x[1]=[0,self.Cos,-self.Sin]
        R_x[2]=[0,self.Sin,self.Cos]
        return R_x
    
    #y軸にθ回転したときの回転行列
    def R_y(self,theta):
        self.Sin = self.S(theta)
        self.Cos = self.C(theta)
        R_y = np.zeros((3,3))
        R_y[0]=[self.Cos,0,self.Sin]
        R_y[1]=[0,1,0]
        R_y[2]=[-self.Sin,0,self.Cos]
        return R_y
    
    #z軸にθ回転したときの回転行列
    def R_z(self,theta):
        self.Sin = self.S(theta)
        self.Cos = self.C(theta)
        R_z = np.zeros((3,3))
        R_z[0]=[self.Cos,-self.Sin,0]
        R_z[1]=[self.Sin,self.Cos,0]
        R_z[2]=[0,0,1]
        return R_z
    
    #ヨー、ロー、ピッチを回転させたときの回転行列
    def rotating_matrix(self,rx,ry,rz):
        self.rx = rx
        self.ry = ry
        self.rz = rz
        R = np.dot(self.R_x(self.rx),np.dot(self.R_y(self.ry),self.R_z(self.rz)))
        return R
    
    #座標変換
    """
    座標系nからみた座標n+1の位置の計算
    """
    def coordinate_conversion(self,X_1,R,t):
        self.X_1 = X_1  #座標系n+1でみた位置
        self.R = R  #回転行列(3×3)
        self.t = t  #並進ベクトル(3×1)
        X_0 = np.dot(self.R,X_1)+self.t
        return X_0

        

    

"""
デバッグ用1
"""
# def main():
#     """
#     クラス内のメソッドの確認→ok
#     """
#     theta = 45
#     rx=45
#     ry=45
#     rz=45
#     cm = calculate_matrix()
#     S = cm.S(theta)
#     C = cm.C(theta)
#     R_x = cm.R_x(theta)
#     R_y = cm.R_y(theta)
#     R_z = cm.R_z(theta)
#     R = cm.rotating_matrix(rx,ry,rz)
#     print("Sin(",theta,"°) = ",S)
#     print("Cos(",theta,"°) = ",C)
#     print("R_x = \n",R_x)
#     print("R_y = \n",R_y)
#     print("R_z = \n",R_z)
#     print("回転行列 = \n",R)

""""
デバッグ用2
"""
# def main():
#     """
#     a→b→cへの座標変換→ok
#     """
#     #座標系aからみたbの回転
#     rx=0
#     ry=0
#     rz=45
#     #座標系bからみたcの回転
#     rx1=45
#     ry1=0
#     rz1=0
#     c=np.array([[1,0,0]]).T #座標系cからみた点の位置
#     t = np.array([[0,0,1]]).T   #座標系aからbへの並進ベクトル
#     t1 = np.array([[1,0,1]]).T  #座標系bからcへの並進ベクトル
#     cm = calculate_matrix()     #インスタンス作成
#     #回転行列
#     R = cm.rotating_matrix(rx,ry,rz)
#     R1 = cm.rotating_matrix(rx1,ry1,rz1)
#     # x = cm.coordinate_conversion(cm.coordinate_conversion(c,R1,t1),R,t)

#     #座標変換
#     b = cm.coordinate_conversion(c,R1,t1)    #b→c
#     a = cm.coordinate_conversion(b,R,t)  #a→b
#     print("ワールド座標での位置(理論値) = \n",np.array([[math.sqrt(2),math.sqrt(2),2]]).T)
#     print("ワールド座標での位置(計算値) = \n",a)

"""
デバッグ用3
"""
def main():
    """
    UR→カメラ座標
    """
    start_time = time.time()  # 処理開始時間

    P = calculate_matrix()
    camera_coordinate = np.array([[10,10,300]]).T #カメラから取得したマーカーのカメラ座標を入手
    marker_coordinate1 = np.array([[0,0,500]]).T #マーカー上の座標からみた皿1の座標
    marker_coordinate2 = np.array([[50,50,500]]).T #マーカー上の座標からみた皿2の座標

    camera_roll_pitch_yow_A = {"rx":90,"ry":0,"rz":180}  #URAのベース座標からみたカメラ座標の回転角度(事前に設定)
    camera_roll_pitch_yow_B = {"rx":-90,"ry":90,"rz":0}  #URBのベース座標からみたカメラ座標の回転角度(事前に設定)
    camera_roll_pitch_yow_C = {"rx":-90,"ry":-90,"rz":0}  #URCのベース座標からみたカメラ座標の回転角度(事前に設定)

    camera_rotating_matrix_A = P.rotating_matrix(camera_roll_pitch_yow_A["rx"],camera_roll_pitch_yow_A["ry"],camera_roll_pitch_yow_A["rz"]) #回転行列を計算
    camera_rotating_matrix_B = P.rotating_matrix(camera_roll_pitch_yow_B["rx"],camera_roll_pitch_yow_B["ry"],camera_roll_pitch_yow_B["rz"]) #回転行列を計算
    camera_rotating_matrix_C = P.rotating_matrix(camera_roll_pitch_yow_C["rx"],camera_roll_pitch_yow_C["ry"],camera_roll_pitch_yow_C["rz"]) #回転行列を計算

    # camera_t_A = np.array([[0,0,-200]]).T  #URAのベース座標からみたカメラ座標の並進ベクトル(事前に設定)
    camera_t_A = np.array([[0,0,-200]]).T  #URAのベース座標からみたカメラ座標の並進ベクトル(事前に設定)
    camera_t_B = np.array([[0,0,-200]]).T  #URBのベース座標からみたカメラ座標の並進ベクトル(事前に設定)
    camera_t_C = np.array([[0,0,-200]]).T  #URCのベース座標からみたカメラ座標の並進ベクトル(事前に設定)

    URA_coordinate = P.coordinate_conversion(camera_coordinate,camera_rotating_matrix_A,camera_t_A)
    URB_coordinate = P.coordinate_conversion(camera_coordinate,camera_rotating_matrix_B,camera_t_B)
    URC_coordinate = P.coordinate_conversion(camera_coordinate,camera_rotating_matrix_C,camera_t_C)

    elapsed_time = time.time() - start_time  # 処理時間を計算

    """
    確認
    """
    # np.set_printoptions(precision=3)
    print("処理時間 = ",elapsed_time)
    print("表示されるはずのURAからみたマーカー中心の位置 = \n",np.array([[0,-300,-200]]).T)
    print("URAからみたマーカー中心の位置 = \n", np.round(URA_coordinate,3))
    print("表示されるはずのURBからみたマーカー中心の位置 = \n",np.array([[300,0,-200]]).T)
    print("URBからみたマーカー中心の位置 = \n", np.round(URB_coordinate,3))
    print("表示されるはずのURCからみたマーカー中心の位置 = \n",np.array([[-300,0,-200]]).T)
    print("URCからみたマーカー中心の位置 = \n", np.round(URC_coordinate,3))


"""
デバッグ用4
"""

# def main():
#     """
#     UR→カメラ座標→tag座標→皿位置
#     """
#     start_time = time.time()  # 処理開始時間


#     elapsed_time = time.time() - start_time  # 処理時間を計算



    





if __name__ == "__main__":
    main()  