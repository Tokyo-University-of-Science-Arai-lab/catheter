import os
#バックエンド設定よりカメラ立ち上げを高速化
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
# os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Media Foundationの優先度を下げる
import cv2
import numpy as np
import glob
import sys
import csv
import time
from pathlib import Path
# sys.path.append('c:/Users/Ritsu6/program_python/UR_ramen_ver4_pra/src')
# sys.path.append('src/UR')
from coordinate_conversion import calculate_matrix
# from pupil_apriltags import Detector  #cv2のarucoライブラリで足りた


"""
皿位置をURベース座標で表示
"""

# --- 1. カメラキャリブレーション設定 ---
# チェスボードの内部コーナーの数（横方向、縦方向）
# CHECKERBOARD_SIZE = (9, 6) # 例: 横9個、縦6個のコーナー
CHECKERBOARD_SIZE = (10, 7) # 横11個、縦8個の四角(自分で印刷したとこ)
# SQUARE_SIZE = 25.0         # チェスボードの1マスのサイズ（mm単位など、任意）
SQUARE_SIZE = 0.024         # チェスボードの1マスのサイズ（m単位)(自分で印刷したとこ)


# キャリブレーション画像保存ディレクトリ
CALIBRATION_IMAGES_DIR = Path("/home/tai/pro_book/pro_hand_book_python/calibration")   #720p
# CALIBRATION_IMAGES_DIR = "calibration_images2"  #1080p

# キャリブレーション結果保存ディレクトリ
RESULT_of_CALIBRATION_DIR = Path("/home/tai/pro_book/pro_hand_book_python/calibration") 

# --- 2. AprilTag設定 ---
APRILTAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11 # 使用するAprilTagのファミリー,36h11になってる,arucoマーカー用の使ってるので直す
APRILTAG_SIZE = 0.060 # AprilTagの実際のサイズ（メートル単位、例: 50mm x 50mmなら0.05）
# APRILTAG_SIZE = 0.030 # AprilTagの実際のサイズ（メートル単位、例: 50mm x 50mmなら0.05）


# --- 3. その他の設定 ---
CAMERA_INDEX = 0 # 使用するWebカメラのインデックス (通常は0)

#--- 4. 解像度設定
# cap_width=640
cap_width = 3840
#cap_width=1920
# cap_height=480
cap_height= 2160
#cap_height=1080

# --- ユーティリティ関数 ---

def capture_calibration_images():
    """
    キャリブレーション用画像を複数枚撮影し、保存する関数
    """
    if not os.path.exists(CALIBRATION_IMAGES_DIR):
        os.makedirs(CALIBRATION_IMAGES_DIR)

    dev = find_video_by_name("Depstech")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

    # cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    # MJPGフォーマットを指定
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')) #←なんでこれonにしたらタグ位置推定バカ精度悪くなるの?????????????

    
    # カメラの自動露光調整をOFFに設定
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) #←1でいけるの？

    # カメラのバッファサイズを最小値に設定
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # カメラの明るさを調整
    cap.set(cv2.CAP_PROP_EXPOSURE,157) #←値が小さくなるほど早くなる？デフォルト157

    # MJPGの圧縮品質を設定（0～100、値が高いほど品質が高い）
    # cap.set(cv2.CAP_PROP_JPEG_QUALITY, 95)  # 100は最高品質
    cap.set(cv2.CAP_PROP_FPS, 15)  # FPSを下げて品質優先
    

    img_count = 0
    print("キャリブレーション画像撮影モードに入りました。")
    print(f"'{CALIBRATION_IMAGES_DIR}' フォルダに画像を保存します。")
    print("チェスボードを様々な角度から映し、's'キーで撮影、'q'キーで終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # チェスボードのコーナーを検出（ライブビューで確認するため）
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
        if ret_corners:
            cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners, ret_corners)

        cv2.imshow('Calibration Capture', frame)

        # # キャリブレーション結果画像の保存(実際にキャリブレーションする場合はコメントアウト？)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s'):
        #     img_filename = os.path.join(CALIBRATION_IMAGES_DIR, f"calibration_image_{img_count:03d}.png")
        #     cv2.imwrite(img_filename, frame)
        #     print(f"画像を保存しました: {img_filename}")
        #     img_count += 1
        # elif key == ord('q'):
        #     break

        # キャリブレーション画像の保存
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            img_filename = os.path.join(CALIBRATION_IMAGES_DIR, f"calibration_image_{img_count:03d}.png")
            cv2.imwrite(img_filename, gray) # グレースケール画像を保存
            print(f"画像を保存しました: {img_filename}")
            img_count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("キャリブレーション画像撮影を終了しました。")

def calibrate_camera():
    """
    保存されたキャリブレーション画像を用いてカメラをキャリブレーションする関数
    """
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    successful_detections = 0  # この行を追加

    images = glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.png'))

    if not images:
        print(f"'{CALIBRATION_IMAGES_DIR}' フォルダに画像がありません。先に画像を撮影してください。")
        return None, None

    for fname in images:
        img = cv2.imread(fname)
        # gray = img  # すでにグレースケールなので、cv2.cvtColorは不要
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #すでにグレースケールなのでいらない

        # 高解像度用のコーナー検出パラメータ
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, flags)

        if ret == True:
            # サブピクセル精度でコーナーを補正（高解像度では重要）
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
            objpoints.append(objp)
            imgpoints.append(corners)
            successful_detections += 1
        else:
            print(f"コーナー検出に失敗: {fname}")

    print(f"成功した検出数: {successful_detections}/{len(images)}")

    if successful_detections < 10:
        print("有効な画像が不足しています。最低10枚は必要です。")
        return None, None
    
    # キャリブレーションフラグの設定（高解像度用）
    calibration_flags = 0
    calibration_flags |= cv2.CALIB_RATIONAL_MODEL  # より正確な歪み補正
    calibration_flags |= cv2.CALIB_THIN_PRISM_MODEL  # 薄いプリズム歪み補正
    calibration_flags |= cv2.CALIB_FIX_K3  # 3次歪み係数を固定

    
    
    if not imgpoints:
        print("有効なチェスボードコーナーが検出された画像がありませんでした。")
        return None, None

    # カメラキャリブレーションを実行
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=calibration_flags) #高解像度の場合のカメラキャリブレーション

    if ret:
        print("\n--- カメラキャリブレーション結果 ---")
        print("内部パラメータ (カメラ行列):\n", mtx)
        print("歪み係数:\n", dist)
        print("再投影誤差:", ret)
        print(type(ret))
        print("-----------------------------------\n")
        """
        キャリブレーション結果を記録
        """
        if not os.path.exists(RESULT_of_CALIBRATION_DIR):
            os.makedirs(RESULT_of_CALIBRATION_DIR)
        calibration_result = os.path.join(RESULT_of_CALIBRATION_DIR, f"calibration_out_720p.csv") #720p用
        # calibration_result = os.path.join(RESULT_of_CALIBRATION_DIR, f"calibration_out_1080p.csv") #1080p用
        with open(calibration_result, 'w', encoding='utf-8', newline='') as f:
            dataWriter = csv.writer(f)
            dataWriter.writerow("カメラ行列")
            dataWriter.writerow(mtx[0])
            dataWriter.writerow(mtx[1])
            dataWriter.writerow(mtx[2])
            dataWriter.writerow("-------------------------------------")
            dataWriter.writerow("歪み係数")
            dataWriter.writerow(dist[0])
            dataWriter.writerow("--------------------------------------")
            dataWriter.writerow("再投影誤差")
            dataWriter.writerow([float(ret)])
            dataWriter.writerow("---------------------------------------")
        
        

        return mtx, dist
    else:
        print("カメラキャリブレーションに失敗しました。")
        return None, None

# new #
# def calibrate_camera():
#     objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

#     objpoints = []  # 3D points in real world space
#     imgpoints = []  # 2D points in image plane

#     images = glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.png'))

#     if not images:
#         print(f"'{CALIBRATION_IMAGES_DIR}' フォルダに画像がありません。先に画像を撮影してください。")
#         return None, None

#     for fname in images:
#         img = cv2.imread(fname)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # コーナー検出
#         ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

#         if ret:
#             # サブピクセル精度でコーナーを補正
#             criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#             corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

#             objpoints.append(objp)
#             imgpoints.append(corners)
#         else:
#             print(f"コーナー検出に失敗しました: {fname}")

#     if not imgpoints:
#         print("有効なチェスボードコーナーが検出された画像がありませんでした。")
#         return None, None

#     # カメラキャリブレーションを実行
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#     if ret:
#         print("\n--- カメラキャリブレーション結果 ---")
#         print("内部パラメータ (カメラ行列):\n", mtx)
#         print("歪み係数:\n", dist)
#         print("再投影誤差:", ret)
#         print("-----------------------------------\n")
#         return mtx, dist
#     else:
#         print("カメラキャリブレーションに失敗しました。")
#         return None, None
def find_video_by_name(keyword="Depstech") -> str:
    key = keyword.lower()
    for v in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_path = os.path.join(v, "name")
        try:
            name = open(name_path, "r").read().strip()
        except Exception:
            continue
        if key in name.lower():
            return "/dev/" + os.path.basename(v)
    raise RuntimeError(f'keyword "{keyword}" を含む video デバイスが見つかりません')

def open_depstech_stream(width=3840, height=2160, fps=30):
    dev = find_video_by_name("Depstech")

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 遅延が減りやすい

    # 本当に設定が効いたか確認（ここ超重要）
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
    print(f"[camera] dev={dev} FOURCC={fcc} "
          f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} "
          f"fps={cap.get(cv2.CAP_PROP_FPS)}")

    # 安定化：readより grab() で捨てた方が軽いことが多い
    for _ in range(30):
        cap.grab()

    return cap, dev

def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    """
    指定された回転ベクトル、平行移動ベクトルに基づいて座標軸を描画する関数
    """
    # 軸の3D座標 (X, Y, Z軸の先端)
    axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)

    # 3D点を2D画像座標に投影
    imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # 軸の描画
    origin_x, origin_y = int(imgpts[0][0][0]), int(imgpts[0][0][1])
    img = cv2.line(img, (origin_x, origin_y), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0, 0, 255), 5) # X軸 (赤)
    img = cv2.line(img, (origin_x, origin_y), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0, 255, 0), 5) # Y軸 (緑)
    img = cv2.line(img, (origin_x, origin_y), (int(imgpts[3][0][0]), int(imgpts[3][0][1])), (255, 0, 0), 5) # Z軸 (青)
    return img

def rvec_to_euler_angles(rvec):

    """
    rvecsをロール，ピッチ，ヨーに変換
    """
    # 回転ベクトルを回転行列に変換
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 回転行列からオイラー角を計算（Z-Y-X順）
    # sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    s = np.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2)

    singular = s < 1e-6  # 特異点の判定

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # x軸回り（ロール）
        pitch = np.arctan2(-rotation_matrix[2, 0], s)                 # Y軸回り（ピッチ）
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) # z軸回り（ヨー）
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], s)
        yaw = 0

    # ラジアンを度に変換
    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)

    return roll, pitch, yaw

def main():
    
    # --- カメラキャリブレーションの実行 ---
    # 既にキャリブレーション画像があるか確認し、なければ撮影を促す
    if not os.path.exists(CALIBRATION_IMAGES_DIR) or not glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.png')):
        print("キャリブレーション画像がありません。画像を撮影します。")
        capture_calibration_images()
    
    if not os.path.exists(RESULT_of_CALIBRATION_DIR) or not glob.glob(os.path.join(RESULT_of_CALIBRATION_DIR, '*720p.csv')): #720p用
    # if not os.path.exists(RESULT_of_CALIBRATION_DIR) or not glob.glob(os.path.join(RESULT_of_CALIBRATION_DIR, '*1080p.csv')): #1080p用
        print("キャリブレーション結果が保存されていません。計算を行います。")
        camera_matrix, dist_coeffs = calibrate_camera()
    
    

    #--- キャリブレーション結果の読み込み ---
    
    calibration_result = os.path.join(RESULT_of_CALIBRATION_DIR, f"calibration_out_720p.csv") #720p用
    # calibration_result = os.path.join(RESULT_of_CALIBRATION_DIR, f"calibration_out_1080p.csv") #1080p用
    with open(calibration_result, 'r', encoding='utf-8', newline='') as f:
        dataReader = csv.reader(f)
        i_cali=1
        camera_matrix = np.zeros((3,3))
        # dist_coeffs = np.zeros((1,5))
        dist_coeffs = np.zeros((1,12))
        for row in dataReader:
            if i_cali>1 and i_cali<5:
                camera_matrix[int(i_cali)-2]=[float(value) for value in row]
            elif i_cali==7:
                dist_coeffs[0] = [float(value) for value in row]
            i_cali+=1
        # print('camera_matrix\n',camera_matrix)
        # print('dist_coeffs=\n',dist_coeffs)
    cap, dev = open_depstech_stream(width=cap_width, height=cap_height, fps=30)

    if camera_matrix is None or dist_coeffs is None:
        print("カメラキャリブレーションが完了していないため、プログラムを終了します。")
        return
    

    # --- AprilTag検出の準備 ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILY)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


    # cap = cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_MSMF)
    #cap = cv2.VideoCapture(CAMERA_INDEX)
    # if not cap.isOpened():
    #     print("カメラを開けませんでした。")
    #     return

    # #解像度設定
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # # カメラの自動露光調整をOFFに設定
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 

    # # カメラの明るさを調整
    # cap.set(cv2.CAP_PROP_EXPOSURE, 157) 

    # # H264フォーマットを指定←MJPGに変更
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')) #←なんでこれonにしたらタグ位置推定バカ精度悪くなるの?????????????
    # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))
    # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('Y','U','Y','2')) 
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
    # カメラのバッファサイズを最小値に設定
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    """
    # MJPGの圧縮品質を設定（0～100、値が高いほど品質が高い）    
    cap.set(cv2.CAP_PROP_JPEG_QUALITY, 100)  # 100は最高品質
    if cap.set(cv2.CAP_PROP_JPEG_QUALITY, 80):  # 100は最高品質
        print("MJPGの圧縮品質を設定しました。")
    else:
        print("MJPGの圧縮品質設定に失敗しました。")
    """

    #カメラのfps設定
    # cap.set(cv2.CAP_PROP_FPS, 15)  # カメラのFPSを30に設定

    print("AprilTag検出モードに入りました。'q'キーで終了します。")


    while True:   
        
        start_time1 = time.time()  # 処理開始時間
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret:
            break
        
        elapsed_time1 = time.time() - start_time1  # 処理時間を計算

        start_time2 = time.time()  # 処理開始時間
        t2 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # AprilTagの検出
        corners, ids, rejected = detector.detectMarkers(gray)
        t2 = time.perf_counter()

        print("read:", round((t1-t0)*1000,2), "ms",
        "detect:", round((t2-t1)*1000,2), "ms",
        "total:", round((t2-t0)*1000,2), "ms",
        "fps:", round(1.0/(t2-t0),2))
        if ids is not None:
            # 検出された各AprilTagについて処理
            for i in range(len(ids)):
                

                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], APRILTAG_SIZE, camera_matrix, dist_coeffs)
                April_rotation_matrix, _=cv2.Rodrigues(rvecs) #ロドリゲスの回転ベクトル→回転行列に変換
                rvec, tvec = rvecs[0], tvecs[0]

                # AprilTagの中心をカメラ座標で表示
                # tvecはすでにカメラ座標系での平行移動ベクトル（X, Y, Z）
                tag_center_camera_x = tvec[0][0]
                tag_center_camera_y = tvec[0][1]
                tag_center_camera_z = tvec[0][2]

                # 例: rvecsからロール、ピッチ、ヨーを計算
                roll, pitch, yaw = rvec_to_euler_angles(rvec)

                # R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列
                # R_T = R.T
                # rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0]) #ロール，ピッチ，ヨー
                # # rvecはカメラ座標での各軸の回転角度
                # tag_angle_camera_x = rvec[0][2]* (180 / np.pi) #np.linalg.norm()でノルムを回転角度に変更
                # tag_angle_camera_y = rvec[0][1]* (180 / np.pi)
                # tag_angle_camera_z = rvec[0][0]* (180 / np.pi)

                # 座標軸を描画
                frame = draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, length=APRILTAG_SIZE * 0.5)

                # 検出したAprilTagの枠とIDを描画
                cv2.polylines(frame, [np.int32(corners[i])], True, (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {ids[i][0]}", (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # AprilTagの中心座標を画像上に表示
                # text_x = f"X: {tag_center_camera_x:.3f}m"
                # text_y = f"Y: {tag_center_camera_y:.3f}m"
                # text_z = f"Z: {tag_center_camera_z:.3f}m"
                # mm表記
                # text_x = f"X: {tag_center_camera_x*10**3:.0f}mm"
                # text_y = f"Y: {tag_center_camera_y*10**3:.0f}mm"
                # text_z = f"Z: {tag_center_camera_z*10**3:.0f}mm"

                #回転角度も表示
                text_x = f"X: {tag_center_camera_x*10**3:.0f}mm {roll:.1f}deg"
                text_y = f"Y: {tag_center_camera_y*10**3:.0f}mm {pitch:.1f}deg"
                text_z = f"Z: {tag_center_camera_z*10**3:.0f}mm {yaw:.1f}deg"

                
                cv2.putText(frame, text_x, (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, text_y, (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, text_z, (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
                """
                URベース座標から見た皿位置
                """
                P = calculate_matrix()
                camera_coordinate = np.array([[tag_center_camera_x*10**3,tag_center_camera_y*10**3,tag_center_camera_z*10**3]]).T #カメラから取得したマーカーのカメラ座標を入手
                marker_coordinate1 = np.array([[0,0,500]]).T #マーカー上の座標からみた皿1の座標
                marker_coordinate2 = np.array([[50,50,500]]).T #マーカー上の座標からみた皿2の座標

                camera_roll_pitch_yow_A = {"rx":90,"ry":0,"rz":180}  #URAのベース座標からみたカメラ座標の回転角度(事前に設定)
                camera_roll_pitch_yow_B = {"rx":-90,"ry":90,"rz":0}  #URBのベース座標からみたカメラ座標の回転角度(事前に設定)
                camera_roll_pitch_yow_C = {"rx":-90,"ry":-90,"rz":0}  #URCのベース座標からみたカメラ座標の回転角度(事前に設定)

                camera_rotating_matrix_A = P.rotating_matrix(camera_roll_pitch_yow_A["rx"],camera_roll_pitch_yow_A["ry"],camera_roll_pitch_yow_A["rz"]) #回転行列を計算
                camera_rotating_matrix_B = P.rotating_matrix(camera_roll_pitch_yow_B["rx"],camera_roll_pitch_yow_B["ry"],camera_roll_pitch_yow_B["rz"]) #回転行列を計算
                camera_rotating_matrix_C = P.rotating_matrix(camera_roll_pitch_yow_C["rx"],camera_roll_pitch_yow_C["ry"],camera_roll_pitch_yow_C["rz"]) #回転行列を計算

                camera_t_A = np.array([[-187.94,-475.44,53.95]]).T  #URAのベース座標からみたカメラ座標の並進ベクトル(事前に設定)
                # camera_t_B = np.array([[-187.94,-475.44,93.95]]).T  #URBのベース座標からみたカメラ座標の並進ベクトル(事前に設定)
                camera_t_B = np.array([[182.94,481.44,30.95]]).T  #URBのベース座標からみたカメラ座標の並進ベクトル(事前に設定),修正
                camera_t_C = np.array([[0,0,-200]]).T  #URCのベース座標からみたカメラ座標の並進ベクトル(事前に設定)

                URA_coordinate_1 = P.coordinate_conversion(P.coordinate_conversion(marker_coordinate1,April_rotation_matrix,camera_coordinate),camera_rotating_matrix_A,camera_t_A) #URAからみた皿1の位置
                URA_coordinate_2 = P.coordinate_conversion(P.coordinate_conversion(marker_coordinate2,April_rotation_matrix,camera_coordinate),camera_rotating_matrix_A,camera_t_A) #URAからみた皿2の位置
                URB_coordinate_1 = P.coordinate_conversion(P.coordinate_conversion(marker_coordinate1,April_rotation_matrix,camera_coordinate),camera_rotating_matrix_B,camera_t_B) #URBからみた皿1の位置
                URB_coordinate_2 = P.coordinate_conversion(P.coordinate_conversion(marker_coordinate2,April_rotation_matrix,camera_coordinate),camera_rotating_matrix_B,camera_t_B) #URBからみた皿2の位置
                URC_coordinate_1 = P.coordinate_conversion(P.coordinate_conversion(marker_coordinate1,April_rotation_matrix,camera_coordinate),camera_rotating_matrix_C,camera_t_C) #URCからみた皿1の位置
                URC_coordinate_2 = P.coordinate_conversion(P.coordinate_conversion(marker_coordinate2,April_rotation_matrix,camera_coordinate),camera_rotating_matrix_C,camera_t_C) #URCからみた皿2の位置

                elapsed_time2 = time.time() - start_time2  # 処理時間を計算
                print("画像処理時間 = ",round(elapsed_time1*1000,3),"ms")
                print("計算時間 = ",round(elapsed_time2*1000,3),"ms")
                print("URAからみた皿の位置1 = \n", np.round(URA_coordinate_1,2))
                print("URAからみた皿の位置2 = \n", np.round(URA_coordinate_2,2))

        cv2.imshow('AprilTag Pose Estimation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()