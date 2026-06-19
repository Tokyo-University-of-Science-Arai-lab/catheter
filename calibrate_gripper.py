import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook
import numpy as np
import time


def main():
    # キャリブレーション中は補正なし（YAML は変更しない）
    HandBook.GRIPPER_CALIB_A = 1.0
    HandBook.GRIPPER_CALIB_B = 0.0

    dxl = HandBook.init_dynamixels()

    targets = list(range(10, 40, 2))  # 10, 12, 14, ..., 38 mm

    commanded = []
    measured_list = []

    print("===== グリッパ開口幅キャリブレーション =====")
    print(f"測定点数: {len(targets)} 点  ({targets[0]}〜{targets[-1]} mm, 2mm 間隔)")
    print("各ステップでグリッパを開いた後、実測値を入力してください。\n")

    try:
        for i, target in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] 目標幅 {target} mm で開きます...")
            HandBook.open_until_width(dxl, float(target), gravity=False)

            while True:
                raw = input(f"  実測値を mm で入力 (目標: {target} mm): ").strip()
                try:
                    val = float(raw)
                    break
                except ValueError:
                    print("  数値を入力してください")

            commanded.append(float(target))
            measured_list.append(val)
            print(f"  記録済み: 目標={target} mm, 実測={val} mm")

            print("  グリッパを閉じています...")
            HandBook.grasp(dxl)
            time.sleep(0.3)
            print()

    finally:
        dxl.disable_torque(HandBook.GRIPPER_ID)
        dxl.close_port()

    # 線形回帰
    x = np.array(commanded)
    y = np.array(measured_list)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    residuals = y - y_pred

    print("\n===== 測定データ =====")
    print(f"{'目標 [mm]':>12}  {'実測 [mm]':>10}  {'残差 [mm]':>10}")
    for cx, cy, r in zip(commanded, measured_list, residuals):
        print(f"{cx:>12.1f}  {cy:>10.1f}  {r:>+10.3f}")

    print("\n===== キャリブレーション結果 =====")
    print(f"近似式: 実測値 = {slope:.6f} × 目標値 + ({intercept:.6f})")
    print()
    print("Dynamixel_config.yaml に設定する値:")
    print(f"  calib_width_a: {slope:.6f}")
    print(f"  calib_width_b: {intercept:.6f}")


if __name__ == "__main__":
    main()
