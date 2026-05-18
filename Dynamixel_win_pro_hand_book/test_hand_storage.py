import time
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook


# 収納動作で使っていた仮のスペーサ回転角
TEST_ROT_DEG = 80.0


def main():
    Hand = None

    try:
        print("start hand-only storage test")

        # Dynamixel 初期化
        Hand = HandBook.init_dynamixels()

        # 1. スペーサ前進
        input("press enter to expand spacer linear:")
        HandBook.expand_sp_lin(dxl=Hand, asynchronous=False)

        # 2. グリッパ全開
        input("press enter to open gripper full:")
        HandBook.open_until_full(dxl=Hand, asynchronous=False)

        # 3. グリッパ把持
        input("press enter to close gripper:")
        HandBook.grasp(dxl=Hand)

        # 4. スペーサ回転
        input("press enter to rotate spacer:")
        HandBook.rotate_spacer(Hand, TEST_ROT_DEG)

        # 5. スペーサ回転リセット
        input("press enter to reset spacer rotation:")
        HandBook.reset_rot(Hand)

        # 6. スペーサ後退 1段目
        input("press enter to contract spacer linear 1:")
        HandBook.contract_sp_lin_1(Hand, asynchronous=False)

        # 7. スペーサ後退 2段目
        input("press enter to contract spacer linear 2:")
        HandBook.contract_sp_lin_2(Hand, asynchronous=False)

        # 8. グリッパ開放
        input("press enter to ungrasp:")
        HandBook.ungrasp_auto(Hand)

        print("hand-only sequence done")

        # 終了処理
        Hand.disable_torque(HandBook.GRIPPER_ID)
        Hand.disable_torque(HandBook.SP_LIN_ID)
        Hand.disable_torque(HandBook.SP_ROT_ID)
        time.sleep(0.2)
        Hand.close_port()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stopping hand")

        if Hand is not None:
            try:
                Hand.disable_torque(HandBook.GRIPPER_ID)
                Hand.disable_torque(HandBook.SP_LIN_ID)
                Hand.disable_torque(HandBook.SP_ROT_ID)
                time.sleep(0.2)
                Hand.close_port()
            except Exception as e:
                print(f"cleanup error: {e}")

        print("dynamixel deactivated")


if __name__ == "__main__":
    main()