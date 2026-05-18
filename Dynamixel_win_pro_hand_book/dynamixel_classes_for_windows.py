"""
Windows環境用Dynamixel制御クラス
ポートの占有特性がUbuntuと異なり1モーター1クラスができないため、id周りが少し異なる。
"""

import os
import msvcrt
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import ctypes

class Dynamixel:  # This class is specified in X_series
    def __init__(self, port, baudrate):
        """インスタンス関数
        ポートを指定して通信を確立する

        Parameters
        ----------
        port : int
            ポート名 (Windowsは通常「"COM?"」)
        baudrate : int
            通信速度 (通常 )
        """
        def getch():
            return msvcrt.getch().decode()


        # ********* DYNAMIXEL Model definition *********
        # ***** (Use only one definition at a time) *****
        # X330 (5.0 V recommended), X430, X540, 2X430
        self.__MY_DXL = "X_SERIES"
        self.__ADDR_TORQUE_ENABLE = 64
        self.__ADDR_OPERATION_MODE = 11

        self.__ADDR_VELOCITY_LIMIT = 44
        self.__ADDR_GOAL_VELOCITY = 104
        self.__ADDR_PRESENT_VELOCITY = 128

        self.__ADDR_MAXIMUM_POSITION = 48
        self.__ADDR_MINIMUM_POSITION = 52
        self.__ADDR_GOAL_POSITION    = 116
        self.__ADDR_PRESENT_POSITION = 132


        self.__BAUDRATE = baudrate
        # DYNAMIXEL Protocol Version (1.0 / 2.0)
        # https://emanual.robotis.com/docs/en/dxl/protocol2/
        self.__PROTOCOL_VERSION = 2.0

        # Use the actual port assigned to the U2D2.
        # ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
        self.__DEVICENAME = port

        self.__TORQUE_ENABLE = 1  # Value for enabling the torque
        self.__TORQUE_DISABLE = 0  # Value for disabling the torque

        self.__VELOCITY_MODE = 1  # Velocity Mode
        self.__POSITION_MODE = 3  # Position Mode
        self.__EX_POSITION_MODE = 4  # Extended Position Mode

        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.__portHandler = PortHandler(self.__DEVICENAME)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.__packetHandler = PacketHandler(self.__PROTOCOL_VERSION)

        # Open port
        if self.__portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()

        # Set port baudrate
        if self.__portHandler.setBaudRate(self.__BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()



    def set_mode_velocity(self,id):
        """速度制御モードに設定

        Parameter
        ----------
        id : int
            IDを指定
        """
        dxl_comm_result, dxl_error = self.__packetHandler.write1ByteTxRx(
            self.__portHandler,
            id,
            self.__ADDR_OPERATION_MODE,
            self.__VELOCITY_MODE,
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))

        # Initialize Goal Velocity
        self.write_velocity(0,0)


    def set_mode_position(self,id):
        """角度制御モードに設定

        Parameter
        ----------
        id : int
            IDを指定
        """
        dxl_comm_result, dxl_error = self.__packetHandler.write1ByteTxRx(
            self.__portHandler,
            id,
            self.__ADDR_OPERATION_MODE,
            self.__POSITION_MODE,
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))

    def set_mode_ex_position(self,id):
        """拡張角度制御モードに設定

        Parameter
        ----------
        id : int
            IDを指定
        """
        dxl_comm_result, dxl_error = self.__packetHandler.write1ByteTxRx(
            self.__portHandler,
            id,
            self.__ADDR_OPERATION_MODE,
            self.__EX_POSITION_MODE,
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))

    def set_max_velocity(self,id,max_velocity):
        """速度制御時最大回転速度を設定

        Parameters
        ------------
        id : int
            IDを指定
        max_velocity : int
            最大回転速度 (通常265)

        """
        dxl_comm_result, dxl_error = self.__packetHandler.write4ByteTxRx(
            self.__portHandler,
            id,
            self.__ADDR_VELOCITY_LIMIT,
            max_velocity,
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))
        print("SET MAX VELOCITY")

    def set_min_max_position(self,id,min_position,max_position):
        """角度制御時最小最大位置を設定

        Parameter
        ----------
        id : int
            IDを指定
        min_position :
            最小位置 (通常 0)
        max_position :
            最大位置 (通常 4095)
        """
        # Write Min Position Limit
        dxl_comm_result, dxl_error = self.__packetHandler.write4ByteTxRx(self.__portHandler, id, self.__ADDR_MINIMUM_POSITION, min_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))
        print("SET MIN POSITION")

        # Write Max Position Limit
        dxl_comm_result, dxl_error = self.__packetHandler.write4ByteTxRx(self.__portHandler, id, self.__ADDR_MAXIMUM_POSITION, max_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))
        print("SET MAX POSITION")




    def enable_torque(self,id):
        """トルクをオンにする

        Parameter
        ----------
        id : int
            IDを指定
        """
        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.__packetHandler.write1ByteTxRx(
            self.__portHandler,
            id,
            self.__ADDR_TORQUE_ENABLE,
            self.__TORQUE_ENABLE,
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))

    def disable_torque(self,id):
        """トルクをオフにする

        Parameter
        ----------
        id : int
            IDを指定
        """
        dxl_comm_result, dxl_error = self.__packetHandler.write1ByteTxRx(
            self.__portHandler,
            id,
            self.__ADDR_TORQUE_ENABLE,
            self.__TORQUE_DISABLE,
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))

    def write_velocity(self,id, vel):
        """目標回転速度を送信する

        Parameter
        ----------
        id : int
            IDを指定
        vel : int
            最大速度以下の速度を指定する (ex. -128 ~ 128 )
        """
        dxl_comm_result, dxl_error = self.__packetHandler.write4ByteTxRx(
            self.__portHandler,
            id,
            self.__ADDR_GOAL_VELOCITY,
            vel,
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))
        print("SET POSITION")
        print("[ID:%03d]  GoalVel:%03d" % (id, ctypes.c_int32(vel).value))

    def read_velocity(self,id):
        """現在回転速度を受信する

        Parameter
        ----------
        id : int
            IDを指定

        Returns
        -------
        dxl_present_velocity : int
            現在の回転速度 (ex. -128 ~ 128 )
        """
        (
            dxl_present_velocity,
            dxl_comm_result,
            dxl_error,
        ) = self.__packetHandler.read4ByteTxRx(
            self.__portHandler, id, self.__ADDR_PRESENT_VELOCITY
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))

        velocity_c_int32 = ctypes.c_int32(dxl_present_velocity).value
        print(
            "[ID:%03d]  PresVel:%03d"
            % (id, velocity_c_int32)
        )

        return velocity_c_int32

    def write_position(self,id,pos):
        """現在位置を受信する # 目標位置を送信するじゃ？

        Parameter
        ----------
        id : int
            IDを指定
        pos : int
            目標位置 ( ex. 0 ~ 4095 )
            0(MinPositionLimit)~1(MaxPositionLimit)で示される位置の値
        """
        dxl_comm_result, dxl_error = self.__packetHandler.write4ByteTxRx(self.__portHandler, id, self.__ADDR_GOAL_POSITION, pos)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))
        print("SET POSITION")
        print("[ID:%03d]  GoalPos:%03d" % (id, ctypes.c_int32(pos).value))


    def read_position(self,id):
        """現在の位置の読みとり

        Parameter
        ----------
        id : int
            IDを指定

        Returns
        -------
        dxl_present_position : int
            0~4095までの値
        """
        dxl_present_position, dxl_comm_result, dxl_error = self.__packetHandler.read4ByteTxRx(self.__portHandler, id, self.__ADDR_PRESENT_POSITION)

        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.__packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.__packetHandler.getRxPacketError(dxl_error))

        position_c_int32 = ctypes.c_int32(dxl_present_position).value
        print("[ID:%03d]  PresPos:%03d" % (id, position_c_int32))

        return position_c_int32

    def close_port(self):
        self.__portHandler.closePort()
