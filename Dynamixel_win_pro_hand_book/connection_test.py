from dynamixel_sdk import *

port = PortHandler('/dev/ttyUSB0')
packet = PacketHandler(2.0)

if not port.openPort():
    print("ポート開けない")
    exit()

for baud in [57600, 115200, 1000000]:
    print(f"\n=== baud: {baud} ===")
    port.setBaudRate(baud)

    for i in range(0, 20):
        model, res, err = packet.ping(port, i)
        if res == 0:
            print(f"見つかった: ID={i}, model={model}")