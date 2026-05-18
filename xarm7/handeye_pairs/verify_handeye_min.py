#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from control.ar_marker_pose import rs_color_K_dist, aruco_marker_pose_target2cam


def now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_h(p):
    return np.array([p[0], p[1], p[2], 1.0])


def load_T_tcp_cam(path, swap):
    d = json.load(open(path))
    T_cam_tcp = np.array(d["T_cam_tcp"])
    T_tcp_cam = np.array(d["T_tcp_cam"])
    return T_cam_tcp if swap else T_tcp_cam


def rpy2R(roll, pitch, yaw):
    rr, rp, ry = np.deg2rad([roll, pitch, yaw])
    cz, sz = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rp), np.sin(rp)
    cx, sx = np.cos(rr), np.sin(rr)

    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx


def build_T_base_tcp(pose):
    x,y,z,r,p,yaw = pose
    T = np.eye(4)
    T[:3,:3] = rpy2R(r,p,yaw)
    T[:3,3] = np.array([x,y,z])/1000.0
    return T


def detect_marker(pipe, K, dist, marker_len, dict_name, marker_id, avg, timeout, preview):
    ts=[]
    t0=time.time()

    while len(ts)<avg:
        if time.time()-t0>timeout:
            raise RuntimeError("marker lost")

        frames=pipe.wait_for_frames()
        color=frames.get_color_frame()
        if not color:
            continue

        bgr=np.asanyarray(color.get_data())
        vis=bgr.copy()

        ret=aruco_marker_pose_target2cam(
            bgr=bgr,K=K,dist=dist,
            marker_len_m=marker_len,
            dict_name=dict_name,
            target_id=marker_id
        )

        if ret is None:
            if preview:
                cv2.putText(vis,"no marker",(20,40),0,1,(0,0,255),2)
                cv2.imshow("v",vis); cv2.waitKey(1)
            continue

        mid,rvec,tvec=ret
        ts.append(tvec)

        if preview:
            cv2.drawFrameAxes(vis,K,dist,rvec.reshape(3,1),tvec.reshape(3,1),0.05)
            cv2.imshow("v",vis); cv2.waitKey(1)

    return np.mean(ts,axis=0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--handeye-json",required=True)
    ap.add_argument("--swap",action="store_true")
    ap.add_argument("--trials",type=int,default=10)
    ap.add_argument("--avg-n",type=int,default=10)
    ap.add_argument("--z-offset-mm",type=float,default=50)
    ap.add_argument("--preview",action="store_true")
    ap.add_argument("--return-view", action="store_true", default=True,
                    help="after each trial, return to initial view pose (default: on)")
    ap.add_argument("--speed", type=float, default=80)
    ap.add_argument("--mvacc", type=float, default=500)
    args=ap.parse_args()

    T_tcp_cam=load_T_tcp_cam(args.handeye_json,args.swap)

    pipe=rs.pipeline()
    cfg=rs.config()
    cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
    prof=pipe.start(cfg)
    intr,K,dist=rs_color_K_dist(prof)

    from xarm.wrapper import XArmAPI
    arm=XArmAPI("192.168.1.197")
    arm.connect(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)

    errors=[]

    for i in range(args.trials):

        input(f"\n置き直してEnter {i+1}/{args.trials}")

        # ====== 0) 初期撮影姿勢（観測姿勢）を保存 ======
        code_v, pose_view = arm.get_position(is_radian=False)
        if code_v != 0 or pose_view is None:
            raise RuntimeError(f"get_position(view) failed: code={code_v}, pose={pose_view}")
        pose_view = [float(v) for v in pose_view]  # [x,y,z,roll,pitch,yaw]

        # ====== 1) 観測（view poseでマーカー中心） ======
        o_cam = detect_marker(pipe, K, dist, 0.05, "DICT_4X4_50", 0, args.avg_n, 10, args.preview)

        # view poseでPoを計算
        T_base_tcp = build_T_base_tcp(pose_view)
        T_base_cam = T_base_tcp @ T_tcp_cam
        Po = (T_base_cam @ to_h(o_cam))[:3] * 1000  # mm

        # ====== 2) リーチ（roll/pitch/yawはviewのまま） ======
        cmd = Po.copy()
        cmd[2] += args.z_offset_mm

        ret = arm.set_position(
            float(cmd[0]), float(cmd[1]), float(cmd[2]),
            float(pose_view[3]), float(pose_view[4]), float(pose_view[5]),
            speed=args.speed, mvacc=args.mvacc, wait=True
        )
        if ret != 0:
            print("[WARN] set_position(reach) ret=", ret)
        time.sleep(0.5)

        # ====== 3) リーチ位置で再観測（Po'） ======
        o_cam2 = detect_marker(pipe, K, dist, 0.05, "DICT_4X4_50", 0, args.avg_n, 10, args.preview)

        code2, pose2 = arm.get_position(is_radian=False)
        if code2 != 0 or pose2 is None:
            raise RuntimeError(f"get_position(after reach) failed: code={code2}, pose={pose2}")
        pose2 = [float(v) for v in pose2]

        T_base_tcp2 = build_T_base_tcp(pose2)
        T_base_cam2 = T_base_tcp2 @ T_tcp_cam
        Po2 = (T_base_cam2 @ to_h(o_cam2))[:3] * 1000  # mm

        # ====== 4) 誤差 ======
        err = float(np.linalg.norm(Po2 - Po))
        errors.append(err)
        print("error(mm):", err)

        # ====== 5) 初期撮影姿勢に戻る ======
        if args.return_view:
            print("[INFO] return to view pose")
            ret3 = arm.set_position(
                float(pose_view[0]), float(pose_view[1]), float(pose_view[2]),
                float(pose_view[3]), float(pose_view[4]), float(pose_view[5]),
                speed=args.speed, mvacc=args.mvacc, wait=True
            )
            if ret3 != 0:
                print("[WARN] set_position(return) ret=", ret3)
            time.sleep(0.3)

    print("\nmean:", float(np.mean(errors)), "max:", float(np.max(errors)))

    pipe.stop(); arm.disconnect()


if __name__=="__main__":
    main()



if __name__=="__main__":
    main()
