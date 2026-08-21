# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリについて

移動マニピュレータによる書籍出納システム（マニピュレータ側）の研究用コード。RealSense D435i による認識、xArm7 によるアプローチ、Dynamixel ロボットハンドによる把持、IAI 直動シリンダー（昇降）、バーコード照合、AMR との UDP/ROS 2 連携を統合する。近年の実験対象は書籍からカテーテル箱に移っており、`master_*.json` の中身やアノテーションデータ（`anno-catheter/`）はカテーテル製品だが、コード上の語彙は「book」のまま。

**ほとんどのメインプログラムは実機（xArm7・RealSense・ハンド・昇降機構）が接続されていないと動かない。** オフライン検証には `detection/pro_handbook/sam_py_demo/offline_pointcloud_debug.py` などの保存データを使うスクリプトを使う。

ドキュメント・コメント・コミットメッセージは日本語が基本。

## 環境セットアップ

仮想環境が2つある点に注意:

```bash
# メイン環境（Python 3.10.12、リポジトリ直下）— ほぼすべてのスクリプトはこれで実行
cd ~/pro_book/pro_hand_book_python
source .pro_hand_book_fixed/bin/activate

# OCR 専用環境（PaddleOCR、依存が衝突するため分離されている）
cd detection/pro_handbook/sam_py_demo/OCR
source .paddle_ocr/bin/activate   # 旧名 .paadle_ocr は2026-07-14のマージで修正済み
```

メインプログラムは rclpy を使うので ROS 2 Humble も必要:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
```

**実行はすべてリポジトリルートから。** パッケージ相対 import が多く、単体スクリプトも `python -m package.module` 形式で起動する。

## 主要コマンド

### 統合動作（実機）

```bash
python Retrieval_integration.py                  # 取り出しメイン（設定は Retrieval_integration.yaml）
python Retrieval_integration_comntinuous.py      # 連続取り出し（ファイル名の綴りはこのまま）
python Strage_integration.py                     # 収納
ros2 launch retrieval_manager retrieval_auto.launch.py   # 20冊自動取り出し（上とセット）
ros2 launch udp_bridge_manip udp_bridge_manip.launch.py  # AMR との UDP ブリッジ
```

### 試験・評価

```bash
# 性能試験（認識＋実機取り出しを冊数×回数分）。中断後は --resume-dir で再開
python performance_test_runner.py --repeat-per-book 5 --order block
python performance_test_runner.py --resume-dir performance_tests/<run_id>

python recognition_accuracy_test.py              # 認識のみの精度試験
python capture_burst.py                          # 撮影のみ
python capture_anno.py                           # アノテーション用画像撮影（anno-catheter/ に保存）
```

### 単体動作確認

```bash
# ハンド各軸の位置確認（id_1: グリッパ, id_2: 回転, id_3: 直動）
python -m Dynamixel_win_pro_hand_book.id_1_gripper_pos_check
python calibrate_gripper.py                      # グリッパ開口幅キャリブレーション

# xArm7 の初期姿勢⇔撮影姿勢移動
python -m xarm7.control.xarm_init_to_capture
python -m xarm7.control.xarm_capture_to_init

# 昇降機構（ROS 2 ノード + トピック指令）
ros2 run iai_cylinder height_controller
ros2 topic pub --once /target_mm std_msgs/msg/Float32 "{data: 140.0}"
```

lint やユニットテストの仕組みはない。検証は実機試験と上記の評価スクリプトで行う。

## 設定

- `Retrieval_integration.yaml` — 中心的な設定ファイル。xArm7 の IP、対象 `shelf_id`（空にするとトピック受信待ち）、使用するマスタ JSON、waypoint ファイルパス、AMR 通信の UDP ポート群。
- `master_*.json` — 対象物マスタ（book_name / ISBN / bookshelf_ID / book_width[mm] など）。どれを使うかは yaml の `books.master_file` で切り替える。
- waypoint（教示点）は `ros2_ws/src/xarm7_teaching/config/*.yaml`。

## アーキテクチャ

取り出し1回の処理の流れ（`Retrieval_integration.py` の `main_sequence()` が統括）:

1. **撮影・認識**: `detection/pro_handbook/sam_py_demo/get_book_points.py` の `run_capture_and_pca()` — RealSense 撮影 → SAM（**SAM3サービス経由がデフォルト・唯一の実行パス**。`BOOK_SEGMENTATION_BACKEND`環境変数のデフォルト値が`"sam3"`固定で、それ以外を指定すると`RuntimeError`になる＝ONNX/SAM2への自動フォールバックは無効化済み。`models/sam_vit_h_*.onnx`を使う`infer_for_retrival.py`/`infer_for_storage.py`はレガシー経路で、"fixed-image parity"のサインオフまでの移行期参考実装として残っているだけで実行時には通らない。詳細は`get_book_points.py`の`_get_sam_runner_compat()`）でマスク → depth+intrinsics で点群化（Open3D）→ PCA で背表紙の姿勢・幅を推定。OCR による書名照合は `sam_py_demo/OCR/`（別 venv の PaddleOCR をワーカー経由で使う）。
2. **座標変換**: `xarm7/control/robot_base_coordinate.py`（`PoseChain`, `cam_mm_to_robot_mm`）でカメラ座標系→ロボット座標系。ハンドアイキャリブレーション関連は `xarm7/` 直下（`save_pose_pair.py`, `calibration_valid*.py`, `tcp_pivot_calibration.py`）。
3. **アーム制御**: `xarm7/control/xarm7.py`（`XArm7` クラス、xArm Python SDK は `xarm7/xArm-Python-SDK` に同梱）。waypoint 再生は `xarm_init_to_capture_integration.py` の `WaypointPlayerNode`。
4. **ハンド制御**: `Dynamixel_win_pro_hand_book/HandBook_Retrieval.py` / `HandBook_Storage.py`。ハンドは3軸（グリッパ・回転・直動）の Dynamixel 構成。
5. **昇降・照合・収納**: `linear_lift.py`（`/target_mm` へパブリッシュ）、`detection/.../bar_code/` のバーコード照合、`xarm7/control/book_return_sequence.py`。
6. **AMR 連携**: `ros2_ws/src/udp_bridge_manip` が ROS 2 トピック（`/shelf_id`, `/navigation_goal(_final)`, `/wall_distance`, `/cmd_vel` など）と UDP を橋渡し。単体テスト時は `ros2 topic pub` で到達信号等を手動送信できる（README 参照）。

### 安全機構（重要）

- `xarm7/control/xarm_monitor.py` の `XArmMonitor` は異常検知時に **`os._exit(1)` でプロセスを即時強制終了する**。例外では捕捉できない。ケース途中でプロセスが死ぬのは想定内の挙動で、`performance_test_runner.py` はこれを前提に progress.json と `--resume-dir` による再開・分類を実装している。
- `Retrieval_integration.py` は SIGINT でハンドを閉じてから xArm を緊急停止するハンドラを登録している。

## 認識精度の研究（catheter-100 SAM評価）

`catheter-100`データセット（棚画像5枚+正解アノテーション、`~/ダウンロード/catheter_test80/`側にある）に
対して本番SAMモデルで実際に認識を行い、IoU精度・書籍幅(px)精度・識別精度（`multikey_matcher.py`/
`match_eval.py`のマルチキー+ハンガリー法照合）を測定した。評価一式（Excel・オーバーレイ画像・
サマリ）や関連データセットは **`reco/`** 配下に集約している（`reco/README.md`が索引、
`reco/stand-100/`が今回の80件評価、`reco/diagonal-40/`が斜め置きデータセット）。
depth付き実撮影データも`reco/*/depth_shots/`に追加済みで、`get_book_points.py`本番の
実mm幅計測パイプライン（SAM3サービス経由）による検証も2026-08-21に実施済み（結果は
`reco/*/width_eval_summary.md`）。環境まわりの既知の地雷（旧`infer_for_retrival.py`の
importバグ、SAM3サービス設定パス破損、PaddleOCRモデル欠落等）は2026-08-21に修正済み。
詳細と再現手順は**`HANDOFF_20260820_catheter100_sam_eval.md`**を参照。

## その他の注意

- `control` はリポジトリ直下から `xarm7/control` へのシンボリックリンク。
- `*.bak_*` / `*_editing.py` / `*_revised.py` といったバックアップ・派生ファイルが多数あるが、実際に import されているのは無印版（例: `get_book_points.py`）。編集対象を間違えないこと。
- 出力先: 撮影は `captures/`、性能試験は `performance_tests/<run_id>/`、取り出しログは `logfile/retrieval_log.csv`（CSV 追記）。
