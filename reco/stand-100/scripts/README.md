# 認識精度Excelレポート（catheter_recognition_report.xlsx）の再現手順

2026-08-20に作成した目視レビュー用Excel（`outputs/catheter_recognition_report.xlsx`）を、
他のチャット/セッションでも同じ手順で作れるようにする記録。

## これは何をするスクリプトか

`reco/stand-100`（棚画像＋COCO正解アノテーション＋マスタJSON）のようなデータセットに対して
**実際にSAM(本番の`sam_vit_h` ONNXモデル)で認識を実行**し、

1. IoU精度（予測マスク vs 正解ポリゴン、ハンガリー法でマッチング）
2. 書籍幅の精度（px。depth無しのため参考値としてmm換算も出す）
3. 識別精度（OCR結果を`multikey_matcher.py`/`match_eval.py`の手法＝マルチキー+ハンガリー法割当
   でマスタと照合。ベースライン(REF単独+独立argmax)との比較列も出す）

を算出し、行ごとにソート可能なExcel + インスタンスごとの拡大レビュー画像を出力する。

## 前提条件

- メインvenv: `cd ~/pro_book/pro_hand_book_python && source .pro_hand_book_fixed/bin/activate`
  （2026-08-21にvenvのactivateスクリプト自体の壊れたパスを修正済みなので、素の`python3`が
  そのまま使える。以前使っていた`.pro_hand_book_fixed/bin/python3`直接パス指定の回避策は不要）
- 導入済みパッケージ: onnxruntime-gpu, opencv-contrib-python, scipy, rapidfuzz, openpyxl, pillow, numpy
- SAMモデル: `pro_hand_book_python/models/sam_vit_h_4b8939.{encoder,decoder}.onnx`
- OCR結果（事前生成済みのものを読むだけ、このスクリプト自体はOCRを実行しない）:
  `catheter/outputs/ocr/<image_id>.json`
- 参照識別結果（GTマスクベースのOCR識別結果、読むだけ）:
  `catheter/outputs/match_multi/assignment.csv`

## 実行場所についての重要な注意（パス依存あり）

このディレクトリ（`reco/stand-100/scripts/`）にあるファイルは**参照用コピー**。
`Path(__file__).resolve().parents[N]`でデータセットのパスを組み立てる作りのため、
**実行は実体である `~/ダウンロード/catheter_test80/catheter/scripts/` から行うこと。**
ここに置いたコピーのまま実行しても動かない。

## 手順

```bash
cd ~/pro_book/pro_hand_book_python
source .pro_hand_book_fixed/bin/activate

cd ~/ダウンロード/catheter_test80/catheter/scripts
python3 run_sam_eval.py         # 1. SAM推論(GPU) + IoU/幅(px)/識別精度を集計
                                 #    → catheter/outputs/sam_recognition_eval/summary.md, per_instance.csv
python3 build_excel_report.py   # 2. 上記の結果からExcel + マスク重畳クロップ画像を生成
```

- `run_sam_eval.py`は`pred_masks_cache/<image_id>.npz`にSAM推論結果をキャッシュする。
  2回目以降（アノテーションだけ更新した等）はGPU推論をスキップして高速に再集計できる。
- 評価対象画像を絞りたい場合は、両スクリプト冒頭の`EXCLUDED_IMAGE_IDS = {...}`を編集する。
  **【2026-08-21実施済み】** 画像4のアノテーションが揃ったため、`ANNOTATIONS_JSON`を
  `catheter-100/annotations/instances_default_100.json`（画像4の20件をマージした版、
  `reco/stand-100/annotations/instances_default_100.json`と同一内容をコピー）に切り替え、
  `EXCLUDED_IMAGE_IDS`を空集合にして100件で再実行した。ただし**画像4だけIoU≥0.5マッチが
  1/20と極端に低い**結果になっており(他画像は18〜20/20)、原因未調査（詳細は
  `../README.md`の該当セクション）。

## 出力

```
catheter/outputs/sam_recognition_eval/
├── catheter_recognition_report.xlsx
│     【2026-08-21変更】画像(認識)ごとにシートを分割(img1, img2, img3, img4, img5)。
│     以前は全画像を1シートにまとめていたが、認識バッチごとに見やすくするため分割した。
│     各シートの列: 認識番号 / 画像番号 / display_name / book_name / book_width_mm /
│         推定幅_mm / 推定幅誤差_mm / 正誤(空欄、T/Fプルダウンで手入力) / スコア /
│         認識した文字列 / IoU / 参考_GTマスクでの識別結果 / 参考識別との一致 /
│         book_name_ベースライン(現行相当) / スコア_ベースライン / 参考識別との一致_ベースライン /
│         オーバーレイ画像
│     全列オートフィルタでソート・絞り込み可能。1行目固定。
├── mask_overlays/<認識番号>-<画像番号>-<display_name>.png
│     インスタンスごとの拡大重畳画像（緑=正解輪郭, 赤=予測輪郭）。
│     display_nameが空の場合は`unknown`（＝識別失敗、詳細はHANDOFF参照）。
├── per_instance.csv, summary.md, identification_ablation.md
├── overlays/<image_id>.png            画像全体のQC用オーバーレイ
└── pred_masks_cache/<image_id>.npz    SAM予測マスクのキャッシュ
```

reco側へ持ってくる場合は、上記`outputs/`一式と`per_instance.csv`等を
`reco/<dataset>/outputs/`へコピーする（このリポジトリでは`reco/stand-100/outputs/`）。

## 別のデータセット（例: diagonal-40）に応用する場合

`run_sam_eval.py` / `build_excel_report.py` 冒頭の `SRC` / `ANNOTATIONS_JSON` / `MASTER_JSON`
等のパス定数は現状catheter-100(stand-100)専用にハードコードされている。他データセットに使う
場合はこれらを書き換えるか、スクリプトをコピーしてデータセットごとに定数を変える。

## 依存している既存ロジック（同ディレクトリに参照用コピーあり）

- `compare_quad_fit.py` — COCOセグメンテーション（ポリゴン/非圧縮RLE）→0/1マスク変換、IoU計算
- `mask_rectify.py` / `quad_fit.py` — マスクの矩形化（min_area_rect、幅(px)算出に使用）
- `match_eval.py` — OCRテキストとマスタのファジーマッチングスコア（`key_score`）
- `multikey_matcher.py` — マルチキー+ハンガリー法の本番用ドロップイン実装
  （`build_excel_report.py`自体は上記`match_eval.key_score`+`scipy`のハンガリー法を直接使っており、
  このファイルを直接importはしていないが、同じ手法の実装として参照している）

詳しい経緯・環境の地雷・結果数値は
[`../../../HANDOFF_20260820_catheter100_sam_eval.md`](../../../HANDOFF_20260820_catheter100_sam_eval.md)
を参照。
