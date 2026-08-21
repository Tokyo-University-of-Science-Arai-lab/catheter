# 実mm幅計測 検証結果 (2026-08-21)

`get_book_points.py`本番の3D計測パイプラインを`depth_shots/`(40件、1画像1品目)に対して
実行し、マスタJSONの`book_width`(mm)と比較した。手法・環境修正の詳細は
[`../stand-100/width_eval_summary.md`](../stand-100/width_eval_summary.md)を参照(同じ
スクリプト・同じ修正を適用)。

結果生データ: [`width_eval_result.csv`](width_eval_result.csv)

目視レビュー用Excel・画像は [`width_eval_report/catheter_width_report.xlsx`](width_eval_report/catheter_width_report.xlsx) /
[`width_eval_report/images/`](width_eval_report/images/)（このデータセット専用。stand-100とは
フォルダを分けてある）。

## 結果(40件、全件成功・エラー0件)

**誤差2mm以内が実機把持成功の目安**（ユーザーの経験則、2026-08-21確認）なので、
まずこの数字を見ること。

| 指標 | 値 |
|---|---|
| **2mm以内(把持成功の目安)** | **3/40 (7.5%)** |
| 5mm未満 | 13/40 (32.5%) |
| 10mm未満 | 30/40 (75%) |
| MAE | 9.96 mm |
| 中央値 | 6.28 mm |

`stand-100`(直立配置、100件、MAE 9.06mm)とほぼ同水準。斜め姿勢固有の劣化ではなく、
このパイプライン全体の一般的な精度限界とみられる。

個別に誤差が大きかった項目(参考、debug出力は`width_eval_work/<shot>/`に残っている):
AXS_DAC_L(49.0mm)、Target_R(39.5mm)、orange_R(32.3mm)、neuroform_L(25.6mm)、
orange_L(21.6mm)、pNOVUS_L(21.4mm)。同一品目のL/R間で誤差の差が大きいケースが複数あり
(例: AXS_DAC L=49.0mm vs R=13.7mm)、置き方向(L/R)によって軸検出の安定性が変わる可能性が
ある。
