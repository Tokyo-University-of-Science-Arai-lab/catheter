# reco/ — カテーテル認識精度評価データセット置き場

このディレクトリ配下に、SAM認識精度検証用のデータセットをまとめている。
今後サンプルを追加する場合もこの下に置く。

## 命名規則

`reco/<配置条件>-<件数or枚数>/` 形式。

- `stand-100` : カテーテルを棚に立てて並べた状態。20種×5画像=100件(画像1,2,3,5は
  2026-08-20から、画像4は2026-08-21にアノテーション追加)。旧名`stand-80`(画像4除く
  80件時代の名残)から2026-08-21に改名。
- `diagonal-40` : カテーテルを斜めに倒した状態。1画像1品目、40枚(20種×L/R)。
  旧名`diagonal-100`(当初の目標件数の名残で実際は40枚だった)から2026-08-21に改名。

新しい撮影条件を追加する場合は `reco/<条件名>-<件数>/` として増やしていく。

認識精度Excelレポート（`catheter_recognition_report.xlsx`）を新しいデータセットにも作りたい場合の
再現手順は [`stand-100/scripts/README.md`](stand-100/scripts/README.md) を参照
（他のチャットから作業を引き継いでも同じ手順で再現できるようにするための記録）。

## 各データセット共通の構成

```
reco/<dataset>/
├── images/ (または images/default/)   COCOアノテーション対象のRGB画像
├── annotations/instances_default*.json COCO正解ポリゴン
├── depth_shots/<N>/                    RealSense実撮影(after_init_rgb.png +
│                                       after_init_depth.npy + camera_params.json)
├── outputs/, scripts/                  (stand-100のみ、80件評価一式)
└── README.md                           このデータセット固有の説明
```

マスタJSONは重複コピーをやめ、**`reco/master_catheter_reco.json` の1箇所のみ**に統一した
(2026-08-21)。中身は元の`master_catheter_20260216.json`と同じだが、「reco/配下の各データセットが
参照する共通マスタ」であることが名前だけで分かるよう、ユーザーが`_reco`付きの名前に変更した。
なお、実行可能な評価スクリプト(`~/ダウンロード/catheter_test80/`側)が実際に読み込むマスタは
そちらに残る`master_catheter_20260216.json`(元の日付入りファイル名)であり、これは別ファイル
(内容は同じ)。reco/側のものはあくまで参照・閲覧用。

## 重要: `images/` と `depth_shots/` は別カットではなく180度回転の関係

2026-08-21にユーザーから確認済み: `images/`(アノテーション用、人が見て正しい向き)と
`depth_shots/`(OCR用、depthと紐付いている)は同じ棚配置を撮った同一カットで、
**`depth_shots`側だけがOCR読み取りの都合で180度回転させてある**。今後の追加撮影でも
この運用(アノテーション用は正立、depth紐付きは180度回転)を継続する予定とのこと。

実際に画素比較で確認済み: `stand-100/images/{1,2,3,5}.png` と対応する
`stand-100/depth_shots/{1,2,3,5}/after_init_rgb.png` は `np.rot90(b, 2)` で
完全一致(mean_abs_diff=0.0)。`diagonal-40`側も同様。

**depth付きの実mm幅計測(`get_book_points.py`)にdepth_shotsを使う際の注意:**
`camera_params.json`のfx/fy/ppx/ppyは撮影時のセンサー本来の向き(=回転前)の値と
考えられる。180度回転後の画像・depthにそのままこの内部パラメータを適用すると、
ppx/ppyが画像中心からわずかにずれている分(実測: 1280x720に対しppx=637.8, ppy=371.0で
中心(639.5, 360)からx方向に約1.7px, y方向に約11pxのずれ)、単純に符号反転するだけでは
数mmオーダーの系統誤差が乗る可能性がある。

**対処方針**: `after_init_rgb.png`と`after_init_depth.npy`を`np.rot90(arr, 2)`で
180度回転させ「撮影時のセンサー向き」に戻してから`camera_params.json`をそのまま使って
3D復元・幅推定を行う(180度回転は自己逆変換なので、この戻し操作だけで済む)。
ppx/ppyを補正する方式は誤差が乗りやすく複雑なので採らない。可視化や既存の
アノテーション(正立向き)と重ねて見せたい場合のみ、最終的な出力画像を再度180度回転して
表示用に戻す。
