# diagonal-40 (旧 diagonal-100 / reco-diagonal-40)

**【2026-08-21訂正】** 当初「1画像1品目を斜めに倒して撮影」と記載していたが誤り。
実際にdepth_shots/annotations双方の生画像を直接確認したところ、**stand-100と同じ
20品目が並ぶ棚全体の写真**だった(1品目だけを写した写真ではない)。しかも全20品目が
傾いているわけではなく、**一部の箱(Surpass Evolve, GuidePost, TransForm305,
オレンジ無地箱等)だけが将棋倒し状に斜めに傾き、残りは直立したまま**という混在状態
(pro-book-cd確認)。つまり「query→棚全体から該当する1箱を選ぶ」というタスク構造自体は
stand-100と同一で、違うのは一部の箱の姿勢(直立 vs 傾斜)だけ。
20種×L/R(カメラ位置違いと思われる)=40ショット、という構成。
2026-08-21に `reco/` 配下へ再編し、depth付き実撮影(`depth_shots/`)を追加した。
フォルダ名の"100"は当初の目標件数の名残で、現状の実ファイル数は40枚(images/default,
annotations, depth_shotsとも40で整合済み)。

## フォルダ構成

```
images/default/*.png                 評価対象の画像40枚 (<product>_{L,R}.png、
                                     アノテーション用に正立させたもの)
annotations/instances_default.json   COCO形式の正解ポリゴン(40画像)
depth_shots/<product>_{L,R}/         RealSenseで実撮影したdepth付きデータ(2026-08-21追加)
                                     各フォルダに <product>_{L,R}.png(OCR用に180度回転済み)
                                     + after_init_depth.npy + camera_params.json。
                                     images/default/の対応画像と180度回転の関係にある
                                     (../README.md参照)。元は撮影順の通し番号(1〜42、歯抜け)
                                     だったが、2026-08-21に中の画像ファイル名(品目名)を
                                     フォルダ名に採用する形にリネームした。その際、
                                     depth_shots側だけにあった表記ゆれ・誤字を
                                     images/default側の(正しい)表記に合わせて修正した:
                                     "Eccelsior XT-27"→"Excelsior XT-27"、
                                     "SHORYU2"/"SYORYU2"→"SHOURYU2"、
                                     "Transform305__R"(二重アンダースコア)→"Transform305_R"。
                                     images/default側とannotations側は元々表記が正しく、
                                     変更していない。
```

まだ評価スクリプト(`outputs/`, `scripts/`)は作成していない。`stand-100/`のIoU/幅/識別
評価と同じ手法(棚全体+query)がそのまま適用できる見込み(データセット構造は実質同一)。

マスタJSONはこのフォルダには置かず、`reco/master_catheter_reco.json` を参照する。
