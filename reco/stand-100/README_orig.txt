80-reco : catheter-100 の4画像(1,2,3,5 / 画像4は破損のため除外) x 20クエリ = 80件 認識精度検証

■ フォルダ構成
  input/
    images/                         評価対象の4画像 (1.png, 2.png, 3.png, 5.png)
    annotations/instances_default_80.json
                                     元の instances_default.json から画像4を除いた
                                     正解アノテーション(COCO形式, 80件)
    master_catheter_20260216.json   製品マスタ(book_name/display_name/book_width[mm]等)

  outputs/
    catheter_recognition_report.xlsx  レビュー用Excel(80行、列は全てソート可、
                                       「正誤」列は手入力用T/Fプルダウン。改善後とベースライン両方の識別結果を列で比較可能)
    per_instance.csv                  Excelの元になった集計データ(CSV)
    summary.md                        IoU精度/幅精度(px)/識別精度のサマリ
    overlays/{1,2,3,5}.png            画像全体にGT(緑)/予測(赤)を重ねたQC画像
    mask_overlays/                    インスタンスごとの拡大クロップ画像(78枚、Excelの
                                       「認識番号」列と対応。ファイル名は
                                       <認識番号>_img<画像番号>.png)
    pred_masks_cache/{1,2,3,5}.npz    SAM予測マスクのキャッシュ

  scripts/
    run_sam_eval.py, build_excel_report.py  今回の集計・Excel生成に使ったスクリプト
    compare_quad_fit.py, mask_rectify.py, match_eval.py, quad_fit.py,
    multikey_matcher.py                     上記が依存する既存ロジック(参照用コピー)
    ※これらは元々 ~/ダウンロード/catheter_test80/catheter/scripts/ にあり、パス関係が
      そこを前提にしているため、このフォルダに置いた状態のままでは再実行できません。
      再実行する場合は元の場所のスクリプトを使ってください(このコピーは参照用です)。

■ 元データの場所(オリジナル、変更なし)
  ~/ダウンロード/catheter_test80/catheter-100/          画像・正解アノテーション・マスタ(5画像分)
  ~/ダウンロード/catheter_test80/catheter/outputs/sam_recognition_eval/
                                                          今回の生成物のオリジナル(5画像分、画像4含む)

[2026-08-21 注記] このファイルの内容はより新しい README.md に統合されている
(フォルダ構成は reco/ 配下への再編後のものに更新済み)。このファイル自体は
オリジナルの記録として残置。
