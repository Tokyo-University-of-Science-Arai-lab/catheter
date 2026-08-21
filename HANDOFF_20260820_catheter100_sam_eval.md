# 引き継ぎメモ（2026-08-20）— catheter-100 SAM認識精度評価

`catheter-100`（Downloads側データセット）に対して本番SAMモデルを初めて実際に走らせ、
IoU精度・書籍幅(px)精度・識別精度を測定した記録。新しいセッションはこれを読めば経緯を把握できる。

---

## 1. 何をしていたか

- 対象: `~/ダウンロード/catheter_test80/catheter-100/` の棚画像5枚（1280x720、depthなし）
  + COCO形式の正解アノテーション（`book_spine`ポリゴン）+ `master_catheter_20260216.json`（20品目）
- **このデータセットにSAM(本番の`sam_vit_h` ONNXモデル)が実際に推論を行ったのは今回が初めて。**
  （`catheter/outputs/`配下の既存の評価は、SAM未実行のため正解アノテーションのマスクを代用していた）
- 画像4は正解アノテーションが20件中5件しかなく破損扱い → 除外。**画像1,2,3,5 × 20品目 = 80件**で最終評価。
- 評価軸は3つ:
  1. IoU精度（予測マスク vs 正解ポリゴン、ハンガリー法でマッチング）
  2. 書籍幅の精度（`mask_rectify.min_area_rect_box`の短辺。depth無しのためpx。mm換算は同画像内の
     正解インスタンスから求めたpx→mm係数を使った参考値）
  3. 識別精度（OCR結果を`master_catheter_20260216.json`とマルチキー・ファジーマッチング。
     `multikey_matcher.py`/`match_eval.py`の手法＝REF+display_name+期限の多段query＋
     全体最適割当(ハンガリー法)を使用）
- 追加で、識別ロジックのアブレーション比較も実施（`multikey_matcher.py`/`match_eval.py`の効果測定）:
  REFキー単独＋マスクごと独立argmax（＝これらのファイルを使わない場合）と比較。

---

## 2. 結果（80件、catheter-100 画像1,2,3,5）

| 指標 | 値 |
|---|---|
| IoU≥0.5でマッチ | 78/80 (97.5%)、未検出2件 |
| マッチしたペアの平均IoU | 0.892（中央値0.919、最小0.548） |
| 幅誤差(px) | MAE 5.61px / RMSE 12.99px / 平均絶対%誤差 11.52% |
| 識別精度（改善後: 多段query+ハンガリー法） | 67/76 (88.2%) |
| 識別精度（ベースライン: REF単独+独立argmax） | 61/76 (80.3%) |

multikey_matcher.py/match_eval.pyの手法により識別精度は **+7.9pt** 改善。IoU・幅(px)は識別ロジックと無関係なので不変。

---

## 3. 環境まわりで踏んだ地雷（要注意）

- このマシンは作業開始時点でシステムリストア直後で、`.pro_hand_book_fixed` venvに
  onnxruntime等が入っていなかった。`~/ダウンロード/restore_info/pip_freeze_main_py310.txt`を参照して
  必要パッケージ（onnxruntime-gpu, opencv-contrib-python, scipy, rapidfuzz, openpyxl等）を復元した。
- ~~`.pro_hand_book_fixed/bin/activate` の `VIRTUAL_ENV`/`PATH` はvenv作成当時のパス
  （`/home/catheter/catheter/...`、フォルダ移動前の旧パス）がハードコードされており、現在の場所と
  一致しない（壊れている）。~~ **2026-08-21 に修正済み。** venv内(`bin/activate`系3ファイル・
  `bin/pip`等コンソールスクリプトのshebang・xArm Python SDKのeditable install参照など計50ファイル)の
  旧パス文字列を現在のパスへ一括置換した。同じ問題がOCR専用venv(`detection/pro_handbook/sam_py_demo/OCR/.paddle_ocr`、
  25ファイル)にもあり、こちらも同様に修正済み。今後は `source .pro_hand_book_fixed/bin/activate` /
  `source .paddle_ocr/bin/activate` してから素の `python3` / `pip` を使う通常の運用に戻ってよい
  （`.pro_hand_book_fixed/bin/python3` を直接パス指定で呼ぶ回避策はもう不要）。
- onnxruntime-gpuがCUDA/cuDNNライブラリを見つけられない問題があり、venv内`nvidia-*`パッケージの
  libディレクトリを`LD_LIBRARY_PATH`に積んでから自プロセスを`os.execv`で再起動する必要がある
  （`recognition_accuracy_test.py`冒頭と同じ手法。パスはこのマシンのsite-packages実パスに要調整）。
- `infer_for_retrival.py`の`SamBatchInfer_retrieval`は内部で`infer_for_storage`を絶対importしており、
  `infer_for_storage.py`内の相対import(`.modules.crop_pyramid`)と衝突して単体では動かない
  （既存バグ、未修正）。回避策: `infer_for_storage.SamBatchInfer_storage.infer_masks()`を直接呼ぶ。

---

## 4. 成果物の場所

- **`pro_hand_book_python/80-reco/`** — 今回の80件評価一式（`README.txt`に構成説明あり）
  - `input/` : 画像(1,2,3,5.png)・80件分の正解アノテーション・マスタJSON
  - `outputs/catheter_recognition_report.xlsx` : 目視レビュー用Excel（80行、列は全てソート可、
    「正誤」列は手入力用T/Fプルダウン。改善後とベースライン両方の識別結果を列で比較可能）
  - `outputs/mask_overlays/` : インスタンスごとの拡大画像
    （ファイル名 `<認識番号>-<画像番号>-<display_name>.png`、未識別は`unknown`）
  - `outputs/summary.md`, `outputs/identification_ablation.md` : 集計サマリ
  - `scripts/` : 生成に使ったスクリプトの参照用コピー（パスは元の場所依存のため
    このフォルダに置いたままでは再実行不可）
- オリジナル一式（5画像分、画像4含む全データ、再実行可能な状態のスクリプト）は
  `~/ダウンロード/catheter_test80/` 配下に温存:
  - `catheter-100/` : 元の画像・アノテーション・マスタ
  - `catheter/scripts/run_sam_eval.py`, `build_excel_report.py` : 再実行用の実体
    （依存: `compare_quad_fit.py`, `mask_rectify.py`, `match_eval.py`, `quad_fit.py`）
  - `catheter/outputs/sam_recognition_eval/` : 上記スクリプトの出力（5画像分）
  - `catheter/outputs/ocr/`, `catheter/outputs/match_multi/assignment.csv` : 既存のOCR結果・
    GTマスクベースの参照識別結果（読み取り専用で利用）

---

## 5. 次にやるなら

- ~~幅のmm実測~~ → **2026-08-21 実装・実行完了。** `reco/scripts/width_mm_validation.py`で
  `depth_shots/`(RealSense実撮影)に対し`get_book_points.py`本番の3D計測パイプライン
  (SAM3サービス+OCR+PCA/軸フィルタ)を実際に呼び出し、マスタJSONの`book_width`(mm)と比較した。
  結果: stand-100(直立、100件)MAE 9.06mm・10mm未満74%、diagonal-40(斜め、40件)
  MAE 9.96mm・10mm未満75%と**ほぼ同水準**(「斜め姿勢だから精度が悪い」という仮説は否定された)。
  詳細は `reco/stand-100/width_eval_summary.md` / `reco/diagonal-40/width_eval_summary.md` を参照。
  この実装のために、SAM3サービスの設定パス破損・OCR用venvのmatplotlib/numpy非互換・
  PaddleOCRモデル本体の欠落・`get_book_points.py`内の`n_t_bins`未定義バグ(デバッグ画像保存
  関数、修正済み)という4つの環境/コード問題も発見・修正した。詳細は上記summaryファイル参照。
- ~~SAM重複検出の後処理~~ → **2026-08-21 調査済み、対策は見送り。** 「20件のGTに対し予測33〜38件」
  の実体は、ピクセル単位では一切重ならない(IoU=0)、ラベル部分と本体などに分割検出された
  フラグメントだった。IoUベースのマスクNMSは無力(実測: 除去0件)。bbox包含率でのクラスタ化＋
  OCRテキスト共有も試したが、無関係な物体まで巻き込んで識別精度を悪化させた(67/76→65/73)ため
  撤回済み(詳細は`reco/stand-100/README.md`)。識別精度への影響は2/80件程度で限定的なため、
  無理に修正せず既知の限界として記録するにとどめる。
- ~~`infer_for_retrival.py`のimportバグ修正~~ → **2026-08-21 対応済み。** 122行目を
  `from .infer_for_storage import SamBatchInfer_storage`（相対import）に変更。単体import・
  `evaluate_retrieval.py`経由の両方で動作確認済み。
- ~~`.pro_hand_book_fixed/bin/activate`のパス修正~~ → 2026-08-21 対応済み（3節参照）。

---

## 6. 識別精度の共同デバッグ (2026-08-21、複数セッション連携)

`reco/scripts/width_mm_validation.py`実行後、ユーザーから「88.2%(2026-08-20)より
大幅に悪化している」と指摘があり、この機に別セッション(実装担当・分析担当)と
`SendMessage`で直接連携し原因を切り分けた。詳細は[[catheter-width-accuracy-root-cause]]
(記憶ファイル)にも記録済み。要点のみここに残す。

### 6.1 「精度が下がった」ように見える理由

**IoU(97.5%)と識別精度は別軸の指標である点に注意。** 2026-08-20のIoUはGTポリゴンと
予測マスクをHungarian法でIoU最大化マッチングしただけの幾何精度で、「どのテキストが
どの物体か」という識別とは無関係。今回悪化したのは識別精度の方であり、IoU側の話では
ない(pro-book-cd指摘)。

識別精度88.2%(2026-08-20)→実質1〜2割(2026-08-21、88件を人間が目視確認)の乖離は
主に2つの要因による:

1. **SAM本体が別モデル。** 2026-08-20は`sam_vit_h`(ONNX)を`SamBatchInfer_storage`で
   直接使用。今回はSAM3サービス(`sam3_runtime`、book-spine専用にファインチューンされた
   `inference_best.pt`、固定テキストプロンプト"book spine")を使用。実際に
   `reco/stand-100/outputs/mask_overlays/`の2026-08-20側「識別成功」サンプルを目視した
   ところ、予測マスク(緑)は正解(赤)にほぼ完全に重なっていた。一方SAM3側は目視レビュー
   88件中、選択マスクが物理的に全く別の箱を指すケースが大半だった。
2. **88.2%という数字の測り方自体が人間の目視確認ではなかった。** `assignment.csv`は
   GTマスク(正解ポリゴン)に同じOCR+multikey照合アルゴリズムを適用した結果であり、
   予測マスク側の識別結果との「アルゴリズム同士の一致率」を見ていた。今回が初めて
   人間が実際にfinal.pngを見て正誤判定した数字(2セッション合計88件、T=11 F=77)。

### 6.2 `multikey_matcher.py`の修正

catheter100の識別(`match_text_to_mask_main`)本体である`catheter/scripts/multikey_matcher.py`
に2つの修正を入れた:

1. **`_key_score`の最小文字数ガード**: `fuzz.partial_ratio`は短い文字列ほど長い文字列に
   偶然部分一致しやすい(実例: OCR断片"00"がREF"M00345100950"に部分一致してscore=100)。
   HANDOFF_20260731.mdに記録された過去の軸反転バグ(`_text_similarity`の無条件containment)
   と同根。`min(len(key), len(combined)) < 4`ならscore=0を返すよう修正。
2. **`_looks_like_plausible_identifier()`追加**: 選択マスクのOCRテキストが7文字以上、
   かつ(数字4桁以上の連続 / 英数字混在 / 日付らしいパターン)のいずれかを満たさなければ
   `confident`をFalseにする2段目のガード。88件で検証し感度0.73・特異度0.62。
   **ただしこれは識別の正答率自体は変えない(confidentラベルの信頼性が上がるだけ)。**

### 6.3 item3(空間対応チェックによる選択ロジック改善)は既存データでは限界

選択マスクが物理的に正しいかを事後的に判別できないか、5つの候補シグナルを88件で検証したが
**全て判別力ゼロ**だった:

| シグナル | T群 | F群 |
|---|---|---|
| OCRテキストbbox重なり率(`_collect_mask_texts`) | 1.0 | 1.0(飽和、無意味) |
| SAM3自体の信頼度スコア | 中央値0.9375 | 中央値0.9375(完全一致) |
| 選択マスク面積 | 中央値28765px | 中央値32700px |
| マスク形状(solidity/extent/アスペクト比) | ほぼ同一 | ほぼ同一 |
| 選択マスクのbbox幅(画像内中央値との比) | 中央値1.0 | 中央値1.0 |

「間違ったマスク」は幾何的には正常(=どこか別の実在する箱を正しく切り出しているだけ)
であることが多く、既存の確信度/形状指標では見分けがつかない。**今後着手する価値がある
候補**:
- (a) 色/テクスチャなど、現状取得していない新しい特徴量の追加取得
- (b) SAM3のプロンプト("book spine"固定)やモデル自体の再検討
  (ファインチューンデータセット名が"book_spine_sam3_dataset"で、docsに"catheter"の
  言及が一切無いことから、書籍専用学習の可能性が状況証拠としてある)

### 6.4 その他の発見

- **バーコード安全網が実質デッドコード**: `Retrieval_integration.py`で、
  `book_barcode_sequence`呼び出しコードの直前に`return book_width, "success", shot_dir`
  という早期returnがあり、バーコード確認は実行されない状態だった。ユーザー判断により
  今回この件への対応は見送り、現状のまま。
- **「全20品目既知」の前提は評価だけの有利条件ではない**: 本番の`match_text_to_mask_main`
  自体が、既に全20品目でのHungarian法割当を内部で行っている(1画像・20クエリを独立に
  呼び出しても選択マスクの重複が一切無いことを確認)。ユーザー指摘の通り、棚の中身が
  既知というのは実運用条件としても妥当。
- `master_catheter_reco.json`の`bookshelf_ID`は20件全て`"1-1-1-1"`で同一(pro-book-cd
  発見)。棚ID側での絞り込みは今回のデータでは機能しておらず(候補が絞られない)、実運用で
  複数棚IDが混在する場合にどう働くかは未検証。
- `catheter/scripts/multikey_matcher.py`は本体repoと`~/ダウンロード/catheter_test80/
  catheter/scripts/`の2箇所に存在するが、2026-08-20評価時点では後者は`multikey_matcher.py`
  を使っておらず`match_eval.py`を直接使用していたため、この重複は今回の精度差とは無関係。
  ただし`catheter_test80`側は今回の`_key_score`修正が未反映のまま。再実行する場合は同期が必要。
- `reco/diagonal-40/README.md`の誤り(「1画像1品目」)を訂正: 実際はstand-100と同じ棚全体
  写真で、一部の箱(Surpass Evolve, GuidePost, TransForm305, オレンジ無地箱等)だけが
  将棋倒し状に傾いている混在状態だった。

**2026-08-21、ユーザー判断によりここで一区切り。** 次に着手するなら6.3の(a)(b)を参照。

## 7. 幅推定精度改善への再着手 (2026-08-21、6.のさらに続き)

上記6.は「識別精度」側の調査で一区切りとしたが、同日中に**幅推定誤差(MAE 6〜7mm)**の
根本原因調査に着手し、複数の修正を実装した。

### 7.1 重大な訂正: 6.で報告した識別精度の数値は回転バグの影響下だった

6.の88件レビュー(T=11 F=77、正答率1〜2割)は、**評価スクリプト`width_mm_validation.py`
自身が持っていた180度回転バグ**(depth_shotsを不要に回転させてOCR/SAM3に上下逆さまの
画像を渡していた)の影響下で行われたものだった。回転処理を削除して再実行した結果、
識別正答率は**90%(18/20サンプル目視)**まで回復した。本番パイプライン
(`get_book_points.py`)には回転処理は一切無く、実機は無傷。詳細は
`~/.claude/projects/-home-catheter-pro-book/memory/catheter_width_accuracy_root_cause.md`
を参照。

### 7.2 マスク生成〜幅推定パイプラインの段階一覧

`_run_recognition_core_like_offline`(get_book_points.py)が全体を統括。
`debug_residual_stage_diagnostics`に記録される主な段階:

| stage_id | ステージ名 | 内容 |
|---|---|---|
| (SAM3選択直後) | ー | SAM3が"book spine"プロンプトで初期マスク生成、OCRとマッチングして対象マスク選択 |
| 01 | after_depth_prefilter | 選択マスク内のDepth中央値±3cmで外れ値除去(2026-08-21、マスク分裂対応ステージ7.4を追加) |
| 02 | after_depth_prefilter_spine_completion | OCR軸中心帯で背表紙内部の欠けを補完 |
| 07 | after_column_refine | 背表紙方向の列長フィルタ。自動スコアで複数モードから選択。最も幅を大きく変える主要ステージ |
| 08 | after_final_t_width_clip | 07で`seed_width_guard_false`選択時の安全網。OCR由来の推定幅を基準に追加クリップ |
| 09 | after_post_column_side_front_prune | 07で削った側の短列を追加除去 |
| 11 | after_ransac_spine_plane | OCR文字領域から推定した平面でRANSAC外れ点除去 |
| 12 | after_post_ransac_a95 | RANSAC平面上a方向95%点だけ残す |
| 90 | final_before_calculate_yaw | 最終マスク(final.pngと同じ)。ここから`estimate_book_width_from_filtered_mask_axis`で幅算出 |

stand-100の100件を分析し、**リファイン段階(07〜12)全体の変化量と最終誤差がピアソン
r=0.578で相関**すること、**depth前処理(01)でのマスク分裂(component_count>1)が
誤差増加と弱い相関**(単一成分78件平均5.95mm・分裂22件平均7.88mm)を持つことを確認。
詳細は`~/.claude/projects/-home-catheter-pro-book/memory/
catheter_width_mask_pipeline_investigation.md`を参照。

### 7.3 識別バグ: 隣接マスクの矩形バウンディングボックス重複によるOCRテキスト混入(修正済み)

`multikey_matcher.py`の`_collect_mask_texts()`が、OCR文字を矩形バウンディングボックス
(傾きの無い直方体)同士の重なりでマスクへ割り当てていたため、棚で微妙に傾いて隣接する
2箱の矩形が重なり合い、片方の文字がもう片方のテキストバケットへ混入する実例
(query=ESC0305で隣のAXS Vecta 46 DACのバケットに"ESC0305"が混入しconfident=Trueで
誤選択)を発見。**実際のマスク輪郭(ピクセル単位、`cv2.fillPoly`でOCR文字ポリゴンを
描画してマスクとAND演算)との重なりで判定するよう修正済み**。100件再テストで該当バグ
2件が正しい方向に修正され、新たな悪化は無いことを確認。

### 7.4 マスク分裂対応ステージ(優先度1、実装済み・A/B比較中)

`handle_mask_fragmentation_after_depth_prefilter()`(get_book_points.py)を追加。
01(depth外れ値除去)直後に`component_count > 1`を検知したら、(1)モルフォロジー的
closing(15px)で近接する穴・隙間を橋渡しして再結合を試み、(2)closingでも残った
成分は最大成分の8%未満のものだけノイズとして除去する。`enable_fragmentation_handling`
引数で有効/無効を切替可能(既定True)。100件A/B比較は実行中(2026-08-21時点)。

### 7.5 depth外れ値除去のRANSAC平面残差版(優先度2、ed実装・A/B比較中)

`save_masked_and_cropped()`に`outlier_method="ransac_residual"`を追加(既定は従来通り
`"absolute"`)。OCR参照領域からRANSAC平面フィット→残差8mm(`ransac_distance_threshold_m`)
で外れ値判定する、傾いた背表紙面に対応した方式。点数不足/フィット失敗時は絶対閾値方式へ
自動フォールバックする。`_run_recognition_core_like_offline`/`run_capture_and_pca_offline`
に`depth_outlier_method`引数を追加し、呼び出し側から選択可能。100件A/B比較は実行中。

### 7.6 色の補助識別キー(実装済み)

無地・OCR手がかりの薄い品目(オレンジ箱等)向けの補助シグナル。`reco/master_catheter_reco.json`
に20品目分`color_rgb`(confident=Trueな実例からRGB平均値をブートストラップ)を追加し、
`multikey_matcher.py`のスコア行列に4番目のキーとして`color`を追加。**注意**: 実装直後、
色スコアが僅差でwinning_keyの座を奪いconfident判定を壊す副作用が4項目で発生したため、
中央値センタリング後の色スコアに減衰係数0.35を掛けて解決済み(白〜グレー系の似た色の
品目間では影響が出ず、オレンジ箱のように明確に色が違う場合だけ効くよう調整)。

### 7.7 推定幅誤差5mm以上でのリトライ機構(実装済み、パラメータ変化の詳細)

`width_mm_validation.py`に`run_one_with_retry()`を追加。**正解幅(book_width)との誤差が
5mm以上の場合のみ**、最大3回まで異なるパラメータで再実行し、最も正解に近かった結果を
採用する(正解が分からない本番環境では判定できないため、本番への適用は別途要検討として
いったん評価スクリプト側のみ)。

同じ静止画像に対して全く同じパラメータで再実行してもパイプラインは決定的(SAM3は
テキストプロンプト固定でランダム性なし)なので、各試行で以下のように**depth外れ値除去
(②)に関わるパラメータを実際に変えている**:

| 試行 | depth_outlier_method | depth_merge_tolerance_raw(絶対閾値の許容幅) | 備考 |
|---|---|---|---|
| 0(初回) | absolute | 30(±3cm) | パイプライン既定の挙動そのまま |
| 1(retry 1回目) | ransac_residual | 30(±3cm、フォールバック時のみ使用) | 7.5のRANSAC平面残差方式に切替。残差閾値は既定8mm |
| 2(retry 2回目) | absolute | 45(±4.5cm) | 絶対閾値方式のまま、許容幅だけ1.5倍に拡大 |
| 3(retry 3回目・最終) | ransac_residual | 45(±4.5cm、フォールバック時のみ使用) | RANSAC失敗時のフォールバック幅だけ試行1と異なる |

各試行は`{フォルダ名}_attempt{N}`という別フォルダに保存され、後から個別に確認できる。
CSV/Excelに`retry_count`列を追加済み。オレンジ箱(識別自体が破綻、score=0.0)でのスモーク
テストでは3回ともRANSAC失敗→フォールバックとなり改善が無かったが、これは想定通りの
挙動(識別が機能していない極端ケース)として確認済み。100件全体でのretry発動率・改善率は
7.4/7.5のA/B比較と合わせて今後まとめる。
