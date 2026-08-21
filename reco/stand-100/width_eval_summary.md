# 実mm幅計測 検証結果 (2026-08-21)

`get_book_points.py`本番の3D計測パイプライン(SAM3サービス + OCR + PCA/軸フィルタ)を、
`depth_shots/`(RealSense実撮影、RGB+depth)に対して実際に呼び出し、推定した実mm幅を
マスタJSON(`reco/master_catheter_reco.json`)の`book_width`(mm、正解値)と比較した。

生成スクリプト: [`reco/scripts/width_mm_validation.py`](../scripts/width_mm_validation.py)
実行: `.pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset stand-100`
結果生データ: [`width_eval_result.csv`](width_eval_result.csv)

目視レビュー用Excel・画像は [`width_eval_report/catheter_width_report.xlsx`](width_eval_report/catheter_width_report.xlsx) /
[`width_eval_report/images/`](width_eval_report/images/)（`reco/scripts/build_width_eval_report.py`で生成、
このデータセット専用。diagonal-40とはフォルダを分けてある）。

## 結果(5棚画像 × 20品目 = 100件、全件成功・エラー0件)

**誤差2mm以内が実機把持成功の目安**（ユーザーの経験則、2026-08-21確認）なので、
まずこの数字を見ること。MAEや「10mm未満」だけでは精度を過大評価しかねない。

| 指標 | 値 |
|---|---|
| **2mm以内(把持成功の目安)** | **9/100 (9%)** |
| 5mm未満 | 39/100 (39%) |
| 10mm未満 | 74/100 (74%) |
| MAE | 9.06 mm |
| 中央値 | 5.72 mm |

→ 2mm基準で見ると、現状は9%しか実用ラインに届いておらず、実機把持にはまだ耐えない可能性が高い。

比較用に `reco/diagonal-40/`(斜め置き、40件)も同じ手法で測定した結果:
MAE 9.96mm・中央値6.28mm・10mm未満30/40(75%)と、**ほぼ同水準**。
「斜め姿勢だから精度が悪い」という仮説は誤りで、直立配置でも同程度の誤差が出ている。
誤差の実体は下記「別セッションとの共同デバッグ」参照(識別失敗と幅推定アルゴリズムの
限界、2つの異なる原因が混在していることが判明した)。

## 別セッションとの共同デバッグで判明したこと(2026-08-21、pro-book-92/pro-book-cdと共同調査)

MAE 9〜10mmという誤差の原因切り分けを他セッションと共同で行った。分かったこと:

1. **`sam_pts_side`はSAM3経路では死んだ引数。** `_get_sam_runner_compat`のsam3分岐は
   最終的に`Sam3BatchInfer()`を**引数無しで**呼んでおり、`sam_pts_side`
   (元はsam_vit_h/ONNX時代のシグネチャの名残)は一切使われない。実際に(32,8)→(64,16)へ
   変えて15件再実行したが、結果はbaselineと**ビット単位で完全一致**した。SAM3はそもそも
   点グリッドでプロンプトしていない。
2. **SAM3は固定テキストプロンプト"book spine"1本で画像全体を分割している。**
   `client.py`の`Sam3BatchInfer.__init__(..., prompt="book spine")`が既定値。
   `SAM3_TEXT_PROMPT`環境変数(`settings.py`)は存在するが、起動時のウォームアップ呼び出し
   にしか使われず、実際の`/infer`リクエストはクライアント側が常に送る"book spine"を使う
   ため**実質無効**。
3. **ファインチューンデータはbook_spine専用の可能性が高い。** `sam3_source/sam3/train/
   configs/book_spine/book_spine_finetune.yaml`の`dataset_root`は
   `book_spine_sam3_dataset`(元テンプレートはRoboflowの書籍データセット)。
   `sam3_runtime/docs/*.md`全体をgrepしても"catheter"の言及は0件。ドメインミスマッチの
   状況証拠はある(確定ではない)。
4. **しかし、プロンプトを"product box"に変えても、識別が正しいケースの誤差は改善しなかった。**
   `_SAM_RUNNER_CACHE`にカスタムprompt付きrunnerを注入して実験(注入タイミング・raw
   SAM3出力の差異を実データで検証済み、キャッシュ起因の見せかけではない)。高confidence
   (score=100等)かつ誤差が大きい7件で試したところ、7件中2件は誤差が完全一致、残りも
   ±1〜2mm程度でランダムに散らばり、系統的な改善は見られなかった。**識別が正しいケースの
   5〜10mm程度の残存誤差は、プロンプト/ドメインミスマッチではなく、幅推定アルゴリズム
   自体(PCA/軸フィルタ、depth品質)の限界と考えられる。**
5. **最悪ケース(M00345100950、オレンジ無地箱、真値44.2mm、予測10〜15mm)は、プロンプト云々
   以前に識別が毎回失敗していた。** final.pngを目視したところ、確認した3棚(shot1,2,3)
   すべて・baseline/product boxプロンプト両方で、**一度もオレンジ箱自体が選ばれておらず**、
   隣接する別の箱(AXS Vecta 46 DAC、GuidePost、Neuroform Atlasなど、棚ごとに毎回違う)を
   誤って測っていた。テキストがほぼ無い無地箱のためOCRが手がかりを拾えないのが原因と
   見られる。
6. **`confident`フラグ(score≥70かつmargin≥20)は物理的な対応の正しさを保証しない。**
   上記5のshot3では`selected_score=100.0, margin=40.0, confident=True`という最高水準の
   スコアだったにもかかわらず、実際には全く別の箱(Neuroform Atlas)を選んでいた。
   テキスト類似度だけのスコアで、マスクとの空間的対応関係までは見ていないため。

**結論**: MAE 9〜10mmは単一の原因ではなく、少なくとも2種類の異なる問題の合成値。
(a)識別が完全に失敗するケース(テキストが薄い無地箱等)→誤差数十mm級、
(b)識別は正しいが幅推定自体に5〜10mm程度のブレがあるケース。(a)は識別ロジック
(OCR依存度を下げる、形状/色ベースの補助手がかりを足す等)、(b)は幅推定アルゴリズム
(`estimate_book_width_from_filtered_mask_axis`のaxis検出やdepthノイズ)を、それぞれ
別々に改善する必要がある。

## この検証のために修正した環境の問題(2026-08-21、いずれも今回発見)

1. **SAM3サービス**: `sam3_runtime/vendor/owner_repo/config/local_paths.json`が旧マシンパス
   (`/home/book/pro_book_SAM3/...`)を指したままで、サービス起動時のモデルロードが失敗して
   いた。現在のパスに書き換えて解消。サービスは`sam3_runtime/scripts/start_service.sh`で
   起動(GPU常駐、`http://127.0.0.1:8765`)。
2. **OCR用venv(.paddle_ocr)のmatplotlib/numpy非互換**: システムのmatplotlib(numpy1系向け
   ビルド)とvenvのnumpy2.2.6が衝突しクラッシュ。venv内に`matplotlib>=3.8`を入れ直して解消。
3. **PaddleOCRモデル本体が未取得**: `PP-OCRv5_server_det`/`_rec`(公式モデル、計不明500MB弱)が
   このマシンのどこにも存在しなかった。公式ソースから再取得し、
   `detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py`内のハードコードパス
   (`/home/book/.paddlex/...`)を`Path.home()`基準に修正。可視化フォント(simfang.ttf)も取得。
4. **`get_book_points.py`の実バグ(未修正のまま残っていたもの)**: デバッグ画像保存用の
   `save_spine_column_length_debug()`が`n_t_bins`という未定義変数を参照しており、呼ばれる
   たびに必ずNameErrorになっていた。幅計測が有効化される(`needs_shape_refine=True`)ケースで
   毎回発生するため、実質いつも例外を吐いていたと考えられる。デバッグ出力専用の処理なので、
   例外を吸収して認識処理自体は継続するようラップして修正(`_save_spine_column_length_debug_impl`
   に実装を退避し、新しい`save_spine_column_length_debug`がtry/exceptで包む形)。
   **この修正はオンライン(実機)の`run_capture_and_pca()`経路にも影響する共通コードなので、
   実機側の挙動にも変化がある可能性がある**(従来は握りつぶされていた例外が今後は出ない
   ようになるだけで、デバッグ画像が保存されるようになる分にはプラス。要ウォッチ)。

## 注意点

- depth_shots画像はOCR都合で撮影時のセンサー向きから180度回転させて保存されている
  (`reco/README.md`参照)。`width_mm_validation.py`はRGB・depthを両方`np.rot90(arr, 2)`で
  撮影時の向きに戻してから処理している。`camera_params.json`の値は`get_book_points.py`の
  オフライン版が使う固定値と完全一致することを確認済みなので、そのまま使っている。
- 識別(query→マスク選択)は`multikey_matcher.py`の`match_text_to_mask_main`を使用。
  一部のqueryで確信度が低い旨の警告が出ているが、識別結果自体はどのqueryでも(誤りも含め)
  何らかのマスクが選ばれるため、今回の幅計測自体は全件完走している。識別精度そのものの
  評価はしていない(既存の80件評価(px比較、`outputs/`)を参照)。
