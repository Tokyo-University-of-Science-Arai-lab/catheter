# master_catheter_0911_2.json 削除・追加クエリの記録

記録日: 2026-09-25
元ファイル: `master_catheter_0911.json`（33件、変更なし・復元元として残してある）
編集対象: `master_catheter_0911_2.json`（2026-09-25 20:19保存版、25件）

## 0911との差分まとめ（増えたもの・消えたもの）

`master_catheter_0911.json`（33件）→ `master_catheter_0911_2.json`（25件）。全項目が一致するかで1行ずつ照合した結果（ディスク上の実ファイルから算出）。

### 消えたもの（10件）

| book_name | display_name | 使用期限 | LOT |
|---|---|---|---|
| KMDA044178 | Guidepost | 2029-03-25 | 不明 |
| APB-1.5-2-HX-ES | Axium Prime | 2028-04-09 | 230926809 |
| SS2-40-070H | SHOURYU2 | 2029-03-31 | KR046055 |
| 1681189 | Excelsior SL-10 | 2028-02-18 | 26124294 |
| SSTD215STR | Synchro SELECT | 2029-02-24 | 不明 |
| 373-040830 | i-ED COIL | 2027-10-31 | KR104430 |
| 612104 | Target XL | 2030-06-07 | 不明 |
| MC1715000 | pNOVUS | 2028-11-27 | MCR00341 |
| FG15160-0615-1S | Phenom | 2028-12-10 | 232896730 |
| OPTI0408CSS10 | OPTIMA | 2030-08-07 | F260200861 |

### 増えたもの（2件）

| book_name | display_name | 使用期限 | LOT | 備考 |
|---|---|---|---|---|
| OPTI0306CSS10 | OPTIMA | 2030-04-15 | F230400435 | 新規品目 |
| OPTI0153CSS10 | OPTIMA | 2030-12-04 | F24120061 | 新規品目 |

計算: 33 − 10（消えた）＋ 2（増えた）＝ 25件。

※20:16保存版では `10CSW01515` が全項目同一で2行になっていたが、20:19の再保存で1行に戻った（0911と同じ）。`validate_master.py` の結果も「問題は見つかりませんでした」。

以下は、経緯の記録（ユーザー申告の削除リストと、その後の実際の差分の照合）。

## 目的

`reco/0911` の検証用に、マスタ（リスト）から一部のクエリを削除したマスタを作る。
（「画像には映っているがリストに無い品目」がある状況を作る目的と思われるが、ユーザーからの明示は無いので未確認。関連: `multikey_matcher.py` の `MULTIKEY_REJECT_LOW_SCORE`）

## 削除したクエリ（11件、ユーザー申告 2026-09-25）

※1〜10件目を申告後、同日に11件目（FG15160-0615-1S）を追加で削除するとの申告あり。

| # | book_name | display_name | 使用期限 | LOT |
|---|---|---|---|---|
| 1 | KMDA044178 | Guidepost | 2029-03-25 | 不明 |
| 2 | APB-1.5-2-HX-ES | Axium Prime | 2028-04-09 | 230926809 |
| 3 | SS2-40-070H | SHOURYU2 | 2029-03-31 | KR046055 |
| 4 | 1681189 | Excelsior SL-10 | 2028-02-18 | 26124294 |
| 5 | 10CSW01010 | Wallaby Avenir Coil System | 2031-05-09 | WCS12609 |
| 6 | OPTI0204CSS10 | OPTIMA | 2030-11-11 | F251100383 |
| 7 | SSTD215STR | Synchro SELECT | 2029-02-24 | 不明 |
| 8 | 373-040830 | i-ED COIL | 2027-10-31 | KR104430 |
| 9 | 612104 | Target XL | 2030-06-07 | 不明 |
| 10 | MC1715000 | pNOVUS | 2028-11-27 | MCR00341 |
| 11 | FG15160-0615-1S | Phenom | 2028-12-10 | 232896730 |

削除後の件数: 33 − 11 = **22件**（の想定）

## 削除した11件の全文（復元用）

```json
{
    "book_name": "KMDA044178",
    "display_name": "Guidepost",
    "LOT": "不明",
    "expiration date": "2029-03-25",
    "ISBN_number": "不明",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "26",
    "SPEC_1": "3.2/3.4Fr.",
    "SPEC_2": "120cm"
},
{
    "book_name": "APB-1.5-2-HX-ES",
    "display_name": "Axium Prime",
    "LOT": "230926809",
    "expiration date": "2028-04-09",
    "ISBN_number": "9784297118846",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "17.1",
    "SPEC_1": "1.5 mm",
    "SPEC_2": "2 cm"
},
{
    "book_name": "SS2-40-070H",
    "display_name": "SHOURYU2",
    "LOT": "KR046055",
    "expiration date": "2029-03-31",
    "ISBN_number": "不明",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "17.1",
    "SPEC_1": "4.0 mm",
    "SPEC_2": "7 mm"
},
{
    "book_name": "1681189",
    "display_name": "Excelsior SL-10",
    "LOT": "26124294",
    "expiration date": "2028-02-18",
    "ISBN_number": "不明",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "17.0",
    "SPEC_1": "150 cm",
    "SPEC_2": "6 cm"
},
{
    "book_name": "10CSW01010",
    "display_name": "Wallaby Avenir Coil System",
    "LOT": "WCS12609",
    "expiration date": "2031-05-09",
    "ISBN_number": "不明",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "15.2",
    "SPEC_1": "1 mm",
    "SPEC_2": "10 cm"
},
{
    "book_name": "OPTI0204CSS10",
    "display_name": "OPTIMA",
    "LOT": "F251100383",
    "expiration date": "2030-11-11",
    "ISBN_number": "9784627846913",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "20",
    "SPEC_1": "2 mm",
    "SPEC_2": "4 cm"
},
{
    "book_name": "SSTD215STR",
    "display_name": "Synchro SELECT",
    "LOT": "不明",
    "expiration date": "2029-02-24",
    "ISBN_number": "9784254209457",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "18.0",
    "SPEC_1": "0.014 in",
    "SPEC_2": "215 cm"
},
{
    "book_name": "373-040830",
    "display_name": "i-ED COIL",
    "LOT": "KR104430",
    "expiration date": "2027-10-31",
    "ISBN_number": "9784254209457",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "15",
    "SPEC_1": "4-8 mm",
    "SPEC_2": "30cm"
},
{
    "book_name": "612104",
    "display_name": "Target XL",
    "LOT": "不明",
    "expiration date": "2030-06-07",
    "ISBN_number": "9784627846913",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "17",
    "SPEC_1": "10 mm",
    "SPEC_2": "40 cm"
},
{
    "book_name": "MC1715000",
    "display_name": "pNOVUS",
    "LOT": "MCR00341",
    "expiration date": "2028-11-27",
    "ISBN_number": "9784798068299",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "21.0",
    "SPEC_1": "17",
    "SPEC_2": "150 cm"
},
{
    "book_name": "FG15160-0615-1S",
    "display_name": "Phenom",
    "LOT": "232896730",
    "expiration date": "2028-12-10",
    "ISBN_number": "9784781908137",
    "bookshelf_ID": "1-1-1-1",
    "book_width": "17.0",
    "SPEC_1": "27",
    "SPEC_2": "160 cm"
}
```

## 保存後の実際の差分（経緯。最終状態は上の「0911との差分まとめ」を参照）

`master_catheter_0911_2.json` は保存され、20:16版は26件、20:19の再保存で**25件**になった（当初の確認時は未保存で33件のままだった）。
`master_catheter_0911.json`（33件）との差分をエントリ単位で照合した結果:

**33 − 10（削除）＋ 2（追加）＝ 25件**（20:16版は＋1（10CSW01515の重複）で26件だった）

### 削除されている（10件）

KMDA044178 / APB-1.5-2-HX-ES / SS2-40-070H / 1681189 / SSTD215STR / 373-040830 / 612104 / MC1715000 / FG15160-0615-1S / **OPTI0408CSS10**

### 上の申告リストとの食い違い（要確認）

- **申告では削除としていたが、実際には残っている（2件）**: `10CSW01010`（Wallaby, 2031-05-09）、`OPTI0204CSS10`（OPTIMA, 2030-11-11）
- **申告になかったが、実際には削除されている（1件）**: `OPTI0408CSS10`（OPTIMA, 2030-08-07）

意図した結果かどうかは未確認。上の「削除した11件」の表・全文は申告時点の記録として残してある。

### 追加されたエントリ（2件、0911には無かったもの）

```json
{
  "book_name": "OPTI0306CSS10",
  "display_name": "OPTIMA",
  "LOT": "F230400435",
  "expiration date": "2030-04-15",
  "ISBN_number": "9784627846913",
  "bookshelf_ID": "1-1-1-1",
  "book_width": "20",
  "SPEC_1": "3 mm",
  "SPEC_2": "6 cm"
},
{
  "book_name": "OPTI0153CSS10",
  "display_name": "OPTIMA",
  "LOT": "F24120061",
  "expiration date": "2030-12-04",
  "ISBN_number": "9784627846913",
  "bookshelf_ID": "1-1-1-1",
  "book_width": "20",
  "SPEC_1": "1.5 mm",
  "SPEC_2": "3 cm"
}
```

いずれもROOB-1（`master_catheter_2.json`）に載っているOPTIMA系の品目。

### 重複したエントリ（解消済み）

20:16保存版では `10CSW01515`（LOT WCS12612）が全項目同一で2行あったが、ユーザーの指摘（1件のはず）を受けて確認し、20:19の再保存で1行になっていた。現在は重複なし。

### 保存後の全25件（20:19版）

10CSW02020, 10CSW01010, 10CSW01515, AIN-CKI-200R, MONORAIL, ESC0407, MAT-110-110, XT275081, EZAS3021, FC-3-6-3D, 393-020310, SPD-040-320, INC-11814-146, AIN-CHI-10-200R, OPTI0306CSS10, OPTI0204CSS10, OPTI0153CSS10, OPTI0152CSS10, ESC0305, MIV-2N316, C1775ST, FG13160-0615-1S, DAC6F115, SD-40040, APB-2-3-HX-ES
