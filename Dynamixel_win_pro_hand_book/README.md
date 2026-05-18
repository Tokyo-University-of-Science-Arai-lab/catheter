# Dynamixel_win_pro_hand_book
書籍出納用ハンド 電動化（プロトタイプ3）Dynamixel サーボモータの windows PC からの制御スクリプト

python 仮想環境を作成，仮想環境 activate
```
git clone git@github.com:RyoseiKanekoTUS/Dynamixel_win_pro_hand_book.git
```
```
cd Dynamixel_win_pro_hand_book/DynamixelSDK/python
```
```
python setup.py install
```
```
cd ../../
```
各スクリプト，スペーサは回転0状態，最後退状態から開始（モータ位置の都合）
RetrievalSequence.py : 取り出し動作時，グリッパ開口幅を左右カーソルキーで指定，その後 g キーを押下でグリッパが把持動作

StorageSequence.py : 収納対象書籍を把持している状態を想定，まずスペーサが前進，その後スペーサ回転が起動，左右カーソルキーで回転，r キー押下でスペーサ回転が0位置に戻る&スペーサが後退，スペーサ後退完了後，u キー押下でグリッパ開動作
