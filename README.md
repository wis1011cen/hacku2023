# mediapipeをインストール
`pip install mediapipe==0.10.3`
#### インストールに失敗する場合：
https://pypi.org/project/mediapipe/#files から自分でビルドする.

# run.py
- src/utils.pyの`def ir_operation()`中にある`irrp.ir_lightning(operation_name)`をコメントアウト．

- Webカメラで実行する場合: `SCALE = 1`に変更．
- 内蔵カメラで実行する場合: `SCALE = 2`に変更．

