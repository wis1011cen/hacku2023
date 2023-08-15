# branchの構成
- main: poseだけ
- legacy: ssh接続を行う

# mediapipeをインストール
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#live-stream
`pip install mediapipe==0.10.3`

#### インストールに失敗する場合：
https://pypi.org/project/mediapipe/#files から自分でビルドする.

# run.pyで実行
- src/utils.pyの`def ir_operation()`中にある`irrp.ir_lightning(operation_name)`をコメントアウト

- ##### Logicool Webカメラで実行する場合
   `SCALE = 1`に変更．
- ##### MacBook Pro内蔵カメラで実行する場合
   `SCALE = 2`に変更．

