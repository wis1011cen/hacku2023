# branchの構成
- __main__:          Pose landmark detectionだけ. 安定
- __with_gesture__:  Gesture recognitionも行う. 改良が必要．
- __legacy__:        従来の方法で動作が遅い．

# mediapipeをインストール
非同期でフレームの処理を行うことで，高FPSを実現できるように改良．
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

