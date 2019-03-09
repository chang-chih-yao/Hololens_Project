# Server

- TSN_Training：我們必須先把 Action Recognintion Network 訓練好，並產生 .pth 的檔案給 pytorch model 用
- Run_Hololens：裡面已經包含 OpenPose跟TSN model，可以直接連上 Hololens

***

## TSN_Training

訓練時，到資料夾內執行：  
```python main_two_class.py ucf101 RGB D:\Dataset\Action\my_dataset\my_train.txt D:\Dataset\Action\my_dataset\my_test.txt --arch resnet34 --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 60 -b 16 -j 0 --dropout 0.8 --gpus 0```
*my_train.txt 跟 my_test.txt 怎麼產生的可以到 my_dataset 資料夾裡面看一下*
