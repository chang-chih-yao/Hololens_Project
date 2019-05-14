# tf-pose-estimation

[Origin-github](https://github.com/ildoonet/tf-pose-estimation)

'Openpose', human pose estimation algorithm, have been implemented using Tensorflow. It also provides several variants that have some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**

**You can even run this on your macbook with a descent FPS!**

## Models & Performances

### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files. You can deploy this graph on your mobile or other platforms.

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)
- mobilenet_v2_large (trained in 432x368)
- mobilenet_v2_small (trained in 432x368)

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```

## Install

### Dependencies

You need dependencies below.

- python 3.6
- tensorflow 1.4.1+ (1.9.0)
- opencv3, protobuf, python3-tk
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Package Install

Alternatively, you can install this repo as a shared package using pip.
1. unzip tf-pose-estimation-master.zip
2. rename "tf-pose-estimation-master" to "tf-pose-estimation"
3. put folder "tf-pose-estimation" in "Hololens_Project/Core/"
4. check requirements.txt, you can install by yourself
5. overwrite "Hololens_Project/Core/tf-pose-estimation/setup.py"
6. overwrite "Hololens_Project/Core/tf-pose-estimation/tf_pose/estimator.py"

```bash
$ cd tf-pose-estimation
$ python setup.py install
```

If setup.py cause error, delete line 15.  
```subprocess.check_output(["bash", "models/graph/cmu/download.sh"], cwd=cwd)```  
But make sure you have already downloaded "graph_opt.pb" in "models/graph/cmu/"

## Demo

### Test Inference

You can test the inference feature with a single image.

```
$ python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```

The image flag MUST be relative to the src folder with no "~", i.e:
```
--image ../../Desktop
```

### Realtime Webcam

```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
```

## Python Usage

This pose estimator provides simple python classes that you can use in your applications.

See [run.py](run.py) or [run_webcam.py](run_webcam.py) as references.

```python
e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
humans = e.inference(image)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
```

If you installed it as a package,

```python
import tf_pose
coco_style = tf_pose.infer(image_path)
```
