# <div align="center">Simple Object Tracking</div>

<div align="center">
<p>Object Tracking with YOLOv5, CLIP and DeepSORT</p>
<p>
<img src="gifs/test_out.gif" width="270"/> <img src="gifs/newyork_out.gif" width="270"/> <img src="gifs/cars_out.gif" width="270"/> 
</p>
</div>

## Introduction

This is a simple mulit-object tracking implementation with zero-shot or self-supervised feature extractors. The implementation is based from Roboflow [zero-shot-object-tracking](https://github.com/roboflow-ai/zero-shot-object-tracking); which incorporates CLIP as a feature extractor in DeepSORT. 

[CLIP](https://openai.com/blog/clip/) is a zero-shot classification model; which is pretrained under vision-langauge supervision with a lot of data. CLIP's zero-shot performance is on par with supervised ResNet models.

The benefit of this approach is that it can track a lof of classes out-of-the-box without needing to re-train the feature extractor (re-identification model) for a specific class in DeepSORT. Its performance may not be as good as traditional re-identification model; which is trained on tracking/re-identification datasets.


## Requirements

* torch >= 1.8.1
* torchvision >= 0.9.1

Other requirements can be installed with `pip install -r requirements.txt`.

Clone the repository recursively:

```bash
$ git clone --recursive https://github.com/sithu31296/simple-object-tracking.git
```

Then download a YOLO model's weight from [YOLOv5](https://github.com/ultralytics/yolov5) and place it in `checkpoints`.

## Tracking

Track all classes:

```bash
## webcam
$ python track.py --source 0 --yolo-model checkpoints/yolov5s.pt --clip-model RN50

## video
$ python track.py --source VIDEO_PATH --yolo-model checkpoints/yolov5s.pt --clip-model RN50
```

Track only specified classes:

```bash
## track only person class
$ python track.py --source 0 --yolo-model checkpoints/yolov5s.pt --clip-model RN50 --filter-class 0

## track person and car classes
$ python track.py --source 0 --yolo-model checkpoints/yolov5s.pt --clip-model RN50 --filter-class 0 2
```

Available CLIP models: `RN50`, `RN101`, `RN50x4`, `ViT-B/32`, `ViT-B/16`.

Check [here](tracking/utils.py#L14) to get COCO class index for your class.

## Evaluate on MOT16

* Download MOT16 dataset from [here](https://motchallenge.net/data/MOT16.zip) and unzip it.
* Download mot-challenge ground-truth [data](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) for evaluating with TrackEval. Then, unzip it under the project directory.
* Save the tracking results of MOT16 with the following command:

```bash
$ python eval_mot.py --root MOT16_ROOT_DIR --yolo-model checkpoints/yolov5m.pt --clip-model RN50
```

* Evaluate with TrackEval:

```bash
$ python TrackEval/scripts/run_mot_challenge.py
    --BENCHMARK MOT16
    --GT_FOLDER PROJECT_ROOT/data/gt/mot_challenge/
    --TRACKERS_FOLDER PROJECT_ROOT/data/trackers/mot_challenge/
    --TRACKERS_TO_EVAL mot_det
    --SPLIT_TO_EVAL train
    --USE_PARALLEL True
    --NUM_PARALLEL_CORES 4
    --PRINT_ONLY_COMBINED True
```

> Notes: `FOLDER` parameters in `run_mot_challenge.py` must be an absolute path.

For tracking persons, instead of using a COCO-pretrained model, using a model trained on multi-person dataset will get better accuracy. You can download a YOLOv5m model trained on [CrowdHuman](https://www.crowdhuman.org/) dataset from [here](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing). The weights are from [deepakcrk/yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman). It has 2 classes: 'person' and 'head'. So, you can use this model for both person and head tracking.

## Results

**MOT16 Evaluation Results**

Detector | Feature Extractor | MOTA↑ | HOTA↑ | IDF1↑ | IDsw↓ | MT↑ | ML↓ | FP↓ | FN↓
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
YOLOv5m<br>(COCO) | CLIP<br>(ViT-B/32) | 35.29 | 35.03 | 38.33 | **519** | 117 | 191 | **7061** | 63865
YOLOv5m<br>(CrowdHuman) | CLIP<br>(ViT-B/32) | **53.21** | 42.94 | 51.02 | 941 | **199** | 91 | 14239 | **36475**
YOLOv5m<br>(CrowdHuman) | CLIP<br>(RN50) | 53.07 | **43.14** | **51.97** | 946 | 198 | **88** | 14331 | 36537


**FPS Results**

Detector | Feature Extractor | GPU | Precision | FPS
--- | --- | --- | --- | ---
YOLOv5s | CLIP (RN50) | GTX-1660ti | FP32 | 34
YOLOv5m | CLIP (RN50) | GTX-1660ti | FP32 | 27
YOLOv5s | CLIP (ViT-B/32) | GTX-1660ti | FP32 | 24
YOLOv5m | CLIP (ViT-B/32) | GTX-1660ti | FP32 | 20

> Note: Image size used is (480, 640) with one detection per frame.

## References

* https://github.com/roboflow-ai/zero-shot-object-tracking
* https://github.com/ultralytics/yolov5
* https://github.com/nwojke/deep_sort
* https://github.com/openai/CLIP
* https://github.com/JonathonLuiten/TrackEval
* https://github.com/deepakcrk/yolov5-crowdhuman
* https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch