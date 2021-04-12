# Ready, Steady, Go AI: A Practical Tutorial on Explainable Artificial Intelligence and Its Applications in Plant Digital Phenomics
----
## Step 1 - Image Dataset Selection
----
We perform our analysis on a subset of the publicly available [PlantVillage](https://data.mendeley.com/datasets/tywbtsjrjv/1) dataset consisting of 18,160 RGB images split over 10 classes of healthy and diseased tomato leaves.

<img src="http://faridnakhle.com/pv/githubimages/Step1.png" width="600"/>

All of our data and trained models are available on Mendeley data at http://dx.doi.org/10.17632/4g7k9wptyd.1

When runing our code in Step 2, 3, and 4, required data and models will be downloaded to the project automatically.

We provide:
1. A randomly split version of the dataset (80% training, 10% validation, and 10% 1. testing).
2. 1000 annotated images with bounding boxes to train YOLOv3 on cropping leaf images.
3. 150 annotated images with segmentation masks to train SEGNET on segmenting leaf images.
4. Cropped version of the split dataset.
5. Segmented version of the cropped and split dataset.
6. Balanced version of the segmented, cropped, and split dataset, balanced to around 1500 images per class.
7. Trained YOLO, SegNet, DCGAN, RF, DCNN, and Pretrained Densenet-161 DCNN models

For more details, check our paper: "Ready, Steady, Go AI: A Practical Tutorial on Explainable Artificial Intelligence and Its Applications in Plant Digital Phenomics"