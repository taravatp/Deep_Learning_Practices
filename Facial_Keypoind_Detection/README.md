# Facial_Keypoint_Detection
The objective of this task was to predict several continuous outputs, which are the locations of keypoints present on an image of a face. this project can be used as a building block in several applications:
* tracking faces in images and videos.
* analysing facial expressions.
* detecting dysmorphic facial signs for medical diagnosis

# Methodology
1. Downloaded the dataset from https://github.com/udacity/P1_Facial_Keypoints.git
2. Preprocessed the dataset.
3. Implemented Resnet architecture from scratch
4. Trained various neural networks and exprimented their results. these neural networks include:
    1. training Resnet50 from scratch
    2. using a pretrained Resnet50 and changing its last layer to fit our task.
    3. using a pretrained VGG16 architecture and changing its last few layers.
5. Used Mean Absolute Error as the loss function, since the outputs are continuous values.
6. Used Adam optimizer.
7. Compared the results of various architectures based on RMSE accuracy metric.

# Results

Model                          | RMSE score after 6 epochs | MAE loss after 6 epochs
--------------------------     | ------------------------  |------------------------
Trained resnet50 from scratch  | Content Cell              | Content Cell
pretrained resnet50            | 0.386                     | 0.0591
pretrained VGG16               | 0.237                     | 0.0508

