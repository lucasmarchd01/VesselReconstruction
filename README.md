# Ultrasound Vessel Segmentation and Reconstruction Analysis

## U-Net Training and Performance Evaluation

The unet was trained monitoring the validation accuracy, since this achieved the best
results from the previous assignment. Additionally, a learning rate of 1e-3 was used with
20 epochs and early stopping. The images were resized to (128, 128) to increase the
speed of training of the network.
The following images are the training metrics and loss curves for the unet that was
trained on the segmentation task. We can see that the model has sufficient training
curves, however, more improvement could be needed for the validation loss and
accuracy as they do not converge to a much lower value than the initial few epochs.
The images show example input images from the test set with their predictions from the
unet. From these images we can see that the model is able to localize the region of
interest of the vessel. Unlike the other segmentation methods, there is less noise in the
predictions, leading to a more concentrated area that is used to reconstruct the volume.
The evaluation metrics show that the model can accurately predict the background
class, with a f1 score of 0.96. The object class, however, has much less consistent
performance, with a f1 score of 0.05. Qualitatively, the unet performs adequately but
more work needs to be done to improve the performance of the network on the object
class.

```
Evaluation Metrics: [0.9427585005760193, 0.931469202041626]
Model Hyperparameters:
Learning rate: 0.0001
Loss function: categorical_crossentropy
Metrics: ['accuracy']
Confusion matrix:
[[6610730 73699]
[ 413600 12627]]
Classification report:
precision recall f1-score support
0 0.94 0.99 0.96 6684429
1 0.15 0.03 0.05 426227
accuracy 0.93 7110656
```
 
## Threshold Segmentation Method and Adjustments
### Threshold Segmentation: 
For threshold segmentation, the first method that was used
for choosing the threshold was by calculating the mean pixel value of all images in the
dataset along with the standard deviation. The lower threshold was selected as mean –
std and the upper threshold was the mean + std.
The images above show an example of a ground truth segmentation with the predicted
segmentation from the thresholding method. Evidently, the thresholding was not able to
localize the vessel without predicted most of the entire background as the object class.
As such, we tried manually changing the threshold value to reflect the characteristics of
the image. The vessel part of the image is relatively dark compared to the rest of the
image, so a lower threshold value of 0 and upper threshold value of 2 was selected.
This method of threshold selected performed much better than using the mean pixel
value and was selected for generating the rest of the segmentation data for volume
reconstruction.
### Region Growing: 
For the region growing algorithm, the seed selection was based on
the characteristics of the image. Since the object class has relatively lower pixel value
than the background, the brightest pixel in the image was chosen as the background.
The object class was chosen to be the darkest pixel in the image. The images also
contain a border region that covers the left and right sides of the image. To account for
this, the border region of the image is not looked at for the region growing and seed
selection algorithms. The maximum difference parameter was selected to reflect the
range of values found in the vessel, which was a pixel intensity of roughly around 0 to 7.
A maximum difference of 7 was selected.
The images above show example ground truth segmentations with the predictions from
the region growing algorithm. From these images we can see that the region growing
algorithm was able to select seeds that describe both the background and object class.
Additionally, the region growing was able to roughly segment the entire vessel.
However, the region growing also contains a lot of noise from growing outside the
region of interest. This could be explained by the pixel intensities that surround the
vessel, which are difficult to distinguish from the vessel in many cases.
U-Net: Selection of hyperparameters, training scheme, and example predictions for the
U-Net can be seen in question 1.


## Region Growing Algorithm and Image Characteristics

### Thresholding:
The above image is the ground truth segmentation volume (blue) superimposed with the
thresholding segmentation method (brown). The thresholding approach was able to
segment the same regions as the ground truth, however, the thresholding approach
also produced a volume that is much larger than the ground truth. This is likely due to
the simplicity of the thresholding algorithm, which selects pixels that fall within the range
of values chosen as the lower and upper threshold. As a result, many of the pixels of
the image that are not part of the segmentation still fall within the threshold range.
Image noise can cause pixels to have intensity values that are higher or lower than their
true values, leading to false positives or false negatives in the segmentation. This can
also result in a larger segmented volume than the ground truth. Additionally, In some
cases, it may be difficult to determine the exact boundary of the object of interest,
leading to uncertainty in the segmentation. This can result in a larger segmented
volume than the ground truth if the threshold is set too high, or if the boundary is
estimated incorrectly.
### Region Growing:
The above image is the ground truth segmentation volume (blue) superimposed with the
region growing segmentation method (green). Here we can see that the thresholding
algorithm did a better job as localizing the vessel compared to the thresholding
approach. This is likely due to the more robust method in selecting the starting seeds for
region growing as well as removing the borders of the image. However, the region
growing also produced segmentations with noise in the image because of neighbouring
pixels that have similar intensities to the vessel. The initial seed pixel or region selected
for the region growing algorithm can significantly affect the final segmentation result. If
the seed region is not representative of the actual target region or is in an area with
weak boundaries, the algorithm may merge neighboring regions that should not be
merged, resulting in a larger volume. In many cases, the region growing algorithm
selects neighbouring pixels of the background after segmenting the vessel, allowing it to
“leak” into the background part of the image. The parameters used for the region
growing algorithm, such as the threshold for similarity between pixels, can also affect
the final segmentation result. If the parameters are not properly tuned or are too lenient,
the algorithm may include more pixels in the segmentation, leading to a larger volume.
### U-Net:
The above image is the ground truth segmentation volume (blue) superimposed with the
U-Net segmentation method (red). We can see that the unet volume prediction matches
the ground truth segmentation it terms of shape and volume much more than the
previous two segmentation approaches. This is likely due to the more advanced
principles of deep learning, which is able to learn more high-level information from the
data as opposed to a simpler segmentation approach that works computationally. This
makes U-Net more adaptive to a wide range of images with varying degrees of noise,
artifacts, and other complexities. Additionally, although the unet approach initially
produced segmentations that included a small area of object class (seen in images for
question 1) these areas were removed due to the larges_segment function which finds
the largest contiguous segment of a binary image. In the case of this segmentation task,
the target region of interest is a small fraction of the entire image. U-Net uses a
weighted loss function to handle class imbalance, which helps to prevent the algorithm
from overemphasizing the background and under-segmenting the target region. These
reasons suggest that the U-Net was the most suitable choice for the segmentation task.

## Comparison of Segmentation Methods on Volume ReconstructionSurface Area

| Segmentation Method | Surface Area (mm²) | Volume (mm³) | IOU   | Accuracy |
|---------------------|--------------------|--------------|-------|----------|
| Ground Truth        | 4,947              | 19,409       | -     | -        |
| Threshold           | 19,680             | 177,727      | 0.062 | 0.641    |
| Region Growing      | 11,202             | 72,164       | 0.137 | 0.741    |
| U-Net               | 5,155              | 22,726       | 0.025 | 0.929    |


The U-Net algorithm achieved the most similar surface area and volume, indicating that
it was able to match the volume’s shape and size better than the other approaches. The
U-Net algorithm achieved the highest accuracy of 0.929. This indicates that U-Net was
able to accurately segment the object of interest in the image. However, the IOU for U-
Net was relatively low at 0.025, indicating that there was relatively little overlap between
the segmented object and the ground truth. This suggests that the segmentation was
not very precise, but rather had a lot of false positives. In contrast, the region growing
algorithm achieved a higher IOU of 0.137, indicating better precision in the
segmentation. However, the accuracy was lower at 0.741, suggesting that region
growing may have missed some true positives. The threshold algorithm’s IOU was the
lowest at 0.062, indicating a high degree of false positives. The accuracy of the
threshold algorithm was intermediate at 0.641, suggesting that it correctly segmented a
portion of the object, but also had a significant number of false positives. The choice of
algorithm will depend on the specific goals of the segmentation task. If precision is a
priority, then region growing may be the best option. If accuracy is the most important
metric, then U-Net may be the best choice.

## Evaluation of Surface Area, Volume, IOU, and Accuracy Metrics
For this question, there were two models of interest that we wanted to improve, the
region growing algorithm and the U-Net model. From initial testing, the region growing
algorithm would leak into the borders of the image and predict the edges as object. This
is because the seed was selected as the darkest pixel of the image, and the borders of
the image (which were not produced from the ultrasound) are black. As such, we hard-
coded the length of the border into the seed selection algorithm and the region growing
algorithm so that they would ignore and not grow into these regions. An example image
segmentation from the original algorithm (left) and the revised version (right) can be
seen below.
We can see that the region growing algorithm was able to successfully ignore the
borders of the image, segmenting the region of interest more precisely.
For improving the U-net model, we would like to investigate ways to improve the training
curves to potentially improve outcomes. As such, we train the model and lower the
learning rate by a factor of 10, since the first model was not able to converge the
validation accuracy and loss. Below are the updated training curves, along with the new
reconstructed volume for the U-Net segmentation.
```
Evaluation Metrics: [0.6100397706031799, 0.9206153154373169]
Model Hyperparameters:
Learning rate: 1e-05
Loss function: categorical_crossentropy
Metrics: ['accuracy']
Confusion matrix:
[[6478557 205872]
[ 358605 67622]]
Classification report:
precision recall f1-score support
0 0.95 0.97 0.96 6684429
1 0.25 0.16 0.19 426227
accuracy 0.92 7110656
```
From these results, we can see that the U-Net has training curves that follow a natural
convergence to a low loss and high accuracy. Additionally, the metrics for both precision
and recall are improved with the new U-Net. By looking at the volume, the U-Net
segmentation aligns more precisely with the shape of the ground truth volume. This
qualitative observation is also reflected in the updated metrics calculation for IOU, which
was found to be 0.102, and is a significant improvement to the previous IOU of 0.025.
Although the IOU metric still did not quite match the performance of the region growing
network, The U-Net segmentation is much better at matching the shape of the vessel
compared to the region growing algorithm. In conclusion, the U-Net architecture offers a
promising approach for vessel segmentation in ultrasound images compared with
threshold and region growing segmentation.