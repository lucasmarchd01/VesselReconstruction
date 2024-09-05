from functools import update_wrapper
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label, regionprops
from natsort import natsorted
import pandas as pd
import re


def point_based_segmentation(image, lower_threshold, upper_threshold):
    """This function implements a simple point based segmentation algorithm that segments
    an image into two regions.

    Args:
        image (array): image to be segmented
        lower_threshold (int): lower threshold of values to be segmented
        upper_threshold (int): upper threshold of values to be segmented

    Returns:
        array: segmented image
    """

    segmentation = np.zeros(image.shape)

    mask = (image >= lower_threshold) & (image <= upper_threshold)
    segmentation[mask] = 1

    return segmentation


def region_growing(image, seeds, maximum_difference):
    """
    Implements the simple region growing algorithm that segments an image into two regions

    Args:
    image (array): the greyscale input image
    seeds (tuple): a list of two tuples, containing the coordinates of the seeds for the object and
                  the background, respectively
    maximum_difference (int): the maximum intensity difference between a pixel and its neighbors that is
                    allowed to add the pixel to the same region as the seed

    Returns:
    array: a binary segmentation of the image, where pixels belonging to the object have
                   value 1 and pixels belonging to the background have value 0.
    """
    segmented_img = np.zeros(image.shape)
    object_seed, background_seed = seeds

    object_mean = np.mean(image[object_seed])
    background_mean = np.mean(image[background_seed])

    border_size = 56

    for seed, seed_value in [(object_seed, 1), (background_seed, 0)]:
        segmented_img[seed] = seed_value
        still_to_grow = [seed]
        already_segmented = set([seed])

        while still_to_grow:
            current = still_to_grow.pop()
            neighbors = [
                (current[0] - 1, current[1]),
                (current[0], current[1] - 1),
                (current[0] + 1, current[1]),
                (current[0], current[1] + 1),
            ]
            for neighbor in neighbors:
                if (
                    neighbor[1] >= border_size
                    and neighbor[1] < image.shape[1] - border_size
                    and neighbor[0] >= 0
                    and neighbor[0] < image.shape[0]
                ):
                    if neighbor not in already_segmented:
                        if segmented_img[current] == 1:
                            if (
                                abs(int(image[neighbor]) - object_mean)
                                <= maximum_difference
                            ):
                                segmented_img[neighbor] = seed_value
                                still_to_grow.append(neighbor)
                                already_segmented.add(neighbor)
                        else:
                            if (
                                abs(int(image[neighbor]) - background_mean)
                                <= maximum_difference
                            ):
                                segmented_img[neighbor] = seed_value
                                still_to_grow.append(neighbor)
                                already_segmented.add(neighbor)
    return segmented_img


def largest_segment(binary_img):
    """This function finds the largest segment of a binary image.

    Args:
        binary_img (array): binary image numpy array

    Returns:
        array: numpy array of the largest object segment in the image
    """
    labels = measure.label(binary_img)

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_counts[0] = 0

    if np.any(label_counts):
        largest_label = unique_labels[np.argmax(label_counts)]
        largest_segment = labels == largest_label
    else:
        largest_segment = np.zeros(binary_img.shape, dtype=np.uint8)

    return largest_segment


def find_contours(binary_image, downsample_factor=10):
    """This function finds the contours of a given image of the object class.

    Args:
        binary_image (array): binary image numpy array
        downsample_factor (int, optional): Downsampling factor to reduce the amount of contour points. Defaults to 10.

    Returns:
        array:  a list of contour points of the image
    """
    binary_image = np.uint8(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contours_array = np.concatenate(contours, axis=0)
    contours_array = contours_array[::downsample_factor]
    contours_array = contours_array.reshape(-1, 2)

    return contours_array


def find_probeToReference(filename):
    """This function finds the probeToReference file using the timestamps of the image file.

    Args:
        filename (string): _description_

    Returns:
        _type_: _description_
    """
    probeToRefTimes = pd.read_csv(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/transforms/ProbeToReferenceTimeStamps.csv"
    )
    testImgTimes = pd.read_csv(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Test_Images/Case_05/Test_Images_Ultrasound_Labels.csv"
    )
    for i in testImgTimes.index:
        if testImgTimes["FileName"][i].replace(".png", ".npy") == filename:
            time = testImgTimes["Time Recorded"][i]
    for i in probeToRefTimes.index:
        if probeToRefTimes["Time"][i] == time:
            probeToReference = np.load(
                os.path.join(
                    "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/transforms/ProbeToReference/",
                    probeToRefTimes["Filepath"][i],
                )
            )
    return probeToReference


def transform_contour_points(contours, img_filename):
    """This function transforms the contour points from the las coordinate system to the ras coordinate system.

    Args:
        contours (array): a list of contour points from the image
        img_filename (string): The filename of the image that is being worked with.

    Returns:
        array: a list of the contour points transformed to the ras coordinate system
    """

    probeToRef = find_probeToReference(img_filename)

    imageToProbe = np.load(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/transforms/ImageToProbe.npy"
    )
    referenceToRAS = np.load(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/transforms/ReferenceToRAS.npy"
    )
    size = contours.shape[0]
    contours_point = np.hstack((contours, np.zeros((size, 1)), np.ones((size, 1))))
    point_ras = np.matmul(
        np.matmul(np.matmul(contours_point, imageToProbe), probeToRef), referenceToRAS
    )
    ras = point_ras[:, :3]
    return ras


def display_image(image, title=None):
    """This function displays a grayscale image on the screen

    Args:
        image (array): array representation of an image
    """
    plt.imshow(image, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.show()  # BrBG


def getImages(dataSetPath):
    """This function gathers the images from a dataset path

    Args:
        dataSetPath (string): directory of the images

    Returns:
        list: a list of arrays containing the images
    """
    images = []
    for root, _, files in os.walk(dataSetPath):
        for file in natsorted(files):
            if file.endswith(".png") and "segmentation" not in file:
                img = cv2.imread(os.path.join(root, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img)
    return images


def getSegmentations(dataSetPath):
    """This function gathers the segmentations from a dataset path

    Args:
        dataSetPath (string): directory of the segmentations

    Returns:
        list: a list of arrays containing the segmentations
    """
    segmentations = []
    for root, _, files in os.walk(dataSetPath):
        for file in natsorted(files):
            if file.endswith(".png") and "segmentation" in file:
                seg = cv2.imread(os.path.join(root, file))
                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
                segmentations.append(seg)
    return segmentations


def getypred(dataSetPath):
    """This function returns a list of predicted segmentations from a model.

    Args:
        dataSetPath (string): path to .npy file predictions

    Returns:
        list: a list of arrays containing the segmentations
    """
    segmentations = []
    for root, _, files in os.walk(dataSetPath):
        for file in natsorted(files):
            if file.endswith(".npy"):
                seg = np.load(os.path.join(root, file))
                segmentations.append(seg)
    return segmentations


def find_optimal_threshold(images):
    """This function finds the optimal thresholds to segment an image

    Args:
        images (list): the images in the dataset

    Returns:
        tuple: (lower threshold, upper threshold)
    """

    images = np.array(images)
    means = []
    std_devs = []
    for image in images:
        image_flat = image.flatten()
        pixel_mean = np.mean(image_flat)
        std_dev = np.std(image_flat)
        means.append(pixel_mean)
        std_devs.append(std_dev)
    pixel_mean = np.mean(means)
    std_dev = np.mean(std_devs)

    lower_threshold = pixel_mean - std_dev
    upper_threshold = pixel_mean + std_dev

    return lower_threshold, upper_threshold


def find_seeds(image):
    """
    Automatically selects seed coordinates for a region growing algorithm based on the characteristics of the image

    Args:
    image (array): the greyscale input image

    Returns:
    tuple: a list of two tuples, containing the coordinates of the seeds for the object and the background, respectively
    """
    border_width = 56
    non_border_region = image[border_width:-border_width, :]
    mean_intensity = np.mean(non_border_region)

    max_intensity_coords = np.argwhere(non_border_region == np.max(non_border_region))
    min_intensity_coords = np.argwhere(non_border_region == np.min(non_border_region))

    max_distance = np.abs(np.max(non_border_region) - mean_intensity)
    min_distance = np.abs(np.min(non_border_region) - mean_intensity)

    if min_distance > max_distance:
        object_seed = tuple(min_intensity_coords[0] + [border_width])
        background_seed = tuple(max_intensity_coords[0] + [border_width])
    else:
        object_seed = tuple(max_intensity_coords[0] + [border_width])
        background_seed = tuple(min_intensity_coords[0] + [border_width])

    return [background_seed, object_seed]


def generate_contours(dataSetPath):
    """This function generates the contour arrays that are used in the Slicer module.

    Args:
        dataSetPath (string): Dataset that points to the .npy files of the individual slices

    Returns:
        array: array of shape (x, 3) containing the contours of the image is ras coordinates.
    """

    contours = np.zeros((0, 3))
    for root, _, files in os.walk(dataSetPath):
        for file in natsorted(files):
            if file.endswith(".npy"):
                image = np.load(os.path.join(root, file))
                biggest_segment = largest_segment(image)
                contour = find_contours(biggest_segment)
                if contour is not None:
                    ras_points = transform_contour_points(contour, file)
                    contours = np.vstack((contours, ras_points))

    return contours


def IOU_acc(gt_mask, pred_mask):
    """
    Computes accuracy and IoU metric for a predicted segmentation mask and its ground truth.

    Args:
    - pred_mask (ndarray): Predicted binary segmentation mask of shape (H, W).
    - gt_mask (ndarray): Ground truth binary segmentation mask of shape (H, W).

    Returns:
    - accuracy (float): Percentage of pixels that are correctly classified.
    - iou (float): Intersection over Union (IoU) metric.
    """

    assert pred_mask.shape == gt_mask.shape

    num_correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    accuracy = float(num_correct_pixels) / total_pixels
    intersection = np.sum(np.logical_and(pred_mask, gt_mask))
    union = np.sum(np.logical_or(pred_mask, gt_mask))
    iou = float(intersection) / union

    return accuracy, iou


if __name__ == "__main__":
    train_images = getImages(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Training_Images"
    )
    lower_threshold, upper_threshold = find_optimal_threshold(train_images)

    test_images = getImages(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Test_Images"
    )
    test_segmentations = getSegmentations(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Test_Images"
    )

    # Point based segmetnation
    count = 1
    for i in range(len(test_images)):
        segmentation = point_based_segmentation(test_images[i], 0, 2)
        display_image(test_segmentations[i + 60])
        display_image(segmentation)
        np.save(
            "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/pointbased_predictions/Test_Images_Ultrasound_{:05d}".format(
                count
            ),
            segmentation,
        )
        count += 1

    # Region based segmentation
    count = 1
    for i in range(len(test_images)):
        seeds = find_seeds(test_images[i + 60])
        segmentation = region_growing(test_images[i], seeds, 7)
        display_image(test_segmentations[i + 60])
        display_image(segmentation)
        np.save(
            "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/regionbased_predictions/Test_Images_Ultrasound_{:05d}".format(
                count
            ),
            segmentation,
        )
        count += 1

    # Ground truth segmentations
    count = 1
    for i in range(len(test_segmentations)):
        np.save(
            "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/groundtruth_segmentations/Test_Images_Ultrasound_{:05d}".format(
                count
            ),
            test_segmentations[i],
        )
        count += 1

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/groundtruth_segmentations"
    groundTruthContours = generate_contours(dataset)
    np.save(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/groundTruthContours.npy",
        groundTruthContours,
    )

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/pointbased_predictions"
    thresholding = generate_contours(dataset)
    np.save(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/thresholdingContours.npy",
        thresholding,
    )

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/unet_predictions2"
    unet = generate_contours(dataset)
    np.save(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/unetContours.npy",
        unet,
    )

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/regionbased_predictions"
    region = generate_contours(dataset)
    np.save(
        "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/regionGrowingContours.npy",
        region,
    )

    # ------------------------------------- Metrics --------------------------------------------------

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/pointbased_predictions"
    predictions = getypred(dataset)

    ious = []
    accs = []
    for i in range(len(test_segmentations)):
        y_true = test_segmentations[i]
        y_pred = predictions[i]
        acc, iou = IOU_acc(y_true, y_pred)
        ious.append(iou)
        accs.append(acc)
    ious = np.array(ious)
    print("----threshold----")
    print(np.mean(ious))

    accs = np.array(accs)
    print(np.mean(accs))

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/groundtruth_segmentations"
    predictions = getypred(dataset)

    ious = []
    accs = []
    for i in range(len(test_segmentations)):
        y_true = test_segmentations[i]
        y_pred = predictions[i]
        acc, iou = IOU_acc(y_true, y_pred)
        ious.append(iou)
        accs.append(acc)
    print("---groundtruth---")
    ious = np.array(ious)
    print(np.mean(ious))

    accs = np.array(accs)
    print(np.mean(accs))

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/regionbased_predictions"
    predictions = getypred(dataset)

    ious = []
    accs = []
    for i in range(len(test_segmentations)):
        y_true = test_segmentations[i]
        y_pred = predictions[i]
        acc, iou = IOU_acc(y_true, y_pred)
        ious.append(iou)
        accs.append(acc)
    print("---region based----")
    ious = np.array(ious)
    print(np.mean(ious))

    accs = np.array(accs)
    print(np.mean(accs))

    dataset = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/unet_predictions2"
    predictions = getypred(dataset)

    ious = []
    accs = []
    for i in range(len(test_segmentations)):
        y_true = test_segmentations[i]
        y_pred = predictions[i]
        acc, iou = IOU_acc(y_true, y_pred)
        ious.append(iou)
        accs.append(acc)
    print("---unet---")
    ious = np.array(ious)
    print(np.mean(ious))

    accs = np.array(accs)
    print(np.mean(accs))
