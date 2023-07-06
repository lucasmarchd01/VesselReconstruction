####################################################################################
#   Train_unet.py
#       Script for implementing and training a unet model for segmentation
#   Name: Lucas March
#   Student Number: 20144315
#   Date: March 24, 2023
#####################################################################################

from imghdr import tests
import cv2
import numpy
import os

import scipy.spatial.distance
import sklearn
import tensorflow
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
import math
import random
from UnetSequence import UnetSequence
from sklearn.model_selection import train_test_split
from natsort import natsorted
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score



def define_UNet_Architecture(imageSize,numClasses,filterMultiplier=10):
    input_ = layers.Input(imageSize)
    skips = []
    output = input_

    num_layers = int(numpy.floor(numpy.log2(imageSize[0])))
    down_conv_kernel_sizes = numpy.zeros([num_layers],dtype=int)
    up_conv_kernel_sizes = numpy.zeros([num_layers], dtype=int)

    down_filter_numbers = numpy.zeros([num_layers],dtype=int)
    up_filter_numbers = numpy.zeros([num_layers],dtype=int)

    for layer_index in range(num_layers):
        up_conv_kernel_sizes[layer_index]=int(4)
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = int((layer_index+1)*filterMultiplier + numClasses)
        up_filter_numbers[layer_index] = int((num_layers-layer_index-1)*filterMultiplier + numClasses)
        
    #Create contracting path
    for kernel_shape,num_filters in zip(down_conv_kernel_sizes,down_filter_numbers):
        skips.append(output)
        output = layers.Conv2D(num_filters,(kernel_shape,kernel_shape),
                               strides=2,
                               padding="same",
                               activation="relu",
                               bias_regularizer=l1(0.))(output)

    #Create expanding path
    lastLayer = len(up_conv_kernel_sizes)-1
    layerNum = 0
    for kernel_shape,num_filters in zip(up_conv_kernel_sizes,up_filter_numbers):
        output = layers.UpSampling2D()(output)
        skip_connection_output = skips.pop()
        output = layers.concatenate([output,skip_connection_output],axis=3)
        if layerNum!=lastLayer:
            output = layers.Conv2D(num_filters,(kernel_shape,kernel_shape),
                                   padding="same",
                                   activation="relu",
                                   bias_regularizer=l1(0.))(output)
        else: #Final output layer
            output = layers.Conv2D(num_filters, (kernel_shape, kernel_shape),
                                   padding="same",
                                   activation="softmax",
                                   bias_regularizer=l1(0.))(output)
        layerNum+=1
    return Model([input_],[output])

#############################################################################################################
# Question 2:
#    Complete the following function to generate your simulated images and segmentations. You may implement
#    many helper functions as necessary to do so.
#############################################################################################################
def generateDataset(dataSetPath):
    '''
    Args:
        datasetDirectory: the path to the directory where your images and segmentations will be stored
        num_images: the number of images that you wish to generate
        imageSize: the shape of your images and segmentations
    Returns:
        None: Saves all images and segmentations to the dataset directory
    '''

    for vol_file in os.listdir(dataSetPath):
        if vol_file.endswith(".npy") and not vol_file.endswith("_segmentation.npy"):
            print(vol_file)
            vol_path = os.path.join(dataSetPath, vol_file)
            vol = numpy.load(vol_path)
            
            seg_file = vol_file.replace(".npy", "_segmentation.npy")
            seg_path = os.path.join(dataSetPath, seg_file)
            seg = numpy.load(seg_path)
            
            vol_dir = os.path.join(dataSetPath, vol_file.replace(".npy", ""))
            seg_dir = os.path.join(dataSetPath, seg_file.replace("_segmentation.npy", "_segmentation"))
            os.makedirs(vol_dir, exist_ok=True)
            os.makedirs(seg_dir, exist_ok=True)
            
            for i in range(vol.shape[0]):
                vol_slice_path = os.path.join(vol_dir, vol_file.replace(".npy", "_") + f"{i:03}")
                seg_slice_path = os.path.join(seg_dir,  seg_file.replace("_segmentation.npy", "_segmentation_") + f"{i:03}")
                vol_slice = vol[i,:,:]
                seg_slice = seg[i,:,:]
                numpy.save(vol_slice_path, vol_slice)
                numpy.save(seg_slice_path, seg_slice)
    return

#############################################################################################################
# Question 3:
#    Complete the following function so that it returns your data divided into 3 non-overlapping sets
# You do not need to read the images at this stage
#############################################################################################################
def splitDataIntoSets(images,segmentations):
    '''
    Args:
        images: list of all image filepaths in the dataset
        segmentations: list of all segmentation filepaths in the dataset

    Returns:
        trainImg: list of all image filepaths to be used for training
        trainSeg: list of all segmentation filepaths to be used for training
        valImg: list of all image filepaths to be used for validation
        valSeg: list of all segmentation filepaths to be used for validation
        testImg: list of all image filepaths to be used for testing
        testSeg: list of all segmentation filepaths to be used for testing
    '''
    patientIDs = [os.path.basename(image).split("_")[0] for image in images]
    uniqueIDs = numpy.unique(patientIDs)

    trainIDs, valIDs = train_test_split(uniqueIDs, test_size=0.25, random_state=42)
    
    test_images = getImages("/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Test_Images")
    test_segmentations = getSegmentations("/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Test_Images")

    trainData = [(image, seg) for image, seg, pid in zip(images, segmentations, patientIDs) if pid in trainIDs]
    valData = [(image, seg) for image, seg, pid in zip(images, segmentations, patientIDs) if pid in valIDs]
    testData = [(image, seg) for image, seg in zip(test_images, test_segmentations)]

    trainImg, trainSeg = zip(*trainData)
    valImg, valSeg = zip(*valData)
    testImg, testSeg = zip(*testData)

    return (trainImg, trainSeg), (valImg, valSeg), (testImg, testSeg)

#############################################################################################################
# Question 5:
#    Complete the following function so that it will create a plot for the training and validation loss/metrics.
#    Training and validation should be shown on the same graph so there should be one plot per loss/metric
#############################################################################################################
def plotLossAndMetrics(trainingHistory):
    '''
    Args:
        trainingHistory: The dictionary containing the progression of the loss and metrics for training and validation
    Returns:
        None: should save each graph as a png
    '''

    for key in trainingHistory.history.keys():
        if key.startswith('val_'):
            continue
        plt.plot(trainingHistory.history[key], label='Training ' + key)
        plt.plot(trainingHistory.history["val_" + key], label='Validation ' + key)
        plt.title(key.capitalize())
        plt.xlabel('Epoch')
        plt.ylabel(key.capitalize())
        plt.legend()
        plt.savefig(key + '.png')
        plt.clf()


def display_image(image, title=None):
    """This function displays a grayscale image on the screen

    Args:
        image (array): array representation of an image
    """
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show() # BrBG

def getImages(dataSetPath):
    images = []
    for root, _, files in os.walk(dataSetPath):
        for file in files:
            if file.endswith('.png') and "segmentation" not in file:
                images.append(os.path.join(root, file))
    images = natsorted(images)
    return images

def getSegmentations(dataSetPath):
    segmentations = []
    for root, _, files in os.walk(dataSetPath):

        for file in files:
            if file.endswith('.png') and "segmentation" in file:
                segmentations.append(os.path.join(root, file))    
    segmentations = natsorted(segmentations)
    return segmentations


def main():

    dataSetPath = os.path.join("/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Training_Images")
    #generateDataset(dataSetPath) #this line only needs to be run once
    images = getImages(dataSetPath)
    segmentations = getSegmentations(dataSetPath)
    trainData,valData,testData = splitDataIntoSets(images,segmentations)

    trainSequence = UnetSequence(trainData)
    valSequence = UnetSequence(valData)
    testSequence = UnetSequence(testData,shuffle=False)

    unet = define_UNet_Architecture(imageSize=(128,128,1),numClasses=2)
    unet.summary()

    #############################################################################################################
    # Set the values of the following hyperparameters
    #############################################################################################################

    learning_rate = 0.00001
    lossFunction = 'categorical_crossentropy'
    metrics=["accuracy"]
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    numEpochs = 25
    modelName = 'unet_ce_vessel2.h5'
    earlyStoppingCallback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
    modelCheckPointCallback = tensorflow.keras.callbacks.ModelCheckpoint(modelName, verbose=1,
                                                      monitor='val_accuracy', mode='max', save_weights_only=True,
                                                      save_best_only=True)
    

    #############################################################################################################
    # Create model checkpoints here, and add the variable names to the callbacks list in the compile command
    #############################################################################################################

    unet.compile(optimizer=optimizer,
                 loss=lossFunction,
                 metrics=metrics,
                 run_eagerly=True)

    history = unet.fit(x=trainSequence,
                       validation_data=valSequence,
                       epochs=numEpochs,
                       callbacks=[modelCheckPointCallback, earlyStoppingCallback])

    plotLossAndMetrics(history)

    #############################################################################################################
    # Add additional code for generating predictions and evaluations here
    #############################################################################################################


    modelPath = "/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/Run3/" + modelName

    # Load the weights from the given path
    unet.load_weights(modelPath)

    # Generate predictions for the test data
    y_pred = unet.predict(testSequence)
    y_pred = numpy.argmax(y_pred, axis=-1)
    print(y_pred.shape)
    count = 1
    for pred in y_pred:
        pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)
        numpy.save("/Users/lucasmarch/OneDrive/CISC 472/Assignment_4/Assignment4Data/unet_predictions2/Test_Images_Ultrasound_{:05d}".format(count), pred)
        count += 1


    # Compute the evaluation metrics on the test data
    evaluation_metrics = unet.evaluate(x=testSequence)

    for i in range(4):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(testSequence[i+16][0][0], cmap='gray')
        ax[0].set_title('Input Image')
        ax[1].imshow(y_pred[i], cmap='gray')
        ax[1].set_title('Predicted Segmentation')
        plt.show()

    # Compute the confusion matrix
    y_pred = y_pred.ravel()
    y_true = testSequence.get_y_true()
    y_true = numpy.argmax(y_true, axis=-1).ravel()
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    cls_report = classification_report(y_true, y_pred)
    print(cls_report)

    # Save the evaluation metrics, confusion matrix, and model hyperparameters to a text file
    with open('evaluation_metrics.txt', 'w') as f:
        f.write(f"Evaluation Metrics: {evaluation_metrics}\n")
        f.write(f"Model Hyperparameters:\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Loss function: {lossFunction}\n")
        f.write(f"Metrics: {metrics}\n")
        f.write(f"Confusion matrix:\n {cm}\n")
        f.write(f"Classification report:\n {cls_report}")
        # f.write("\nUsing Image contrast enhancement")


def IOU(y_true,y_pred):
    y_true_f = K.flatten(y_true[:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,1])
    intersection = K.sum(y_true_f*y_pred_f)
    return(intersection)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection)

def IOU_Loss(y_true,y_pred):
    return 1-IOU(y_true,y_pred)

#############################################################################################################
# Question 7:
#    Complete the following function to compute the mean hausdorff distance
#############################################################################################################
def hausdorffDistance(ytrue,ypred):
    '''
    Computes the mean hausdorff distance between predicted segmentation and ground truth. All values are one-hot encoded.
    Args:
        ytrue: ground truth segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)
        ypred: predicted segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)

    Returns:
        mean_hausdorff_distance: the mean hausdorff distance across all samples in the batch
    '''

    y_true = tensorflow.argmax(ytrue, axis=-1)
    y_pred = tensorflow.argmax(ypred, axis=-1)


    dists = []
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[-1]):
            dists.append(directed_hausdorff(y_true[i,:,:], y_pred[i,:,:])[0])
    dists = numpy.array(dists)
    
    mean_hausdorff_distance = numpy.mean(dists)

    return mean_hausdorff_distance

#############################################################################################################
# Question 7:
#    Complete the following function to compute the mean dice coefficient
#############################################################################################################
def diceCoefficient(ytrue,ypred):
    '''
    Computes the mean dice coefficient between predicted segmentation and ground truth. All values are one-hot encoded.
    Args:
        ytrue: ground truth segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)
        ypred: predicted segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)

    Returns:
        mean_dice_coefficient: the mean dice coefficient across all samples in the batch
    '''
    ytrue_flat = K.flatten(ytrue[:,:,1])
    ypred_flat = K.flatten(ypred[:,:,1])
    intersection = numpy.sum(ytrue_flat * ypred_flat, axis=0)
    union = numpy.sum(ytrue_flat + ypred_flat, axis=0)
    
    eps = 1e-6
    dice_coefficients = (2 * intersection + eps) / (union + eps)
    
    mean_dice_coefficient = numpy.mean(dice_coefficients)
    
    return mean_dice_coefficient

main()



