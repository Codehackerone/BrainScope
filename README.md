# Brain-Tumor-Detection-and-Segmentation
## What are Brain Tumors?

![download (17)](https://user-images.githubusercontent.com/64341057/230724439-4c96da6f-75aa-4438-bd61-bfc4aa28d9a1.jpeg)


A brain tumor is an abnormal growth of cells within the brain or surrounding tissues. Brain tumors can be either benign (non-cancerous) or malignant (cancerous), and they can occur at any age.

Benign brain tumors grow slowly, have clear borders and do not spread to other parts of the body. However, even a benign brain tumor can cause serious health problems if it grows in a critical area of the brain or becomes too large.

Malignant brain tumors, on the other hand, grow quickly and can invade nearby healthy brain tissue. They can also spread to other parts of the brain or to other parts of the body through the bloodstream or lymphatic system. Malignant brain tumors are generally more serious and may require aggressive treatment, including surgery, radiation therapy, and chemotherapy.

There are two parts:-
  - Classification. Also, a comparison between ViTs and CNNS
  - Segmentation using Mask RCNNs
  
<img src="https://user-images.githubusercontent.com/64341057/230724568-c2167763-ce1a-4c6e-9cc9-d238aca329e3.png" height="500" width="500">

  
## Datasets
Kaggle: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

## Part 1 - Classification. A comparative Study between ViTs and CNNs.
### CNN

![download (18)](https://user-images.githubusercontent.com/64341057/230724450-d532189e-c8b2-43b8-9707-92af74d5e98c.jpeg)


CNN stands for Convolutional Neural Network, which is a type of deep learning algorithm commonly used for image and video processing tasks. CNNs are designed to automatically learn and extract relevant features from input images and classify them into different categories.

The basic architecture of a CNN consists of several layers, including convolutional layers, pooling layers, and fully connected layers. In the convolutional layer, a set of learnable filters are applied to the input image to produce a feature map, which captures the presence of specific features such as edges, lines, and textures. The pooling layer then reduces the spatial size of the feature map by down-sampling, which helps to reduce the computational cost and prevent overfitting.

The output of the pooling layer is fed into the fully connected layer, which performs the classification task. The fully connected layer consists of several neurons, each representing a different class label, and uses a softmax function to compute the probabilities of each class.

During the training phase, the weights of the filters in the convolutional layers and the neurons in the fully connected layers are optimized through a process called backpropagation, which involves computing the gradient of the loss function with respect to the weights and updating them using an optimization algorithm such as stochastic gradient descent (SGD).

CNNs have achieved state-of-the-art performance in many computer vision tasks, such as image classification, object detection, and semantic segmentation. They are widely used in various applications, such as autonomous driving, medical imaging, and facial recognition.

<img src="https://user-images.githubusercontent.com/64341057/230724661-8f5ed3ec-5500-47b6-bd26-d237ccc859b3.png" width="250" height="250">


### ViT

<img src="https://user-images.githubusercontent.com/64341057/230724461-36f69ee0-27e9-49cd-85df-08ca3b28d141.png" height="250" width="250">


The Vision Transformer (ViT) is a state-of-the-art deep learning model for image recognition, introduced in 2020 by Dosovitskiy et al. Unlike traditional Convolutional Neural Networks (CNNs), ViT uses a transformer architecture, which was originally developed for natural language processing tasks.

The transformer architecture consists of several layers of self-attention and feedforward networks. In the self-attention layer, the input sequence is mapped to a set of queries, keys, and values, which are then used to compute a weighted sum of the values based on their relevance to each query. This allows the model to attend to different parts of the input sequence and capture long-range dependencies.

To apply the transformer to image data, ViT first converts each image into a sequence of patches, which are flattened and passed through an embedding layer to obtain a sequence of feature vectors. These feature vectors are then fed into the transformer layers, which learn to attend to different patches and capture the spatial relationships between them.

To enable the model to perform classification tasks, ViT adds a classification head on top of the transformer output. The classification head consists of a global average pooling layer, which computes the mean of the transformer output along the sequence dimension, followed by a fully connected layer and a softmax activation function.

ViT has achieved state-of-the-art performance on several benchmark image recognition datasets, including ImageNet and CIFAR-100. It has also been shown to generalize well to unseen data and outperform CNNs with a comparable number of parameters. However, ViT requires more computational resources than CNNs, especially for large-scale datasets, and may be less interpretable due to its complex attention mechanism.

#### ViT Architecture.


##### Multi-layer perceptron
MLP stands for Multilayer Perceptron, which is a type of feedforward neural network consisting of multiple layers of neurons, including an input layer, one or more hidden layers, and an output layer. MLPs are commonly used for a wide range of machine learning tasks, including classification, regression, and time series forecasting.

In the case of the Vision Transformer (ViT), an MLP is used as a part of the model's classification head. After passing the image patches through the transformer layers, the resulting feature vectors are concatenated and fed into an MLP that performs the final classification. This MLP takes the concatenated feature vectors as input and applies a series of nonlinear transformations to produce the final output logits.

The use of an MLP in the classification head allows the model to learn complex decision boundaries and perform nonlinear transformations on the feature vectors extracted by the transformer layers. This can be particularly important for image recognition tasks, where the input data can be highly complex and nonlinear

##### Patches as a layer
Patch creation in ViT refers to the process of dividing an input image into a grid of non-overlapping patches, which are then used as input to the transformer layers of the model. This approach allows the ViT model to process images of arbitrary sizes and aspect ratios, without requiring any explicit spatial pooling or downsampling operations.

The size of the patches in ViT is a hyperparameter that can be adjusted depending on the size of the input images and the desired spatial resolution of the model. Typically, patches of size 16x16 or 32x32 pixels are used, although larger or smaller patch sizes can also be used depending on the application.

After the image is divided into patches, each patch is flattened into a 1D vector and passed through an embedding layer to obtain a fixed-size representation of the patch. The resulting sequence of patch embeddings is then fed into the transformer layers of the ViT model, which learn to attend to different patches and capture the spatial relationships between them.

One advantage of patch-based approaches like ViT is that they allow the model to process images of arbitrary sizes and aspect ratios without requiring any explicit spatial pooling or downsampling operations. This can be particularly useful in applications where the input images vary in size and resolution, such as in medical imaging or remote sensing.

However, one potential limitation of patch-based approaches is that they may not capture fine-grained spatial details as well as traditional convolutional neural networks (CNNs), which are specifically designed to learn local and translation-invariant features. Nonetheless, the state-of-the-art performance of ViT on several benchmark image recognition datasets suggests that patch-based approaches can be highly effective for a wide range of computer vision tasks

#### Creating the ViT
The ViT model is composed of multiple Transformer blocks, which use the layers.MultiHeadAttention layer as a self-attention mechanism applied to the sequence of patches. This allows the model to attend to different parts of the input image and capture long-range dependencies between patches. The output of the Transformer blocks is a [batch_size, num_patches, projection_dim] tensor, which is then processed by a classifier head with a softmax activation function to produce the final class probabilities output.

In contrast to the technique described in the original ViT paper, which adds a learnable embedding to the sequence of encoded patches to serve as the image representation, all the outputs of the final Transformer block are reshaped with the layers.Flatten() function and used as the image representation input to the classifier head. This approach is more efficient and simplifies the architecture of the model. However, in cases where the number of patches and the projection dimensions are large, the layers.GlobalAveragePooling1D layer can be used instead to aggregate the outputs of the Transformer block. This can help reduce the number of parameters in the model and improve its generalization performance.

<img src="https://user-images.githubusercontent.com/64341057/230724667-2c4a901f-9132-4c23-beab-692560e26e76.png" width="250" height="250">


### CNNs Vs ViT - Conclusion
CNNs have been the state-of-the-art for image recognition tasks for many years, and are known to be highly effective at capturing local and translation-invariant features in images. They are particularly good at identifying fine-grained details, such as edges and textures, and can learn to recognize complex patterns in large datasets. One of the main advantages of CNNs is that they require relatively few parameters compared to other deep learning models, which makes them faster to train and easier to deploy.

However, CNNs have some limitations. One of the main drawbacks is that they are sensitive to the size and aspect ratio of input images, and may require a lot of preprocessing and data augmentation to achieve good performance on datasets with varying image sizes. They are also not very good at modeling global dependencies between different parts of the image, which can make them less effective for certain types of tasks, such as object detection and segmentation.

ViTs, on the other hand, have shown very promising results for image recognition tasks, especially on datasets with large variations in image sizes and aspect ratios. They are highly effective at modeling long-range dependencies between different parts of the image, which makes them well-suited for tasks such as object detection and segmentation. They also require relatively few parameters compared to other state-of-the-art models, which makes them faster to train and easier to deploy.

However, ViTs also have some limitations. One of the main drawbacks is that they may not capture fine-grained spatial details as well as CNNs, which are specifically designed to learn local and translation-invariant features. They may also be less effective for tasks that require a lot of data augmentation, as the patches used as input to the model may not capture all the relevant details in the image. Additionally, ViTs may require more computational resources than CNNs, especially for larger image sizes and patch resolutions.

## Part 2 - Segmentation using Mask RCNNs

### What is Mask RCNN?
![download (19)](https://user-images.githubusercontent.com/64341057/230724513-199fde98-43b2-4069-b018-889fade46a74.jpeg)


Mask R-CNN is a deep learning model for object detection and instance segmentation, which extends the popular Faster R-CNN model by adding a third branch for predicting object masks. In Mask R-CNN, the model is trained to predict bounding boxes, class labels, and pixel-level masks for each object in an image.

The model consists of a backbone network, such as ResNet or VGG, followed by a Region Proposal Network (RPN) that generates a set of candidate object proposals. These proposals are then refined using a Region of Interest (RoI) pooling layer, which extracts a fixed-size feature map for each proposal. The extracted features are then fed into two separate fully connected branches for object classification and bounding box regression, and a third fully connected branch for mask prediction.

The mask prediction branch in Mask R-CNN uses a fully convolutional network to predict a binary mask for each RoI, where the value of each pixel in the mask corresponds to the probability that the pixel belongs to the object. The predicted masks are then used to perform instance segmentation, which involves separating each object instance in the image and assigning a unique label to each pixel belonging to that instance.

Mask R-CNN has achieved state-of-the-art performance on several benchmark datasets for object detection and instance segmentation, and is widely used in various computer vision applications, including robotics, autonomous vehicles, and medical image analysis.

### Some Results
![__results___27_1](https://user-images.githubusercontent.com/64341057/230725278-8c6ca150-13e6-4c91-b6bf-23117bfcd0db.png)
