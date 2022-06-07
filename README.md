[Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf)

[Batch-wise Dice Loss: Rethinking the Data Imbalance for Medical Image Segmentation](https://profs.etsmtl.ca/hlombaert/public/medneurips2019/73_CameraReadySubmission_Med_NeurIPS_2019.pdf)

[Multimodal volume-aware detection and segmentation for brain metastases radiosurgery](https://arxiv.org/pdf/1908.05418.pdf)

[3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes?](https://arxiv.org/pdf/1809.00076.pdf)

#### Overfitting 방지법

We also include a Gaussian noise layer and a dropout layer to avoid overfitting.

<img src="https://github.com/Hyeseong0317/Hutom/blob/main/images/overfitting방지.PNG" width="80%">

#### Deep supervision 효과 (Faster convergence, better accuracy, computational overhead)

We utilize deep supervision which allows more direct backpropagation to the hidden layers for faster convergence and better accuracy [3]. Although deep supervision significantly improves convergence, it is memory expensive especially in 3D networks

[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)

[UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf)

[Nested U-Net code; deep supervision](https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py)

---

#### Vessel Segmentation

[DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection in 3-D Angiographic Volumes](https://www.frontiersin.org/articles/10.3389/fnins.2020.592352/full)

[Pyramid U-Net for Retinal Vessel Segmentation](https://arxiv.org/pdf/2104.02333.pdf)

[Automatic Artery/Vein Classification Using a Vessel-Constraint Network for Multicenter Fundus Images](https://www.readcube.com/articles/10.3389/fcell.2021.659941)

[Transformers in Medical Imaging: A Survey](https://arxiv.org/abs/2201.09873)

[This repo supplements our Survey on Transformers in Medical Imaging](https://github.com/fahadshamshad/awesome-transformers-in-medical-imaging) - reference

[3D Medical image segmentation with transformers tutorial](https://theaisummer.com/medical-segmentation-transformers/)

[Self-attention building blocks for computer vision applications in PyTorch -> Related articles](https://github.com/The-AI-Summer/self-attention-cv/tree/5246e550ecb674f60df76a6c1011fde30ded7f44)

[TransBTS: Multimodal Brain Tumor Segmentation Using Transformer](https://arxiv.org/abs/2103.04430) -> the first attempt to leverage Transformers for 3D multimodal
brain tumor segmentation by effectively modeling local
and global features in both spatial and depth dimensions.

Usefulness of Vit in Multi-organ Segmentation -> Multi-organ segmentation aims to segment several organs simultaneously and is challenging due to inter-class imbalance and varying sizes, shapes, and contrast of different organs. ViT models are particularly suitable for the multiorgan segmentation due to their ability to effectively model global relations and differentiate multiple organs.

Pure Transformers' drawback -> Extensive experiments show the effectiveness of their convolutionfree network on three benchmark 3D medical imaging
datasets related to brain cortical plate [154], pancreas, and hippocampus. One of the drawbacks of using Pure Transformer-based models in segmentation is the quadratic complexity of self-attention with respect to the input image dimensions. This can hinder the ViTs applicability in the segmentation of high-resolution medical images. To mitigate this issue, Cao et al. [125] propose Swin-UNet that, like Swin Transformer [126], computes self-attention within a local window and has linear computational complexity with respect to the input image. Swin-UNet also contains a patch expanding layer for upsampling decoder’s feature maps and shows superior performance in recovering fine details compared to bilinear upsampling. Experiments on Synapse and ACDC [155] dataset demonstrate the effectiveness of the Swin-UNet architectural design.

Hybrid Architectures -> Hybrid architecture-based approaches combine the complementary strengths of Transformers and CNNs to effectively model global context and capture local features for accurate segmentation. We have further categorized these hybrid models into single and multi-scale approaches.

#### Transformer between Encoder and Decoder. 
In this category, Transformer layers are between the encoder and decoder of a U-Shape architecture. These architectures are more suitable to avoid the loss of details during downsampling in the encoder layers. The first work in this category is TransAttUNet [169] that leverages guided attention and multi-scale skip connection to enhance the flexibility of traditional UNet. Specifically, a robust selfaware attention module has been embedded between the encoder and decoder of UNet to concurrently exploit the expressive abilities of global spatial attention and transformer self-attention. Extensive experiments on five benchmark medical imaging segmentation datasets demonstrate the effectiveness of TransAttUNet architecture. Similarly, Yan et al. [170] propose Axial Fusion Transformer UNet (AFTerUNet) that contains a computationally efficient axial fusion layer between encoder and decoder to effectively fuse inter and intra-slice information for 3D medical image segmentation. Experimentation on BCV [171], Thorax-85 [172], and SegTHOR [173] datasets demonstrate the effectiveness of their proposed fusion layer.

#### Transformer in Encoder and Decoder.
Few works integrate Transformer layers in both encoder and decoder of a U-shape architecture to better exploit the global context for medical image segmentation. The first work in this category is UTNet that efficiently reduces the complexity of the self-attention mechanism from quadratic to linear [174]. Furthermore, to model the image content effectively, UTNet exploits the two-dimensional relative position encoding [20].

Similarly, to optimally combine convolution and transformer layers for medical image segmentation, Zhou et al. [144] propose nnFormer, an interleave encoder-decoder based architecture, where convolution layer encodes precise spatial information and Transformer layer encodes global context as shown in Fig. 9

#### 2D Segmentation -> Multi-resolution Vit
Most ViT-based multi-organ segmentation approaches struggle to capture information at multiple scales as they partition the input image into fixed-size patches, thereby losing useful information. To address this issue, Zhang et. al. [183] propose a pyramid medical transformer, PMTrans, that leverage multi-resolution attention to capture correlation at different image scales using a pyramidal architecture [201]. PMTrans works on multiresolution images via an adaptive partitioning scheme of patches to access different receptive fields without changing the overall complexity of self-attention computation.

#### 3D Segmentation
The majority of multi-scale architectures have been proposed for 2D medical image segmentation. To directly handle volumetric data, Hatamizadeh et. al. [35] propose a ViT-based architecture (UNETR) for 3D medical image segmentation. UNETR consists of a pure transformer as the encoder to learn sequence representations of the input volume. The encoder is connected to a CNNbased decoder via skip connections to compute the final segmentation output.

<img src="https://github.com/sandokim/Medical_CNN_IDEA/blob/main/images/UNETR.png" width="80%">

#### The drawbacks of using ViT in Medical Imaging datasets 
* The large domain gap between natural and medical image modalities hinder the usefulness of Vit-based models that are pre-trained on the ImageNet dataset.
* Self-supervised pre-training on medical imaging datasets -> ViT pre-trained on one modality (CT) gives unsatisfactory performance when applied directly to other medical imaging modalities (MRI) due to the large domain gap
* Moreover, recent ViT-based approaches mainly focus on 2D medical image segmentation. Designing customized architectural components by incorporating temporal information for efficient high-resolution and high-dimensional segmentation of volumetric images has not been extensively explored. Recently, few efforts have been made, e.g., UNETR [35] uses Swin Transformer [126] based architectures to avoid quadratic computing complexity; however, it requires further attention from the community

#### Taxanomy of Vit-Based medical image classification approaches

<img src="https://github.com/sandokim/Medical_CNN_IDEA/blob/main/images/Taxanomy of ViT-based.PNG" width="80%">

[Saliency Based Image Segmentation](https://towardsdatascience.com/saliency-based-image-segmentation-473b4cb31774)

[Saliency map](https://en.wikipedia.org/wiki/Saliency_map)

In computer vision, a saliency map is an image that highlights the region on which people's eyes focus first. The goal of a saliency map is to reflect the degree of importance of a pixel to the human visual system.

[Understand your Algorithm with Grad-CAM](https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353)

### The final layer(=Conv_1) captures the genearl region of the object, but it doesn't capture the nuances of the object.

Grad-CAM heat-maps의 final output은 고양이의 전체적인 윤곽을 캡쳐하지만 Grad-CAM 앞단의 레이어들이 캡쳐했던 고양이의 특성들(눈, 수염)을 캡처하지 못하고 있다. 하지만 Grad-CAM heat-maps들의 모든 레이어들을 평균내면 Grad-CAM heat-map이 고양이의 얼굴, 눈, 발을 강조하고 사람의 손은 de-emphasize하는 것을 볼 수 있다.

<img src="https://github.com/sandokim/Medical_CNN_IDEA/blob/main/images/Grad-cam.PNG" width="80%">

We can use the Grad-CAM heat-maps to give us clues into why the model had trouble making the correct classification.
Heat-map에서 고양이의 꼬리가 Blue랑 Green이면 고양이의 꼬리를 제대로 캡쳐하고 있지 못하는 것이며 이는 Grad-CAM heat-maps이 모델이 고양이의 특징을 캡쳐하지 못하고 있다는 것을 설명할 수 있는 근거가 된다.

##### Albumentations -> Data Augmentation tools
We created a supplemental dataset of cats & dogs in transporter crates from google images. Using albumentations, we beefed up the dataset with additional image augmentations.

##### Grad-CAM의 intense (red) areas에 mask로 thresholding을 주어 Data Augmentation의 효과로 인해 모델이 Cat을 더 잘 구분하는지를 확인할 수 있다.
To see where the model is really keying in on, we created a mask that uses thresholding to capture the most intense (red) areas of the Grad-CAM. 

[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)
