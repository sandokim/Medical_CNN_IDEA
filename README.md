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
