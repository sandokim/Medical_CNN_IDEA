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
