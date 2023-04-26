# RadImageNet
Welcome to the official repository for RadImageNet. 

The RadImageNet database is an open-access medical imaging database. This study was designed to improve transfer learning performance on downstream medical imaging applications. The experiments were designed in four phases. The RadImageNet dataset are available by request at https://www.radimagenet.com/, and the dataset split csv can be found here https://drive.google.com/drive/folders/1FUir_Y_kbQZWih1TMVf9Sz8Pdk9NF2Ym?usp=sharing or https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc/view?usp=sharing. The RadImageNet pretrained models are availble at https://drive.google.com/drive/folders/1Es7cK1hv7zNHJoUW0tI0e6nLFVYTqPqK?usp=sharing or https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc/view?usp=share_link. We will release pretrained Swin Transformer weights later. 

Note: if you don't receive a response to your inquiry (check your junk/spam folder first) you can directly email twdeyer@radimagenet.com or twdeyer@gmail.com.

![alt text](https://github.com/BMEII-AI/RadImageNet/blob/main/util/Slide1.JPG)

The RadImageNet database includes 1.35 million annotated CT, MRI, and ultrasound images of musculoskeletal, neurologic, oncologic, gastrointestinal, endocrine, and pulmonary pathology. The RadImageNet database contains medical images of 3 modalities, 11 anatomies, and 165 pathologic labels. 

![alt text](https://github.com/BMEII-AI/RadImageNet/blob/main/util/Slide2.JPG)


## Top1 and Top5 Accuracy
| Model Name        | Top1 Accuracy    | Top5 Accuracy  |
| ------------- |:---------:| -----:|
| InceptionResNetV2      | 74.0% | 94.3% |
| ResNet50      | 72.3%  | 94.1% |
| DenseNet121 | 73.1%   | 96.1% |
| InceptionV3 | 73.2%   | 92.7% |

## Citation
If you find RadImageNet dataset and/or models useful in your research, please cite:

```yaml
@article{doi:10.1148/ryai.210315,
author = {Mei, Xueyan and Liu, Zelong and Robson, Philip M. and Marinelli, Brett and Huang, Mingqian and Doshi, Amish and Jacobi, Adam and Cao, Chendi and Link, Katherine E. and Yang, Thomas and Wang, Ying and Greenspan, Hayit and Deyer, Timothy and Fayad, Zahi A. and Yang, Yang},
title = {RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning},
journal = {Radiology: Artificial Intelligence},
volume = {0},
number = {ja},
pages = {e210315},
year = {0},
doi = {10.1148/ryai.210315},

URL = { 
        https://doi.org/10.1148/ryai.210315
    
},
eprint = { 
        https://doi.org/10.1148/ryai.210315
}
}
```





## Pretained RadImageNet Models: 
Our RadImageNet pretrained networks include ResNet50, DenseNet121, InceptionResNetV2, and InceptionV3. They are trained solely from RadImageNet medical images and can be used as the starting point on downstream applications using transfer learning. We evaluated RadImageNet pretrained models on 8 medical imaging applications and compared the results to ImageNet pretrained models by using publically available datasets, including thyroid nodule malignancy prediction on ultrasound (1), breast lesion classification on ultrasound (2), ACL and meniscus tear detection on MR (3); pneumonia detection on chest radiographs(4), SARS-CoV-2 detection and COVID-19 identification on chest CT (5,6); and hemorrhage detection on head CT (7). For each medical application, we simulated 24 scenarios to fine tune the models.  The four  CNN bottlenecks were performed with varied learning rates and different numbers of freezing layers. Unfreezing all layers was conducted with learning rates of 0.001 and 0.0001, while freezing all layers and unfreezing top 10 layers were trained with learning rates of 0.01 and 0.001. The average AUROC and standard deviation of these 24 settings were compared between RadImageNet and Imagenet pre-trained models.

Comparions on small datasets (5-fold cross validation)
![alt text](https://github.com/BMEII-AI/RadImageNet/blob/main/util/f3_final.jpg)

Comparions on large datasets
![alt text](https://github.com/BMEII-AI/RadImageNet/blob/main/util/f4_final.jpg)



The sample code for each application is listed upward. 

## Seven public medical applications datasets are available at:

[Thyroid ultrasound](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9287/92870W/An-open-access-thyroid-ultrasound-image-database/10.1117/12.2073532.full?SSO=1
)

[Breast ultrasound](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)

[ACL and meniscus tear detection](https://stanfordmlgroup.github.io/competitions/mrnet/)

[Pneumonia detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

[SARS-CoV-2 detection](http://ncov-ai.big.ac.cn/download?lang=en)

[Intracranial hemorrhage detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)


