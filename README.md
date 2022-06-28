# RadImageNet
Welcome to the official repository for RadImageNet. 

The RadImageNet database is an open-access and standardized database, and includes 1.35million annotated CT, MRI, and ultrasound images of musculoskeletal, neurologic, oncologic, gastrointestinal, endocrine, and pulmonary pathology. The RadImageNet database contains medical images of three modalities, 11 anatomies, and 165 pathologic labels. 

The RadImageNet pretrained models and the RadImageNet dataset will be available by request at https://www.radimagenet.com/.  

![alt text](https://github.com/BMEII-AI/RadImageNet/blob/main/util/database.png)

__Representative images and data structure of the RadImageNet database.__ __a.__ Overview of RadImageNet modalities and anatomies. RadImageNet was constructed with CT, MRI, and ultrasound images, including CT of the chest, CT of the abdomen and pelvis, MRI of the ankle and foot, MRI of the knee, MRI of the hip, MRI of the shoulder, MRI of the brain, MRI of the spine, MRI of the abdomen and pelvis, ultrasound of the abdomen and pelvis, and ultrasound of the thyroid. These images represent the diversity and fundamental structure of the RadImageNet database. **b,c,d The components of the RadImageNet database sub-divided by modalities, anatomies, and classes.** __b.__ summary of anatomies, number of classes, and number of associated images within each anatomy for CT studies.__c.__ summary of anatomies, number of classes, and number of associated images within each anatomy for ultrasound studies.__d.__ summary of anatomies, number of classes, and number of associated images within each anatomy for MRI studies.



Our RadImageNet pretrained networks include ResNet50, DenseNet121, InceptionResNetV2, and InceptionV3. They are trained solely from RadImageNet medical images and can be used as the basis of transfer learning for medical imaging application.


## Pretained RadImageNet Models: 

Transfer learning using pretrained models has been extensively explored in medical imaging.  We evaluated RadImageNet pretrained models on 8 medical imaging applications and compared the results to ImageNet pretrained models by using publically available datasets, including thyroid nodule malignancy prediction on ultrasound (1), breast lesion classification on ultrasound (2), ACL and meniscus tear detection on MR (3); pneumonia detection on chest radiographs(4), SARS-CoV-2 detection and COVID-19 identification on chest CT (5,6); and hemorrhage detection on head CT (7). For each medical application, we simulated 24 scenarios to fine tune the models.  The four  CNN bottlenecks were performed with varied learning rates and different numbers of freezing layers. Unfreezing all layers was conducted with learning rates of 0.001 and 0.0001, while freezing all layers and unfreezing top 10 layers were trained with learning rates of 0.01 and 0.001. The average AUROC and standard deviation of these 24 settings were compared between RadImageNet and Imagenet pre-trained models.

The sample code for each application is listed upward. 

## Seven public medical applications datasets are available at:

[Thyroid ultrasound](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9287/92870W/An-open-access-thyroid-ultrasound-image-database/10.1117/12.2073532.full?SSO=1
)

[Breast ultrasound](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)

[ACL and meniscus tear detection](https://stanfordmlgroup.github.io/competitions/mrnet/)

[Pneumonia detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

[SARS-CoV-2 detection](http://ncov-ai.big.ac.cn/download?lang=en)

[Intracranial hemorrhage detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)


