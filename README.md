# FireDectection
Deep learning-based approach for wildfire prediction using satellite images.  
*Group project as part of the course CSC_5AI23_TA*

> [Report](Final_report.pdf).  

---

## Dataset

**Wildfire Prediction Dataset (Satellite Images):** https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data  

## Methods

### Semi-supervised learning

See the code [here](semisupervised_learning/).  
-> [(Model)](semisupervised_learning/resnet50_student_model.zip)

### Fine-tuning ResNet50 + data augmentation

See the code [here](ResNet_data_augmentation/).  
-> [(Model)](ResNet_data_augmentation/saved_models/resnet_finetuned_pytorch_final.pth)

### Fine-tuning EfficientNet-B0 

See the code [here](EffiecentNet/).  
-> [(Model)](EffiecentNet/efficientnet_b2_wildfire.pth)

### Masked AutoEncoding

See the code [here](masked-encoding/).  
-> [(Model)](https://www.dropbox.com/scl/fi/ovb15q41q0h4zcwwzx02q/masked_auto-wildfire-encoder.pt?rlkey=1ttrhwk1rn6eprn040ds2dnuh&st=1lwq2p0y&dl=0)


