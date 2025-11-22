This repository contains the resources for the explainability homework.

# The Use Case: Blindness Detection 
The use case is to detect [diabetic retinopathy](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy) to stop blindness before it's too late. 

Consider you are building a smartphone app that could be used by trained users (nurses, volunteers) to perform screenings to detect potential problems simply by having patients look into the smart phone's camera through a specialized lens attachment (a small 3-d printed holder for a lens). 

Deploying such medical diagnostics as a smart phone app and low-cost hardware extension has the potential to drastically reduce screening costs and make screenings much more available, especially in underresourced regions of the world. Instead of having to walk to a clinic with specialized equipment, trained users could perform screenings at mobile clinics or in patients homes. The app would provide information about a potential risk and encourage the patients to get in contact with medical professionals for more accurate testing and potential treatment. 

In this assignment, you will focus on the model, but also consider its integration into a smartphone app and its use for screening by trained users in underresourced regions (e.g., remote areas, high-poverty areas).


# Training Data

You are provided with a large set of retina images taken using fundus photography under a variety of imaging conditions.
A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:

```
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR
```

Like any real-world data set, you will encounter noise in both the images and labels. 
For details, you can check this [Kaggle competition](https://www.kaggle.com/c/aptos2019-blindness-detection/overview).

# Model
A `ResNet50` model is trained using this data for the classification task. We strongly suggest you use the provided trained model, but you can also train your own or use a different model.

The pre-trained model and data are available in this [Google Drive folder](https://drive.google.com/drive/folders/1X_tTwEixtZdkVWrCae3LK7maP6m2wF4T?usp=drive_link).

# Notebooks
This repository contains two notebooks. 

The training notebook is used for training the model. This is for reference. *You do not need to run this notebook*. You can simply download the pre-trained model from the provided link above.

There is a playground notebook that shows how to load the model and use it to make predictions. This is the kind of code that would be integrated into the smartphone app, either locally or as a remotely deployed model inference service. You can use, extend, and modify the notebook, create your own, or work outside of notebooks -- whichever you prefer.

