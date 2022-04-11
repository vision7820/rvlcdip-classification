from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image

image = Image.open('D:\\Freelance\\Ayame\\document-image-classification-TL-SG\\archive\\test\\invoice\\0000023361.tif').convert('RGB')

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 16 RVL-CDIP classes
predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

if predicted_class_idx == 11:
    print("Predicted class: invoice")
else:
    print("Predicted class: others")