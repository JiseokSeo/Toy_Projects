from flask import Flask
import torch
from flask import Flask, jsonify, request
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import skimage


app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = None
with open('model.pkl','rb') as pickle_file:
  model = pickle.load(pickle_file).to(device)

def transform_image(image):
  data_transforms = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
                               ToTensorV2()], p=1.0)
  image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
  transformed = data_transforms(image=image)['image']
  transformed = transformed / transformed.max()
  transformed = transformed.unsqueeze(0)
  return transformed.to(device)

def get_prediction(image):
  image = transform_image(image)
  output_mask = model(image)
  return output_mask

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    image = file.read()
    mask = get_prediction(image)
    return mask

if __name__ == '__main__':
  app.run()