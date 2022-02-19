from typing import Dict

import SimpleITK
import tqdm
import json
from pathlib import Path
import tifffile
import numpy as np


import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pretrainedmodels import se_resnext50_32x4d
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from evalutils.io import ImageLoader

GPU = torch.cuda.is_available()
if GPU:
    device = "cuda"
else:
    device = "cpu"

def getOD(model, image):
    results = model(image, size=288).pandas().xywhn[0]
    cols = ["xcenter", "ycenter", "width",  "height", "confidence", "class", "name"]
    df = pd.DataFrame([[0.5, 0.5, 1, 1, 0.001, 0, "OD"]], columns=cols)
    df = df.append(results, ignore_index=True)
    if df.shape[0]>1:
          df = df[df["confidence"] == np.max(df["confidence"])]
    return df.iloc[0]["xcenter"], df.iloc[0]["ycenter"], df.iloc[0]["width"], df.iloc[0]["height"], df.iloc[0]["confidence"]

def getBoundsVanilla(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(gray, 5, 255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         
    curated_contours = []
    for c in contours:
        if len(c) > 1000:
            curated_contours.append(c)

    if not len(curated_contours) >= 1:
        curated_contours = contours
            
    stacked = np.vstack(curated_contours)
        
    reshaped_contours = stacked.reshape((stacked.shape[0], 2))
    d_x = np.max(reshaped_contours[:, 0]) - np.min(reshaped_contours[:, 0])
    d_y = np.max(reshaped_contours[:, 1]) - np.min(reshaped_contours[:, 1])

    d = np.max((d_x, d_y))
    center = [(np.max(reshaped_contours[:, 0]) + np.min(reshaped_contours[:, 0]))/2, (np.max(reshaped_contours[:, 1]) + np.min(reshaped_contours[:, 1]))/2 ]

    x_min = int(center[1] - d/2)
    x_max = int(center[1] + d/2)
    y_min = int(center[0] - d/2)
    y_max = int(center[0] + d/2)
        
    if x_min < y_min and x_min < 0:
        x_min = 0
        x_max = img.shape[0]
        y_min = np.max((0, y_min))
        y_max = np.min((y_max, img.shape[1]))
    elif y_min < x_min and y_min < 0:
        y_min = 0
        y_max = img.shape[1]
        x_min = np.max((0, x_min))
        x_max = np.min((x_max, img.shape[0]))

    to_pad = np.abs((x_max+1 - x_min) - (y_max+1 - y_min))    
    cropped_image = img[x_min:x_max+1, y_min:y_max+1]

    if x_max+1 - x_min > y_max+1 - y_min:
        padded = cv2.copyMakeBorder(cropped_image, top=0, bottom=0, left=int(to_pad/2), right=int(to_pad/2), borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    else:
        padded = cv2.copyMakeBorder(cropped_image, top=int(to_pad/2), bottom=int(to_pad/2), left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

    return padded

#image = cv2.imread("Training\\TRAIN000054.jpg")
#Conveting to PIL image
def getODFromNumpyArray(model, image):
    #image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = getBoundsVanilla(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)

    #Crop to 288 for Yolo
    im288 = im_pil.resize((288,288))
    OD_xcent, OD_ycent, OD_width, OD_height, OD_conf = getOD(model, im288)
    #print(OD_xcent, OD_ycent, OD_width, OD_height, OD_conf)
    
    #cropping based on bounds
    x_start = OD_xcent - OD_width/2
    x_end = x_start + OD_width
    y_start = OD_ycent - OD_height/2
    y_end = y_start + OD_height

    scale = im_pil.size[0]
    x_start = int(x_start * scale)
    x_end = int(x_end * scale)
    y_start = int(y_start * scale)
    y_end = int(y_end * scale)

    y_diff = y_end - y_start
    x_diff = x_end - x_start
    if x_diff < y_diff:
        x_adj = y_diff - x_diff
        left_adj = int(x_adj/2)
        right_adj = x_adj - left_adj
        x_start -= left_adj
        x_end += right_adj
    elif y_diff < x_diff:
        y_adj = x_diff - y_diff
        up_adj = int(y_adj/2)
        down_adj = y_adj - up_adj
        y_start -= up_adj
        y_end += down_adj

    OD_fullsize = im_pil.crop((x_start, y_start, x_end, y_end))
    return OD_fullsize, OD_conf

def getPreds(OD_fullsize, OD_conf, densenet, seresnext, effnet, vgg16, inception):
    im120 = OD_fullsize.resize((120,120))
    im224 = OD_fullsize.resize((224,224))
    im299 = OD_fullsize.resize((299,299))
    
    seresnext_pred = predictSingle(seresnext, im224, 224)
    densenet_pred = predictSingle(densenet, im120, 120)
    vgg16_pred = predictSingle(vgg16, im120, 120)
    effnet_pred = predictSingle(effnet, im224, 224)
    inception_pred = predictSingle(inception, im299, 299)
    
    rg_likelihood = np.mean((seresnext_pred, densenet_pred, vgg16_pred, effnet_pred, inception_pred))
    rg_binary = bool(rg_likelihood > 0.5)
    
    pred_std = np.std((seresnext_pred, densenet_pred, vgg16_pred, effnet_pred, inception_pred))
    
    if rg_likelihood >= 0 and rg_likelihood <= 0.5:
        pred_scale =  2 * rg_likelihood
    else:
        pred_scale = -2 * (rg_likelihood - 1)
    
    ungradability_score = (1-OD_conf) * pred_scale * pred_std
    ungradability_binary = bool(ungradability_score > 0.075)
    
    out = {
        "referable-glaucoma-likelihood": rg_likelihood,  # Likelihood for 'referable glaucoma'
        "referable-glaucoma-binary": rg_binary,  # True if 'referable glaucoma', False if 'no referable glaucoma'
        "ungradability-score": ungradability_score,  # True if 'ungradable', False if 'gradable'
        "ungradability-binary": ungradability_binary  # The higher the value, the more likely the label is 'ungradable'
    }
    
    return out
    
def getModels(yolo_weights, densenet_weights, seresnext_weights, effnet_weights, vgg_16_weights, inception_weights):
    densenet = models.densenet161(pretrained=False)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(num_ftrs, 1, bias=True))
    if GPU:
        densenet.load_state_dict(torch.load(densenet_weights))
    else:
        densenet.load_state_dict(torch.load(densenet_weights, map_location=torch.device('cpu')))

    seresnext = se_resnext50_32x4d(num_classes=1000, pretrained=None)
    num_ftrs = seresnext.last_linear.in_features
    seresnext.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    seresnext.last_linear = nn.Sequential(nn.Dropout(0.0), nn.Linear(num_ftrs, 1, bias=True))
    if GPU:
        seresnext.load_state_dict(torch.load(seresnext_weights)) 
    else:
        seresnext.load_state_dict(torch.load(seresnext_weights, map_location=torch.device('cpu')))
    
    effnet = EfficientNet.from_name('efficientnet-b7')
    num_ftrs = effnet._fc.in_features
    effnet._fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(num_ftrs, 1, bias=True))
    if GPU:
        effnet.load_state_dict(torch.load(effnet_weights)) 
    else:
        effnet.load_state_dict(torch.load(effnet_weights, map_location=torch.device('cpu')))

    vgg16 = models.vgg16()
    vgg16.classifier[5] = nn.Dropout(0.0)
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1, bias=True))
    if GPU:
        vgg16.load_state_dict(torch.load(vgg_16_weights)) 
    else:
        vgg16.load_state_dict(torch.load(vgg_16_weights, map_location=torch.device('cpu')))

    yolo = torch.hub.load(yolo_weights, 'custom', path='yolov5_weights.pt', source="local")
    
    inceptionv3 = models.inception_v3(pretrained=False)
    inceptionv3.dropout = nn.Dropout(0.0)
    num_ftrs = inceptionv3.fc.in_features
    inceptionv3.fc = nn.Linear(num_ftrs, 1, bias=True)
    if GPU:
        inceptionv3.load_state_dict(torch.load(inception_weights)) 
    else:
        inceptionv3.load_state_dict(torch.load(inception_weights, map_location=torch.device('cpu'))) 


    return {"yolo": yolo, "densenet": densenet,
        "seresnext": seresnext, "effnet": effnet, "vgg16": vgg16, "inception": inceptionv3}

def predictSingle(model, img, size):
    transform = transforms.ToTensor()
    tensor = transform(img)
    resizeTransform = transforms.Resize([size, size])
    tensor = resizeTransform(tensor)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if GPU:
        tensor = normalize(tensor).cuda()
    else:
        tensor = normalize(tensor).cpu()
    
    if GPU:
        model.cuda()
    else:
        model.cpu()
    with torch.no_grad():
        model.eval()
        output = torch.sigmoid(model(tensor[None, ...]))

    result = output.cpu().detach().numpy().flatten()
    return result[0]

def predict(image, models):
    OD_fullsize, OD_conf = getODFromNumpyArray(models["yolo"], image)

    out = getPreds(OD_fullsize, OD_conf, models["densenet"], models["seresnext"],
        models["effnet"], models["vgg16"], models["inception"])
    return out, OD_conf
    

class DummyLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return str(fname)


    @staticmethod
    def hash_image(image):
        return hash(image)


class airogs_algorithm(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self._file_loaders = dict(input_image=DummyLoader())

        self.output_keys = ["multiple-referable-glaucoma-likelihoods", 
                            "multiple-referable-glaucoma-binary",
                            "multiple-ungradability-scores",
                            "multiple-ungradability-binary"]
    
    def load(self):
        for key, file_loader in self._file_loaders.items():
            fltr = (
                self._file_filters[key] if key in self._file_filters else None
            )
            self._cases[key] = self._load_cases(
                folder=Path("/input/images/color-fundus/"),
                file_loader=file_loader,
                file_filter=fltr,
            )

        pass
    
    def combine_dicts(self, dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                if k not in out:
                    out[k] = []
                out[k].append(v)
        return out
    
    def process_case(self, *, idx, case):
        # Load and test the image(s) for this case
        if case.path.suffix == '.tiff':
            results = []
            with tifffile.TiffFile(case.path) as stack:
                for page in tqdm.tqdm(stack.pages):
                    input_image_array = page.asarray()
                    results.append(self.predict(input_image_array=input_image_array))
        else:
            input_image = SimpleITK.ReadImage(str(case.path))
            input_image_array = SimpleITK.GetArrayFromImage(input_image)
            results = [self.predict(input_image_array=input_image_array)]
        
        results = self.combine_dicts(results)

        # Test classification output
        if not isinstance(results, dict):
            raise ValueError("Expected a dictionary as output")

        return results

    def predict(self, *, input_image_array: np.ndarray) -> Dict:
        print("----Submission 2-----")
        print("DEVICE is: " + device)
        yolo_weights = 'yolov5'
        se_resnext_weights = "se_resnext_weights.pth"
        densenet_weights = "densenet_weights.pth"
        vgg16_weights = "vgg16_weights.pth"
        effnet_weights = "effnetb7_weights.pth"
        inception_weights = "inceptionv3_weights.pth"
        my_models = getModels(yolo_weights, densenet_weights, se_resnext_weights, effnet_weights, vgg16_weights, inception_weights)

        results, OD_conf = predict(input_image_array, my_models)
        rg_likelihood = float(results["referable-glaucoma-likelihood"])
        rg_binary = results["referable-glaucoma-binary"]
        ungradability_score = float(results["ungradability-score"])
        ungradability_binary = results["ungradability-binary"]

        out = {
            "multiple-referable-glaucoma-likelihoods": rg_likelihood,
            "multiple-referable-glaucoma-binary": rg_binary,
            "multiple-ungradability-scores": ungradability_score,
            "multiple-ungradability-binary": ungradability_binary
        }

        return out

    def save(self):
        for key in self.output_keys:
            with open(f"/output/{key}.json", "w") as f:
                out = []
                for case_result in self._case_results:
                    out += case_result[key]
                json.dump(out, f)

if __name__ == "__main__":
    airogs_algorithm().process()