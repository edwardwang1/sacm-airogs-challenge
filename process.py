from typing import Dict

import SimpleITK
import tqdm
import json
from pathlib import Path
import tifffile
import numpy as np

import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import (resnet18_decoder,
    resnet18_encoder,
)
from pl_bolts.models.autoencoders import VAE

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

class VAE2(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    """

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE2, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
        }

        self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
        self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    
        self.train_loss = 0
        self.val_loss = 0
        self.epoch = 0
        self.images = []
        self.input_images = []
        self.val_step = 0
        self.train_losses = []
    
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def stepfadfadsf(self, batch, batch_idx):
        x = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs, 

    def training_step(self, batch, batch_idx):
        x = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        
        #self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        self.train_losses.append(recon_loss.cpu().detach().numpy())
        self.train_loss = recon_loss.cpu().detach().numpy()
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        #self.log_dict({f"val_{k}": v for k, v in logs.items()})
        self.val_loss += recon_loss.cpu().detach().numpy()
        
        imgs = x_hat.cpu().detach().numpy()
        self.images.append([Image.fromarray((image*255).astype(np.uint8).transpose(1,2,0)) for image in imgs])
        
        if self.epoch == 0 or self.epoch==1:
            self.input_images.append([Image.fromarray((image*255).astype(np.uint8).transpose(1,2,0)) for image in x.cpu().detach().numpy()])
        
        self.val_step += 1
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_epoch_end(self, outputs):
        #print(self.epoch)
        
        self.val_loss = self.val_loss / self.val_step
        
        if len(self.train_losses) > 626:
            self.train_loss = np.mean(self.train_losses[:-625])
        
        if self.epoch == 0 or self.epoch == 1:
            self.images = [item for sublist in self.images for item in sublist]
            self.input_images = [item for sublist in self.input_images for item in sublist]
            logs = {
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "epoch": self.epoch,
                "recons": [wandb.Image(image) for image in self.images[:100]],
                "originals": [wandb.Image(image) for image in self.input_images[:100]],
            }
            self.images = []
            self.input_images = []
        else:
            self.images = [item for sublist in self.images for item in sublist]
            logs = {
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "epoch": self.epoch,
                "recons": [wandb.Image(image) for image in self.images[:100]],
            }
            self.images = []
        
        save_str = "VAE_resnet18" + "_epoch+15_" + str(self.epoch) + ".pth"
        torch.save(self.state_dict(), save_str)
        shutil.copy(save_str, os.path.join(wandb.run.dir, save_str))
        wandb.save(os.path.join(wandb.run.dir, save_str))

        wandb.log(logs)
        
        self.epoch += 1
        self.val_step = 0

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 8, 3, stride=2, padding=1), #128x128x8
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), #64x64x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),#32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), #16x16x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), #8x8x128
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), #4x4x256
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), #4x4x512
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

def getODFromCroppedImage(model, image):
    #image = cv2.imread(imagePath)
    #Crop to 288 for Yolo
    im288 = image.resize((288,288))
    OD_xcent, OD_ycent, OD_width, OD_height, OD_conf = getOD(model, im288)
    #print(OD_xcent, OD_ycent, OD_width, OD_height, OD_conf)
    
    #cropping based on bounds
    x_start = OD_xcent - OD_width/2
    x_end = x_start + OD_width
    y_start = OD_ycent - OD_height/2
    y_end = y_start + OD_height

    scale = image.size[0]
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

    OD_fullsize = image.crop((x_start, y_start, x_end, y_end))
    return OD_fullsize, OD_conf

def getAutoencoderLoss(image, model, imgSize):    
    criterion = nn.MSELoss()
    transform = transforms.ToTensor()
    tensor = transform(image)
    resizeTransform = transforms.Resize([imgSize, imgSize])

    if GPU:
        tensor = resizeTransform(tensor).cuda()
    else:
        tensor = resizeTransform(tensor).cpu()
    
    if GPU:
        model.cuda()
    else:
        model.cpu()

    with torch.no_grad():
        model.eval()
        output = model(tensor[None, ...])
        loss = criterion(output, tensor[None, ...])

    return loss.cpu().detach().numpy().flatten()[0]

def getPreds(OD_fullsize, imageCroppedBounds, OD_conf, densenet, seresnext, effnet, vgg16, inception, autoencoder, vae):
    densenet_preds = []
    vgg16_preds = []
    seresnext_preds = []
    effnet_preds = []
    inception_preds = []
    autoencoder_preds = []
    vae_preds = []

    for flip in ["h", "v", "neither"]: 
        if flip == "h":
            imTransposed = OD_fullsize.transpose(Image.FLIP_TOP_BOTTOM)
            imageCroppedBounds_flipped = imageCroppedBounds.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip == "v":
            imTransposed = OD_fullsize.transpose(Image.FLIP_LEFT_RIGHT)
            imageCroppedBounds_flipped = imageCroppedBounds.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            imTransposed = OD_fullsize
            imageCroppedBounds_flipped = imageCroppedBounds
    
        seresnext_pred = predictSingle(seresnext, imTransposed, 224)
        densenet_pred = predictSingle(densenet, imTransposed, 120)
        vgg16_pred = predictSingle(vgg16, imTransposed, 120)
        effnet_pred = predictSingle(effnet, imTransposed, 224)
        inception_pred = predictSingle(inception, imTransposed, 299)

        seresnext_preds.append(seresnext_pred)
        densenet_preds.append(densenet_pred)
        vgg16_preds.append(vgg16_pred)
        effnet_preds.append(effnet_pred)
        inception_preds.append(inception_pred)
    
        autoencoder_loss = getAutoencoderLoss(imageCroppedBounds_flipped, autoencoder, 256)
        vae_loss = getAutoencoderLoss(imageCroppedBounds_flipped, vae, 224)

        autoencoder_preds.append(autoencoder_loss)
        vae_preds.append(vae_loss)
        
    #rg_likelihood = np.mean((seresnext_pred, densenet_pred, vgg16_pred, effnet_pred, inception_pred))
    
    rg_likelihood = np.mean((np.mean(seresnext_preds), 
                                np.mean(densenet_preds),
                                    np.mean(vgg16_preds),  
                                        np.mean(effnet_preds), 
                                            np.mean(inception_preds)))
    rg_binary = bool(rg_likelihood > 0.5)
    
    if rg_likelihood >= 0 and rg_likelihood <= 0.5:
        pred_scale =  rg_likelihood
    else:
        pred_scale = -(rg_likelihood - 1)
    
    ungradability_score = pred_scale * np.mean(autoencoder_preds) * np.mean(vae_preds)
    ungradability_binary = bool(pred_scale > 0.2)
    
    out = {
        "referable-glaucoma-likelihood": rg_likelihood,  # Likelihood for 'referable glaucoma'
        "referable-glaucoma-binary": rg_binary,  # True if 'referable glaucoma', False if 'no referable glaucoma'
        "ungradability-score": ungradability_score,  # True if 'ungradable', False if 'gradable'
        "ungradability-binary": ungradability_binary  # The higher the value, the more likely the label is 'ungradable'
    }
    
    return out
    
def getModels(yolo_weights, densenet_weights, seresnext_weights, effnet_weights, vgg_16_weights, inception_weights, autoencoder_weights, vae_weights):
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

    autoencoder = Autoencoder()
    if GPU:
        autoencoder.load_state_dict(torch.load(autoencoder_weights))
    else:
        autoencoder.load_state_dict(torch.load(autoencoder_weights, map_location=torch.device('cpu')))
    
    vae = VAE2(224)
    if GPU:
        vae.load_state_dict(torch.load(vae_weights))
    else:
        vae.load_state_dict(torch.load(vae_weights, map_location=torch.device('cpu')))
    
    yolo = torch.hub.load(yolo_weights, 'custom', path='yolov5_weights.pt', source="local")
    
    inceptionv3 = models.inception_v3(pretrained=False)
    inceptionv3.dropout = nn.Dropout(0.0)
    num_ftrs = inceptionv3.fc.in_features
    inceptionv3.fc = nn.Linear(num_ftrs, 1, bias=True)
    if GPU:
        inceptionv3.load_state_dict(torch.load(inception_weights)) 
    else:
        inceptionv3.load_state_dict(torch.load(inception_weights, map_location=torch.device('cpu'))) 

    return {"yolo": yolo, "densenet": densenet, "seresnext": seresnext, 
        "effnet": effnet, "vgg16": vgg16, "inception": inceptionv3,
         "autoencoder": autoencoder, "vae": vae}

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
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imageCroppedBounds = getBoundsVanilla(image)
    imageCroppedBounds = cv2.cvtColor(imageCroppedBounds, cv2.COLOR_BGR2RGB)
    imageCroppedBounds = Image.fromarray(imageCroppedBounds)
    
    OD_fullsize, OD_conf = getODFromCroppedImage(models["yolo"], imageCroppedBounds)

    out = getPreds(OD_fullsize, imageCroppedBounds, OD_conf, models["densenet"], models["seresnext"],
        models["effnet"], models["vgg16"], models["inception"], models["autoencoder"], models["vae"])
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
        print("----Submission FINAL-----")
        print("DEVICE is: " + device)
        yolo_weights = 'yolov5'
        se_resnext_weights = "se_resnext_weights.pth"
        densenet_weights = "densenet_weights.pth"
        vgg16_weights = "vgg16_weights.pth"
        effnet_weights = "effnetb7_weights.pth"
        inception_weights = "inceptionv3_weights.pth"
        autoencoder_weights = "autoencoder_weights.pth"
        vae_weights = "vae_weights.pth"
        my_models = getModels(yolo_weights, densenet_weights, se_resnext_weights, effnet_weights, vgg16_weights, inception_weights, autoencoder_weights, vae_weights)

        
        if case.path.suffix == '.tiff':
            results = []
            with tifffile.TiffFile(case.path) as stack:
                for page in tqdm.tqdm(stack.pages):
                    input_image_array = page.asarray()
                    results.append(self.predict(my_models=my_models, input_image_array=input_image_array))
        else:
            input_image = SimpleITK.ReadImage(str(case.path))
            input_image_array = SimpleITK.GetArrayFromImage(input_image)
            results = [self.predict(input_image_array=input_image_array)]
        
        results = self.combine_dicts(results)

        # Test classification output
        if not isinstance(results, dict):
            raise ValueError("Expected a dictionary as output")

        return results

    def predict(self, *, my_models: Dict, input_image_array: np.ndarray) -> Dict:
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
