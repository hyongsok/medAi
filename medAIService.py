import torch
import torchvision
import numpy as np
import RetinaCheckerPandas
from PIL import Image

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample = (256, 256)):
    # generate the class activation maps upsample to 256x256
    
    _, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = Image.fromarray(cam_img)
        if size_upsample is not None:
            cam_img = cam_img.resize(size_upsample, resample=Image.BILINEAR)
        output_cam.append(cam_img)
    return output_cam

class medAIService(object):

    def __init__(self, checkpoint, *args, **kwargs):
        self.retina_checker = None
        self.checkpoint = checkpoint
        self.transform = None
        self.model_image_size = None
        self.test_image_size_overscaling = None

        self.retina_checker = RetinaCheckerPandas.RetinaCheckerPandas()
        self.retina_checker.initialize(self.checkpoint)
        self.retina_checker.initialize_model()
        self.retina_checker.initialize_criterion()
        self.retina_checker.load_state(self.checkpoint)

        self.retina_checker.model.eval()

        self.model_image_size = self.retina_checker.image_size

        # This is the factor that the image will be scaled to before cropping the center
        # for the model. Empirically a factor between 1.0 and 1.1 yielded the best results
        # as if further reduces possible small boundaries and focuses on the center of the image
        self.test_image_size_overscaling = 1.1

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(self.model_image_size * self.test_image_size_overscaling)),
            torchvision.transforms.CenterCrop(self.model_image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.retina_checker.normalize_mean, self.retina_checker.normalize_std)
        ])

        # This is the initialization of the class activation map extraction
        
        finalconv_name = list(self.retina_checker.model._modules.keys())[-2]
        # hook the feature extractor

        self.retina_checker.model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    
    def classify_image(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError('Only PIL images supported by now')

        # Convert image to tensor
        x_input = self.transform(image)

        #Reshape for input intp 1,n,h,w
        x_input = x_input.unsqueeze(0)

        return self._classify(x_input)

    def _classify(self, x_input):
        with torch.no_grad():
            output = self.retina_checker.model(x_input.to(self.retina_checker.device))
            prediction = torch.nn.Sigmoid()(output).detach().cpu().numpy()
        
        return prediction

    
    def get_class_activation_map(self, image, single_cam=None):
        # get the softmax weight
        params = list(self.retina_checker.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())
        
        # calculating the features_blobs
        self.classify_image(image)

        if single_cam is None:
            idx = np.arange(self.retina_checker.num_classes, dtype=np.int)
        elif isinstance(single_cam, int):
            idx = [single_cam]
        elif isinstance(single_cam, tuple) or isinstance(single_cam, list):
            idx = single_cam
        else:
            raise ValueError('single_cam not recognized as None, int, or tuple')
        
        CAMs = returnCAM(features_blobs[-1], weight_softmax, idx, (self.model_image_size, self.model_image_size))
        return CAMs

        
    
    def __str__(self):
        desc = 'medAI Service:\n'
        desc += 'Loaded from {}\n'.format(self.checkpoint)
        desc += 'RetinaChecker:\n' + self.retina_checker._str_core_info()
        desc += 'Transform:\n' + str(self.transform)
        return desc

