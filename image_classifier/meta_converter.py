from pathlib import Path
from torchvision import models
import torchvision.transforms as T
import PIL.Image as Image
from torch import nn
import torch
import os
# import cv2
# import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def create_model(n_classes):
    model = models.resnet34(pretrained=False)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

    return model.to(device)

def create_50_model(n_classes):
    model = models.resnet50(pretrained=False)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

    return model.to(device)

def create_next101_model(n_classes):
    model = models.resnext101_32x8d(pretrained=True)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

    return model.to(device)

class MetaConverter():
    def __init__(self,datafolder):
        self.datafolder= Path(os.path.abspath(datafolder))
#         print(self.datafolder)
        
        fashion_type_classes = 18
        fashion_pattern_classes = 36
        furniture_type_classes = 10
        furniture_materials_classes = 7
        color_classes = 31
        color_furn_classes =9
        ########
        fashion_type_model = None
        fashion_pattern_model = None
        furniture_type_model = None
        furniture_materials_model = None
        color_model = None
        
        #######
        fashion_type_model = create_model(fashion_type_classes)
        fashion_type_model.load_state_dict(torch.load(str(self.datafolder /'best_model_state_fashion_type.bin')))
        fashion_type_model.eval()
        
        fashion_pattern_model = create_model(fashion_pattern_classes)
        fashion_pattern_model.load_state_dict(torch.load(str(self.datafolder /'best_model_state_fashion_pattern.bin')))
        fashion_pattern_model.eval()

        furniture_type_model = create_model(furniture_type_classes)
        furniture_type_model.load_state_dict(torch.load(str(self.datafolder /'best_model_state_furniture_type.bin')))
        furniture_type_model.eval()
        
        furniture_materials_model = create_model(furniture_materials_classes)
        furniture_materials_model.load_state_dict(torch.load(str(self.datafolder /'best_model_state_furniture_materials.bin')))
        furniture_materials_model.eval()

        color_model = create_model(color_classes)
        color_model.load_state_dict(torch.load(str(self.datafolder /'fashion_color_last.bin')))
        color_model.eval()
        
        color_furn_model = create_model(color_furn_classes)
        color_furn_model.load_state_dict(torch.load(str(self.datafolder /'furniture_color_last.bin')))
        color_furn_model.eval()

        furniture_type_folder = self.datafolder / "vision_data/furniture/type_data"
        fashion_type_folder = self.datafolder / "vision_data/fashion/type_data"
        furniture_materials_folder = self.datafolder / "vision_data/furniture/materials_data"
        fashion_pattern_folder = self.datafolder / "vision_data/fashion/pattern_data"
        color_folder = self.datafolder /"vision_data/color/color_data"
        
        furniture_type_labels = ['AreaRug', 'Bed', 'Chair', 'CoffeeTable', 'CouchChair', 'EndTable', 'Lamp', 'Shelves', 'Sofa', 'Table']
        furniture_materials_labels = ['leather', 'marble', 'memory foam', 'metal', 'natural fibers', 'wood', 'wool']
        fashion_type_labels = ['blouse', 'coat', 'dress', 'hat', 'hoodie', 'jacket', 'jeans', 'joggers', 'shirt', 'shirt, vest',
         'shoes', 'skirt', 'suit', 'sweater', 'tank top', 'trousers', 'tshirt', 'vest']
        fashion_pattern_labels = ['camouflage', 'canvas', 'cargo', 'checkered', 'checkered, plain', 'denim', 'design', 'diamonds', 'dotted',
         'floral', 'heavy stripes', 'heavy vertical stripes', 'holiday', 'horizontal stripes', 'knit', 'leafy design', 'leapard print',
         'leather', 'light spots', 'light stripes', 'light vertical stripes', 'multicolored', 'plaid', 'plain', 'plain with stripes on side',
         'radiant', 'spots', 'star design', 'streaks', 'stripes', 'text', 'twin colors', 'velvet', 'vertical design', 'vertical stripes',
         'vertical striples']
        color_labels = ['beige', 'black', 'blue', 'brown', 'dark blue', 'dark brown', 'dark green', 'dark grey', 'dark pink', 'dark red', 
                        'dark violet', 'dark yellow', 'dirty green', 'dirty grey', 'golden', 'green', 'grey', 'light blue', 'light grey', 
                        'light orange', 'light pink', 'light red', 'maroon', 'olive', 'orange', 'pink', 'purple', 'red', 'violet', 
                        'white', 'yellow']
        color_furn_labels = ['black','black and white','blue','brown','green','grey','red','white','wooden']
        
        self.image_models=None
        self.image_models={
            "furniture_type": {
                "model" : furniture_type_model,
                "folder" : furniture_type_folder,
                "classes" : furniture_type_labels
            },
            "furniture_materials": {
                "model" : furniture_materials_model,
                "folder" : furniture_materials_folder,
                "classes" : furniture_materials_labels
            },
            "fashion_type": {
                "model" : fashion_type_model,
                "folder" : fashion_type_folder,
                "classes" : fashion_type_labels
            },
            "fashion_pattern": {
                "model" : fashion_pattern_model,
                "folder" : fashion_pattern_folder,
                "classes" : fashion_pattern_labels
            },
            "color": {
                "model" : color_model,
                "folder" : color_folder,
                "classes" : color_labels
            },
            "color_furn": {
                "model" : color_furn_model,
                "classes" : color_furn_labels
            }
        }
        mean_nums = [0.485, 0.456, 0.406]
        std_nums = [0.229, 0.224, 0.225]
        
        self.transform = T.Compose([
          T.Resize(size=[256,256]),
          T.CenterCrop(size=256),
          T.ToTensor(),
          T.Normalize(mean_nums, std_nums)
        ])
        self.pp_cand_dict = {
            "grey" : ["grey","white"],
            "blue" : ["grey","brown","black","blue"],
            "purple" : ["purple","pink"],
            "brown" : ["blue"],
            "violet" : ["red","black"],
            "dark green" : ["brown","grey","dark green"],
            "dark blue" : ["brown"],
            "yellow" : ["blue"],
            "dark red" : ["purple"],
            "light grey" : ["blue"],
            "maroon" : ["violet"],
            "golden" : ["light blue"],
            "beige" : ["blue"],
            "light orange" : ["blue"],
            "dark pink" : ["purple"],
            "olive" : ["yellow"],
            "light red" : ["grey"]
        }
                
    def convert_image(self, image, data_type):
        """ data_type : 'furniture' or 'fashion'
        """
#         img = Image.open(image_path)
        img = image.convert('RGB')
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.transform(img).unsqueeze(0)
#         print(img)
        m = nn.Softmax(dim=1)
    
        output_meta = []
        if data_type == "furniture":
            type_pred,_ = self.image_models["furniture_type"]["model"](img.to(device))
            _, type_label = torch.max(type_pred, 1)
            output_meta.append(self.image_models["furniture_type"]["classes"][type_label])
            materials_pred,_ = self.image_models["furniture_materials"]["model"](img.to(device))
            _, materials_label = torch.max(materials_pred, 1)      
            output_meta.append(self.image_models["furniture_materials"]["classes"][materials_label])
            color_pred,_ = self.image_models["color_furn"]["model"](img.to(device))
            color_pred = m(color_pred)
            color_max, color_label = torch.max(color_pred, 1)
#             print(color_max.double())
#             if color_max.double() < 0.8:
#                 output_meta.append(" ".join(self.pp_cand_dict[self.image_models["color_furn"]["classes"][color_label]]))
#             else:
            output_meta.append(self.image_models["color_furn"]["classes"][color_label])
            
        else: # fashion
            type_pred,_ = self.image_models["fashion_type"]["model"](img.to(device))
            _, type_label = torch.max(type_pred, 1)
#             print(type_pred)
            output_meta.append(self.image_models["fashion_type"]["classes"][type_label])
            pattern_pred,_ = self.image_models["fashion_pattern"]["model"](img.to(device))
            _, pattern_label = torch.max(pattern_pred, 1)
            output_meta.append(self.image_models["fashion_pattern"]["classes"][pattern_label])
            
            color_pred,_ = self.image_models["color"]["model"](img.to(device))
            color_pred = m(color_pred)
            color_max, color_label = torch.max(color_pred, 1)
            

            output_meta.append(self.image_models["color"]["classes"][color_label])
            
        return output_meta

        
#     def crop_bbox(self, scene, bbox):
#         """ scene : 'm_cloth_store_1416238_woman_9_2'
#             bbox : [10,20,120,130]
#         """ 
#         # open image with scene
        
        
#         # crop using bbox information
#         try:
#             bbox = obj["bbox"]
#             x=bbox[0]
#             y=bbox[1]
#             h=bbox[2]
#             w=bbox[3]
#             if h==0 or w==0:
#                 raise Exception('bounding box size is zero!')
#             # error if image has 
#             crop_img = im[y:y+h, x:x+w]

#         except Exception as e:
#             return None
        
#         # return cropped image.
        
#         pass
    
if __name__ == "__main__":
    mc=None
    # sepecify path for data folder which has saved weights
    mc = MetaConverter("../youngjae/data")
    # sepecifiy path for image
    image_path = mc.datafolder / "samples/tshirt.jpeg"
    
    # open image
    img = Image.open(image_path)
    
    # run converter
    output = mc.convert_image(img,"fashion")
    print(output)
        
        
        
        
        
        
        
        
        
        
        
