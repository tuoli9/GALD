from torch.utils.data import Dataset
from torchvision import transforms
import glob
import json
import os 
from PIL import Image

class AdvDataset(Dataset):
    def __init__(self, adv_path):
        self.transform =  transforms.Compose([
            transforms.ToTensor(),
        ])
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        print ('Using ', len(paths))
        paths = [i.strip() for i in paths]
        self.query_paths = [i.split('.')[0]+'.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]
        
        with open('image_name_to_class_id_and_name.json', 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]
        # deal with image
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, class_id, class_name, image_name
