from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
from nets.crackformer import crackformer
import os
netName = "crackformer"
valid_log_dir = "./log/" + netName
best_model_dir = "./model/" + netName + "/"
image_format = "jpg"
lable_format = "bmp"
lable_format="png"
datasetName = "CrackLS315"
# datasetName="CrackTree"
datasetName='Crack537'
valid_img_dir = "./datasets/" + datasetName + "/valid/Valid_image/"
valid_lab_dir = "./datasets/" + datasetName + "/valid/Lable_image/"
if os.path.exists(valid_img_dir)==False:
    os.makedirs(valid_img_dir)
if os.path.exists(valid_lab_dir)==False:
    os.makedirs(valid_lab_dir)
# pretrain_dir="model/crack260.pth"
pretrain_dir="model/crack315.pth"
pretrain_dir='model/crack537.pth'
valid_result_dir = "./datasets/" + datasetName + "/valid/Valid_result/"
def Test():
    crack=crackformer()
    crack.load_state_dict(torch.load(pretrain_dir))
    validator = Validator(valid_img_dir, valid_lab_dir,
                          valid_result_dir, valid_log_dir, best_model_dir, crack, image_format, lable_format)
    validator.validate('0')

if __name__ == '__main__':
    Test()