import os
import numpy as np
import torch
import argparse
import time
import torchvision.transforms as transforms
import torch.distributed as dist
import warnings
import datetime
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

from network.data_provider import datasets_factory
from network.models import model_factory
from network import trainer
from network.utils.visualization import HeatMap,Img_Integration
from network.utils.tool import make_dir


# os.environ['CUDA_VISIBLE_DEVICES']='2'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

parser = argparse.ArgumentParser(description='UAE')
parser.add_argument('--model_name', type=str, default='UAE_Unet')
parser.add_argument('--data_provider', type=str, default='Img_Img')
parser.add_argument('--mode', type=str,default="train")
parser.add_argument('--loss_function', type=str,default="MSELoss")
parser.add_argument('--device', type=str, default='cuda:0',help="If not ddp")
parser.add_argument('--cpu_worker', type=int, default=4,help="how many subprocesses to use for data loading")
parser.add_argument('--world_size', type=int,default=4)
parser.add_argument('--timestamp', type=str, default=timestamp)
#ddp configs
parser.add_argument('--is_ddp', type=int,default=0)
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--rank', type=int,default=0)
#data configs
parser.add_argument('--dataset_input', type=str,default='./data/input')
parser.add_argument('--dataset_output', type=str,default='./data/output')
parser.add_argument('--maximum_sample_size', type=int, default=5000)
parser.add_argument('--test_ratio', type=float, default=0.2,help="Divide the test data from all data")
parser.add_argument('--is_val', type=int, default=1)
parser.add_argument('--val_ratio', type=float, default=0.05,help="Divide the validation data from the training data")
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--batch_size_test', type=int, default=5)
parser.add_argument('--batch_size_val', type=int, default=5)
#train configs
parser.add_argument('--checkpoint_path', type=str,default='results/checkpoints')
parser.add_argument('--train_load_name', type=str,default=None)
parser.add_argument('--test_load_name', type=str,default='model_best')
parser.add_argument('--save_name', type=str,default='model_best')
parser.add_argument('--learn_rate', type=float,default=0.0001)
parser.add_argument('--learn_rate_patience', type=int,default=2)
parser.add_argument('--learn_rate_factor', type=float,default=0.5)
parser.add_argument('--learn_rate_min', type=float,default=1e-10)
parser.add_argument('--learn_cooldown', type=int,default=2)
parser.add_argument('--learn_step_size_up', type=int,default=20)
parser.add_argument('--learn_threshold', type=int,default=0.001)
parser.add_argument('--learn_threshold_mode', type=str,default="abs")
parser.add_argument('--max_grad_norm', type=float,default=100)
parser.add_argument('--output_dir', type=str, default='results/results_output', help='Path to save generated images')
parser.add_argument('--log_dir', type=str,default='results/log')
parser.add_argument('--epochs', type=int,default=200)
parser.add_argument('--epoch_data_num', type=int,default=1280)
parser.add_argument('--test_data_num', type=int,default=495)
parser.add_argument('--l2_weight_decay', type=float,default=0.01)
#optional
parser.add_argument('--input_mean', type=list, default=[0.01],help="[] indicates no normalization")
parser.add_argument('--input_std', type=list, default=[20],help="[] indicates no normalization")
parser.add_argument('--output_times', type=float, default=1/20)
parser.add_argument('--p_RandomHorizontalFlip', type=float, default=0.5)
parser.add_argument('--p_RandomVerticalFlip', type=float, default=0.5)
parser.add_argument('--p_RandomRotate', type=float, default=0)#Deprecated


try:
    configs = parser.parse_args()
except:
    configs = parser.parse_args(args=[])

torch.manual_seed(configs.seed)

# transform = transforms.Compose([
#     transforms.CenterCrop(200),
#     transforms.Resize((128,128))
# ])


def main(configs):
    
    if configs.is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        configs.rank=rank
        print(f"Start running basic DDP example on rank {rank}.")
        configs.device = "cuda:"+str(rank % torch.cuda.device_count())
    
    train_loader,val_loader,test_loader = datasets_factory.data_provider(configs,mode="train")
        
    input_example, target_example, data_name_example = next(iter(test_loader))
    print("Input shape:", list(input_example.shape))
    print("Output shape:", list(target_example.shape))
    configs.input_shape=list(input_example.shape)
    configs.output_shape=list(target_example.shape)

    model =model_factory.Model(configs)
    
    
    from fvcore.nn import FlopCountAnalysis

    inputs = (torch.randn(1, 1, 512, 512),)  # 你模型的输入
    
    flops = FlopCountAnalysis(model.network, inputs)
    print(flops.total())  # 输出总 FLOPs 数量（单位为次）
    print(flops.total() / 1e9, "GFLOPs")  # 转换为 GFLOPs
    
    try:
        if not configs.is_ddp:
            model.net_structure(input_size=configs.input_shape,mode="torchsummary")#torchviz or torchsummary
    except:
        pass
    
    if configs.mode=="train":
        trainer.train(configs,model,train_loader,test_loader,val_loader)
    if configs.mode=="test":   
        trainer.test(configs,model,test_loader)
        
    if configs.is_ddp:
        dist.destroy_process_group()
    time.asctime()

def Visualization(configs):
    make_dir(configs.output_dir+'/grayscale')
    maxnum=-np.inf
    minnum=np.inf
    
    for filename in os.listdir(configs.output_dir):
        if not filename.endswith('.npz'):
            continue
        filepath = os.path.join(configs.output_dir, filename)
        filename_temp=filename.rsplit('.', 1)[0]
        save_path = os.path.join(configs.output_dir,'grayscale', f'{filename_temp}.png')
        
        target = Image.open(os.path.join(configs.dataset_output, f'{filename_temp}.png')) 
        target_array = np.array(target).astype('float32')
        
        #When constructing the data, each value is increased by 120 and then multiplied by 3.
        maxnum=max(np.max(target_array)/3-120,maxnum)
        minnum=min(np.min(target_array)/3-120,minnum)
        
        if os.path.exists(save_path):
            continue

        data = np.load(filepath)
        first_key = list(data.keys())[0]
        array = data[first_key]
        array = array.squeeze(0)
        array=np.clip(array,0,255)
        
 
        array_uint8 = array.astype(np.uint8)

        img = Image.fromarray(array_uint8, mode='L')
        
        img.save(save_path)
    
    NRMSE=0
    N=0
    

    make_dir(configs.output_dir+'/grayscale_mse')
    for filename in os.listdir(configs.output_dir):
        if not filename.endswith('.npz'):
            continue
        filepath = os.path.join(configs.output_dir, filename)
        filename_temp=filename.rsplit('.', 1)[0]
        save_path = os.path.join(configs.output_dir,'grayscale_mse', f'{filename_temp}.png')
        data = np.load(filepath)
        first_key = list(data.keys())[0]
        array = data[first_key]
        array = array.squeeze(0)
        
        target = Image.open(os.path.join(configs.dataset_output, f'{filename_temp}.png')) 
        target_array = np.array(target).astype('float32')
        
        NRMSE+=np.mean((array/3-target_array/3)**2)
        N=N+1
        if os.path.exists(save_path):
            continue

        array=np.clip(255*np.abs(target_array-array)/(maxnum*3-minnum*3),0,255)
        array_uint8 = array.astype(np.uint8)

        img = Image.fromarray(array_uint8, mode='L')
        
        img.save(save_path)
        
    NRMSE=np.sqrt(NRMSE/N)/(maxnum-minnum)
    print("NRMSE=",NRMSE)
        
    color=["#2E0854", "#00008B", "#008B8B", "#228B22", "#CCCC00", "#FFA500", "#FF4500"]
    color_red = ["#FFFFFF", "#FFEFD5", "#FFC857", "#FF0000"]
    #color=["#0000FF", "#007FFF", "#00FFFF", "#7FFF00", "#FFFF00", "#FF7F00", "#FF0000"]
    #color=["#A9A9A9", "#4682B4", "#5F9EA0", "#90EE90", "#D3FFD3", "#F5F5F5", "#FFFFFF"]
    #color=["#000000", "#800000", "#FF4500", "#FFA500", "#FFFF00", "#FFFFE0", "#FFFFFF"]
    #color=["#001F3F", "#003F7F", "#007FFF", "#00BFFF", "#87CEFA", "#E0FFFF", "#FFFFFF"]
    regeneration=False
    
    HeatMap(folder=configs.dataset_input, save_folder="data/input_heatmap", regeneration=regeneration,times=1)
    HeatMap(folder=configs.dataset_output, save_folder="data/output_heatmap", regeneration=regeneration,times=1,color=color)
    Img_Integration("data/input_heatmap","data/output_heatmap","data/output_integration",alpha=0.8,regeneration=regeneration)
    
    HeatMap(folder=configs.output_dir+'/grayscale', save_folder=configs.output_dir+"/heatmap", regeneration=regeneration,times=1,color=color)
    Img_Integration("data/input_heatmap",configs.output_dir+'/heatmap',configs.output_dir+"/integration",alpha=0.8,regeneration=regeneration)
    
    HeatMap(folder=configs.output_dir+'/grayscale_mse', save_folder=configs.output_dir+"/heatmap_mse", regeneration=regeneration,times=1,color=color_red)
    Img_Integration("data/input_heatmap",configs.output_dir+'/heatmap_mse',configs.output_dir+"/integration_mse",alpha=0.8,regeneration=regeneration)
    return

   
if __name__ == "__main__":
    # configs.model_name='UAE_TransformerUnet'
    # configs.model_name='UAE_Transformer'
    # configs.model_name='UAE_Unet'
    
    # configs.mode='train'
    # main(configs)   
    
    #==================================================
    
    configs.mode='test'  
    
    configs.model_name='UAE_TransformerUnet' 
    configs.test_load_name='model_best_UAE_TransformerUnet'
    
    # configs.model_name='UAE_Transformer'
    # configs.test_load_name='model_best_UAE_Transformer'
    
    # configs.model_name='UAE_Unet'
    # configs.test_load_name='model_best_UAE_Unet'
    
    configs.output_dir="results/"+configs.model_name
    main(configs)
    Visualization(configs)
        
        
        

            

        

