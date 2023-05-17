import argparse
import numpy as np
import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import glob
import pickle
from tqdm import tqdm
import open3d as o3d
from unet import Hand2Object_unet
# from models.cnn_3layer import conv3layer
from pyhocon import ConfigFactory
from torch.utils.data import Dataset, DataLoader


# from loss import trans_Loss
from data_preprocess import obj_load


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    
    train_x, train_y, test_x, test_y, test_right = obj_load("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans")
    learning_rate = 0.05
    epoch_num = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Hand2Object_unet()
    model.to(device)
    # criterion = nn.CosineEmbeddingLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    # class customData(Dataset):
    #     def __init__(self, x_data, y_data):
    #         self.x_data = x_data
    #         self.y_data = y_data
    #         self.len = self.x_data.shape[0]
    #     def __getitem__(self, index):
    #         return self.x_data[index], self.y_data[index]
        
    #     def __len__(self):
    #         return self.len

    # train_data = customData(train_x, train_y)
    # train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)

    # train_x = torch.tensor(np.array(train_x), dtype=torch.float32)
    # train_y = torch.tensor(np.array(train_y), dtype=torch.float32)
    # x_av = torch.mean(train_x, dim=0)
    # x_st = torch.std(train_x, dim=0)
    # y_av = torch.mean(train_y, dim=0)
    # y_st = torch.mean(train_y, dim=0)
    # train_x = (train_x - x_av)/x_st
    # train_y = (train_y - y_av)/y_st

	# Training
    model.train()
    for epoch in tqdm(range(epoch_num)):
        # batch_x = train_x.to(device)
        # batch_y = train_y.to(device)
        # optimizer.zero_grad()
        # tr_output = model(batch_x)
        # loss = criterion(batch_y, tr_output)
        # loss.backward()
        # optimizer.step()
        # if (epoch+1) % 1000 == 0:
        #     print("Epoch: %i, Loss: "%(epoch+1) + str(loss.item()))
        # for i in range(len(train_x)):
        # # for batch_x, batch_y in train_loader:
        batch_x = torch.tensor(np.array(train_x[0]), dtype=torch.float32).unsqueeze(0)
        batch_y = torch.tensor(np.array(train_y[0]), dtype=torch.float32).unsqueeze(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        tr_output = model(batch_x)
        loss = criterion(batch_y, tr_output)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1000 == 0:
            print("Epoch: %i, Loss: "%(epoch+1) + str(loss.item()))

	# Evaluation        
    model.eval()
    test_x = torch.tensor(np.array(test_x), dtype=torch.float32).unsqueeze(0)
    test_y = torch.tensor(np.array(test_y), dtype=torch.float32).unsqueeze(0)
    # test_x = (test_x - x_av)/x_st
    # test_y = (test_y - y_av)/y_st
    test_output = model(test_x.to(device))
    test_loss = criterion(test_output, test_y.to(device))
    print("Test loss: " + str(test_loss.item()))
    # loss = 0
    # for i in range(len(test_x)):
    #     testb_x = torch.tensor(np.array(test_x[i]), dtype=torch.float32).unsqueeze(0)
    #     testb_y = torch.tensor(np.array(test_y[i]), dtype=torch.float32).unsqueeze(0)
    #     testb_x = testb_x.to(device)
    #     testb_y = testb_y.to(device)
    #     test_output = model(testb_x)
    #     test_loss = criterion(test_output, testb_y)
    #     loss += test_loss.item()
    # print("Test loss: " + str(loss / len(test_x)))
    
	# Visualization
    test_output = test_output.detach().cpu().squeeze()
    # test_output = test_output * y_st + y_av
    test_output = test_output.numpy()
    
	# # Putting sliding window back to (# of frame X 18)
    # concat_window = []
    # for i in range(test_output.shape[0] + 29):
    #     concat_window.append([])
    # for i in range(test_output.shape[0]):
    #     nx_win = test_output[i]
    #     for j in range(30):
    #         concat_window[i+j].append(nx_win[j])
    # for i in range(len(concat_window)):
    #     concat_window[i] = np.array(concat_window[i]).mean(axis=0).reshape(-1)
    # concat_window = np.array(concat_window)
    
    concat_window = test_output
    final_output = {}
    for fn in range(concat_window.shape[0]):
        fn_str = str(fn)
        final_output[fn_str] = {}
        final_output[fn_str]["kettle"] = {}
        ket_feat = concat_window[fn][:9]
        ket_R = np_rot6d_to_mat(ket_feat[:6]).reshape((3,3))
        ket_trans = np.concatenate((ket_R, ket_feat[6:].reshape((3,1))), axis = 1)
        ket_trans = np.concatenate((ket_trans, np.array([0, 0, 0, 1]).reshape((1,4))))
        final_output[fn_str]["kettle"]["base"] = ket_trans 
        
        final_output[fn_str]["cup"] = {}
        cup_feat = concat_window[fn][9:]
        cup_R = np_rot6d_to_mat(cup_feat[:6]).reshape((3,3))
        cup_trans = np.concatenate((cup_R, cup_feat[6:].reshape((3,1))), axis = 1)
        cup_trans = np.concatenate((cup_trans, np.array([0, 0, 0, 1]).reshape((1,4))))
        final_output[fn_str]["cup"]["base"] = cup_trans 
        
        final_output[fn_str]["right"] = {}
        final_output[fn_str]["right"]["transform"] = test_right[fn]
        
    np.savez("/home/jeonghyeon/hoi/0511_p2_train/cup_kettle_trans/inferred_trans2.npz", **final_output)
    
    

## utility function to convert from r6d space to rotation matrix
def np_rot6d_to_mat(np_r6d):
    shape = np_r6d.shape
    np_r6d = np.reshape(np_r6d, [-1,6])
    x_raw = np_r6d[:,0:3]
    y_raw = np_r6d[:,3:6]

    x = x_raw / np.linalg.norm(x_raw, ord=2, axis=-1)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, ord=2, axis=-1)
    y = np.cross(z, x)

    x = np.reshape(x, [-1,3,1])
    y = np.reshape(y, [-1,3,1])
    z = np.reshape(z, [-1,3,1])
    np_matrix = np.concatenate([x,y,z], axis=-1)

    if len(shape) == 1:
        np_matrix = np.reshape(np_matrix, [9])
    else:
        output_shape = shape[:-1] + (9,)
        np_matrix = np.reshape(np_matrix, output_shape)

    return np_matrix


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--conf", type=str)
    # arg = parser.parse_args()
    main()