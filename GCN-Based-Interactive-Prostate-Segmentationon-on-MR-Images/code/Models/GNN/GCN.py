import torch
import torch.nn as nn
from GCN_layer  import GraphConvolution
from GCN_res_layer import GraphResConvolution
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 feature_dim=256):

        super(GCN, self).__init__()
        self.state_dim = state_dim

        self.gcn_0 = GraphConvolution(feature_dim, 'gcn_0', out_state_dim=self.state_dim)
        self.gcn_res_1 = GraphResConvolution(self.state_dim, 'gcn_res_1')
        self.gcn_res_2 = GraphResConvolution(self.state_dim, 'gcn_res_2')
        self.gcn_res_3 = GraphResConvolution(self.state_dim, 'gcn_res_3')
        self.gcn_res_4 = GraphResConvolution(self.state_dim, 'gcn_res_4')
        self.gcn_res_5 = GraphResConvolution(self.state_dim, 'gcn_res_5')
        self.gcn_res_6 = GraphResConvolution(self.state_dim, 'gcn_res_6')
        self.gcn_7 = GraphConvolution(self.state_dim , 'gcn_7', out_state_dim=32)  # original out_state_dim: 32

        self.fc = nn.Linear(
            in_features=32,
            out_features=2,
        )

    def forward(self, input, adj):
        input1 = self.gcn_0(input, adj) # (2, 40, 128)
        input2 = self.gcn_res_1(input1, adj)
        input3 = self.gcn_res_2(input2, adj)
        input4 = self.gcn_res_3(input3, adj)
        input5 = self.gcn_res_4(input4, adj)
        input6 = self.gcn_res_5(input5, adj)
        input7 = self.gcn_res_6(input6, adj)  # (2, 40, 128)
        output = self.gcn_7(input7, adj)  # (2, 40, 32)

        # #
        # input7_0 = input7[0,:,:]
        # bs01 = output[0,:,:]  # (40, 32)
        # bs01_numpy = np.array(bs01.cpu().numpy())
        # print(bs01_numpy)
        # print("==================")
        # ## visual the GCN layer
        # fea = bs01.detach()
        # fea = (fea - fea.min()) / (fea.max() - fea.min())
        # heatmap = cv2.applyColorMap(np.uint8(255 * fea), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        # # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
        # # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
        # cam = heatmap
        # cam = cam / np.max(cam)
        # import os
        # len_th = len(os.listdir("/home/lxj/work_station/curve_gcn_release/CAM/"))
        # cv2.imwrite("/home/lxj/work_station/curve_gcn_release/CAM/" +"cam"+str(len_th)+ ".jpg", np.uint8(255 * cam))
        # return_out = self.fc(output)  # (2, 40 ,2)
        # return_out_numpy = return_out.cpu().numpy()
        # print(return_out_numpy)
        return self.fc(output)
