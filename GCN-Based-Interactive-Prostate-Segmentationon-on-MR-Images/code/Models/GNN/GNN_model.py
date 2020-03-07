import torch
import torch.nn as nn
import Utils.utils as utils
import torch.nn.functional as F
from Models.Encoder.deeplab_resnet_skip import DeepLabResnet
from first_annotation import FirstAnnotation
from GCN import GCN
import sys
sys.path.append('')
from ActiveSpline import ActiveSplineTorch
from Evaluation import losses
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNN_model(nn.Module):
    def __init__(self,
                 state_dim=256,
                 n_adj=6,
                 cnn_feature_grids=None,
                 coarse_to_fine_steps=0,
                 get_point_annotation=False
                 ):

        super(GNN_model, self).__init__()

        self.state_dim = state_dim
        self.n_adj = n_adj
        self.cnn_feature_grids = cnn_feature_grids
        self.coarse_to_fine_steps = coarse_to_fine_steps    # 3
        self.get_point_annotation = get_point_annotation

        print 'Building GNN Encoder'


        if get_point_annotation:
            nInputChannels = 4
        else:
            nInputChannels = 3



        self.encoder = DeepLabResnet(nInputChannels=nInputChannels, classifier='psp', cnn_feature_grids=self.cnn_feature_grids, edge_annotation=True)

        self.grid_size = self.encoder.feat_size # 28*28

        self.psp_feature = [self.cnn_feature_grids[-1]]  #28,  "cnn_feature_grids":[112, 56, 28, 28],


        if self.coarse_to_fine_steps > 0:  # 3 GCN model
            for step in range(self.coarse_to_fine_steps):
                if step == 0:
                    self.gnn = nn.ModuleList(
                        [GCN(state_dim=self.state_dim, feature_dim=self.encoder.final_dim + 2).to(device)]) #GCN(256, 130)
                else:
                    self.gnn.append(GCN(state_dim=self.state_dim, feature_dim=self.encoder.final_dim + 2).to(device))
        else:

            self.gnn = GCN(state_dim=self.state_dim, feature_dim=self.encoder.final_dim + 2)  #(state_dim:128, final_dim:256)

        self.interactiveGCN =[]
        for i in range(5):
            self.interactiveGCN.append(GCN(state_dim=self.state_dim, feature_dim=self.encoder.final_dim+2+2))

        self.first_annotation = FirstAnnotation(28, 512, 16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0.0, 0.00002)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.spline = ActiveSplineTorch(40, 1280, device=device, alpha=0.5)


    def forward(self, x, init_polys):
        """
        x: [16, 3, 224, 224]
        pred_polys: in scale [0,1]
        init_polys shape: [16, 40, 2]
        gt_polys: GT polys for interactiveGCN to calc distance
        """
        #instance_name = img_path[0][-17:-4]
        # print(img_path)
        # print(img_path[0][-17:-4])
        ## save the x
        # img = torch.squeeze(x)
        # img = img.cpu().numpy().transpose(1, 2, 0)  # (224, 224, 3)
        # img = (img - img.min()) / (img.max() - img.min())  # normalize to 0~1
        # import numpy as np
        # import os
        # from scipy.misc import imsave
        # img_save = np.uint8(255 * img)
        # #len_th = len(os.listdir("/home/lxj/work_station/curve_gcn_release/orig_img/"))
        # imsave("/home/lxj/work_station/curve_gcn_release/orig_img/" + str(instance_name) + '.png', img_save, 'PNG')


        image_size = x.size(2), x.size(3)
        out_dict = {}

        tic = time.time()
        # x_prob: (bs, 2, 224, 224); conv_layers: (b_s, 2048, 28, 28)
        x_prob, conv_layers = self.encoder.forward(x, return_final_res_feature=True)

        #conv_layers, psp_out = conv_layers[:-1], conv_layers[-1]
        #psp_out = conv_layers
        psp_out = conv_layers   #(bs, 512, 28, 28)

        ##edge_logits:(2, 784); vertex_logits: (2, 784)
        edge_logits, vertex_logits, logprob, _ = self.first_annotation.forward(psp_out)
        out_dict['edge_logits'] = edge_logits
        out_dict['vertex_logits'] = vertex_logits

        #del(conv_layers)

        edge_logits = edge_logits.view(
            (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1) #(2, 1, 28, 28)
        vertex_logits = vertex_logits.view(
            (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1) #(2, 1, 28, 28)
        ## F = Fc + edge_output:28x28+vertex_output:28x28
        feature_with_edges = torch.cat([psp_out, edge_logits, vertex_logits], 1)  # (2, 514, 28, 28)

        ## bellow code for visualing the cat(x_prob, edge_logits, vertex_logits)
        # fea = x_prob[:,0,:,:]
        # fea = torch.squeeze(fea).detach().cpu().numpy() #convert to (224, 224)
        # edge_logits_up = F.interpolate(edge_logits, size=224, mode='bilinear', align_corners=True)
        # vertex_logits_up = F.interpolate(vertex_logits, size=224, mode='bilinear', align_corners=True)
        # fea_cat = torch.add(edge_logits_up[0,0,:,:], vertex_logits_up[0,0,:,:]).detach().cpu().numpy()
        # #fea_cat = (edge_logits_up[0,0,:,:]).detach().cpu().numpy()
        # fea = (fea-fea.min())/(fea.max()-fea.min())
        # fea_cat = (fea_cat-fea_cat.min())/(fea_cat.max()-fea_cat.min())
        # import numpy as np
        # fea_cat = np.add(fea_cat,fea)
        # fea_cat_norm = (fea_cat-fea_cat.min())/(fea_cat.max()-fea_cat.min())
        # import cv2
        # import os
        # heatmap = cv2.applyColorMap(np.uint8(255 * fea_cat_norm), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        # # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
        # # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
        # # print(type(img))
        # cam = heatmap * 0.3 + np.float32(img) * 0.7
        # cam = cam / np.max(cam)
        # len_th = os.listdir("/home/lxj/work_station/curve_gcn_release/edge_vertex_cnn_visual/")
        # cv2.imwrite("/home/lxj/work_station/curve_gcn_release/edge_vertex_cnn_visual/edge_vertex_" + str(instance_name) + ".jpg", np.uint8(255 * cam))


        out_dict['feature_with_edges'] = feature_with_edges
        #conv_layers: [(2, 784, 256)]
        conv_layers = [self.encoder.edge_annotation_cnn(feature_with_edges)]   #edge_annotation_cnn model not use upsample


        out_dict['pred_polys'] = []
        out_dict['gcn_layer'] = []
        out_dict['x_prob'] = x_prob
        #adj = torch.zeros(x.shape[0], 40, 40) # (bs, 40, 40)
        for i in range(self.coarse_to_fine_steps):  # 3 steps
            if i == 0:
                component = utils.prepare_gcn_component(init_polys.numpy(),
                                                        self.psp_feature,
                                                        init_polys.size()[1],
                                                        n_adj=self.n_adj) # n_adj: 4, super parameter
                init_polys = init_polys.to(device)  # [16, 40, 2]


                # #bellow code for visualizing the init_polys
                # import os
                # import numpy as np
                # len_th = len(os.listdir("/home/lxj/work_station/curve_gcn_release/CAM_init2/"))
                # pred_spline_numpy = torch.squeeze(init_polys).cpu().numpy() # (cp_num,2)
                # to_axis = (pred_spline_numpy * 224).astype(np.int32)
                # CAM = np.zeros((224, 224))
                # for index, item in enumerate(to_axis):
                #     CAM[item[1] - 2, item[0] - 2] =1
                #     CAM[item[1] - 2, item[0] - 1] = 1
                #     CAM[item[1] - 1, item[0] - 2] = 1
                #     CAM[item[1]-1, item[0]-1] = 1 # top-left
                #     CAM[item[1] - 2, item[0]] = 1
                #     CAM[item[1]-1, item[0]] = 1 # top
                #     CAM[item[1] - 2, item[0] + 1] = 1
                #     CAM[item[1] - 2, item[0] + 2] = 1
                #     CAM[item[1] - 1, item[0] + 2] = 1
                #     CAM[item[1]-1, item[0]+1] = 1 # top-right
                #     CAM[item[1], item[0] - 2] = 1
                #     CAM[item[1], item[0]-1] = 1 # left
                #     CAM[item[1], item[0] + 2] = 1
                #     CAM[item[1], item[0]+1] = 1 # right
                #     CAM[item[1] + 1, item[0] - 2] = 1
                #     CAM[item[1] + 2, item[0] - 1] = 1
                #     CAM[item[1] + 2, item[0] - 2] = 1
                #     CAM[item[1]+1, item[0]-1] = 1 # bottom-left
                #     CAM[item[1] + 2, item[0]] = 1
                #     CAM[item[1]+1, item[0]] = 1 # bottom
                #     CAM[item[1] + 1, item[0] + 2] = 1
                #     CAM[item[1] + 2, item[0] + 2] = 1
                #     CAM[item[1] + 2, item[0] + 1] = 1
                #     CAM[item[1]+1, item[0]+1] = 1  #bottom-right
                #     CAM[item[1], item[0]] = 1
                # import cv2
                # heatmap = cv2.applyColorMap(np.uint8(255 * CAM), cv2.COLORMAP_JET)
                # heatmap = np.float32(heatmap) / 255
                #     # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
                #     # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
                # cam = 0.7*heatmap + np.float32(img)*0.3
                # cam = cam / np.max(cam)
                # import os
                #
                # cv2.imwrite("/home/lxj/work_station/curve_gcn_release/CAM_init2/" + str(i) + "cam" + str(len_th) + ".jpg", np.uint8(255 * cam))
                # #cv2.imwrite("/home/lxj/work_station/curve_gcn_release/CAM/" + str(i) + "gray_pre" + str(len_th) + ".jpg", np.uint8(255 * CAM))




                adjacent = component['adj_matrix'].to(device)  # [16, 40, 40]
                init_poly_idx = component['feature_indexs'].to(device)  # [16, 1, 40]

                cnn_feature = self.encoder.sampling(init_poly_idx, conv_layers)  # [bs, 40, 256]   # (bs, 784, 256) -> (bs, 40, 256)
                ## initialization
                input_feature = torch.cat((cnn_feature, init_polys), 2)  # [16, 40, 258]

            else:
                init_polys = gcn_pred_poly
                #F=cat(F(xi,yi), xi, yi) -> feature extracted
                cnn_feature = self.interpolated_sum(conv_layers, init_polys, self.psp_feature)  # [16, 40, 256]
                # input_feature: initialization
                input_feature = torch.cat((cnn_feature, init_polys), 2)  # [bs, 40, 258]

            gcn_pred = self.gnn[i].forward(input_feature, adjacent)  # [16, 40, 2]
            #adj = adjacent
            gcn_pred_poly = init_polys.to(device) + gcn_pred    # (bs, 40, 2)

            out_dict['pred_polys'].append(gcn_pred_poly)
            out_dict['gcn_layer'].append(gcn_pred)

        out_dict['adjacent'] = adjacent
        out_dict['conv_layers'] = conv_layers
        toc = time.time()
        print("cost time: ", toc-tic)
        return out_dict

    def interpolated_sum(self, cnns, coords, grids):

        X = coords[:,:,0]
        Y = coords[:,:,1]

        cnn_outs = []
        for i in range(len(grids)):
            grid = grids[i]

            Xs = X * grid
            X0 = torch.floor(Xs)
            X1 = X0 + 1

            Ys = Y * grid
            Y0 = torch.floor(Ys)
            Y1 = Y0 + 1

            w_00 = (X1 - Xs) * (Y1 - Ys)
            w_01 = (X1 - Xs) * (Ys - Y0)
            w_10 = (Xs - X0) * (Y1 - Ys)
            w_11 = (Xs - X0) * (Ys - Y0)

            X0 = torch.clamp(X0, 0, grid-1)
            X1 = torch.clamp(X1, 0, grid-1)
            Y0 = torch.clamp(Y0, 0, grid-1)
            Y1 = torch.clamp(Y1, 0, grid-1)

            N1_id = X0 + Y0 * grid
            N2_id = X0 + Y1 * grid
            N3_id = X1 + Y0 * grid
            N4_id = X1 + Y1 * grid

            M_00 = utils.gather_feature(N1_id, cnns[i])
            M_01 = utils.gather_feature(N2_id, cnns[i])
            M_10 = utils.gather_feature(N3_id, cnns[i])
            M_11 = utils.gather_feature(N4_id, cnns[i])
            cnn_out = w_00.unsqueeze(2) * M_00 + \
                      w_01.unsqueeze(2) * M_01 + \
                      w_10.unsqueeze(2) * M_10 + \
                      w_11.unsqueeze(2) * M_11

            cnn_outs.append(cnn_out)
        concat_features = torch.cat(cnn_outs, dim=2)
        return concat_features



    def reload(self, path, strict=False):
        print "Reloading full model from: ", path
        self.load_state_dict(torch.load(path)['state_dict'],
            strict=strict)


    ## bellow function code for InteractiveGCN stuff
    def worst_pre(self, gcn_pre_polys, gt_polys):
        '''
        use manhattan distance to find the worst gcn_pre_polys
        :param gcn_pre_polys:
        :param gt_polys:
        :return: the index of pre_polys with the largest manhattan distance between pre and gt
        '''
        pass