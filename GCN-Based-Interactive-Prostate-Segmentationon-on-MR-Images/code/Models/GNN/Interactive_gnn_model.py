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
from GNN_model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class interactiveGNN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 n_adj=6,
                 cnn_feature_grids=None,
                 coarse_to_fine_steps=0,
                 get_point_annotation=False
                 ):

        super(interactiveGNN, self).__init__()

        self.state_dim = state_dim
        self.n_adj = n_adj
        self.cnn_feature_grids = cnn_feature_grids
        self.coarse_to_fine_steps = coarse_to_fine_steps    # 3
        self.get_point_annotation = get_point_annotation

        if get_point_annotation:
            nInputChannels = 4
        else:
            nInputChannels = 3

        self.psp_feature = [self.cnn_feature_grids[-1]]  #28,  "cnn_feature_grids":[112, 56, 28, 28],

        self.interactive_gnn = nn.ModuleList(
            [GCN(state_dim=self.state_dim, feature_dim=260).to(device)])
        for i in range(9):
            # self.interactiveGCN.append(GCN(state_dim=self.state_dim, feature_dim=260))
            #self.interactiveGCN.append(GCN(state_dim=self.state_dim, feature_dim=258))
            self.interactive_gnn.append(GCN(state_dim=self.state_dim, feature_dim=260).to(device))

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

        ##define auto GCN
        self.autoGCN = Model(state_dim=128,
                                      n_adj=4,
                                      cnn_feature_grids=[112, 56, 28, 28],
                                      coarse_to_fine_steps=5,
                                      get_point_annotation=False,
                                      ).to(device)


    def forward(self, x, init_polys, gt_polys):
        """
        x: [16, 3, 224, 224]
        pred_polys: in scale [0,1]
        init_polys shape: [16, 40, 2]
        gt_polys: GT polys for interactiveGCN to calc distance
        """
        # img = x[0]
        # img = img.cpu().numpy().transpose(1, 2, 0)  # (224, 224, 3)
        # img = (img - img.min()) / (img.max() - img.min())  # normalize to 0~1
        # import numpy as np
        # import os
        # from scipy.misc import imsave
        # img_save = np.uint8(255 * img)
        # len_th = len(os.listdir(""))
        # imsave("" + str(len_th) + '.png', img_save, 'PNG')
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
        # #len_th = len(os.listdir(""))
        # imsave("" + str(instance_name) + '.png', img_save, 'PNG')
        out_dict_autoGCN = self.autoGCN(x, init_polys)
        # import time
        # tic = time.time()
        out_dict = out_dict_autoGCN
        conv_layers = out_dict['conv_layers']
        adjacent = out_dict['adjacent']
        del out_dict_autoGCN
        ##for 4, 5 step, do InteractiveGCN
        gt_polys = (gt_polys).to(device).type(torch.cuda.FloatTensor)   #(bs, p_num, 2)
        last_step_gcn_pred = out_dict['pred_polys'][-1].to(device) #(bs, cp_num, 2)
        #last_gcn_pred_spline = self.spline.sample_point(last_step_gcn_pred)  #(bs,p_num, 2)
        #last_gcn_pred_spline = last_step_gcn_pred
        ## test k = 3, 2, 4, 5, 6, 7, 8
        ## test (k=3) n_c: 1, 2, 4, 5, 6, 7, 8
        radius = 3  # super parameter k, means k neighbors on either side of node i will be predicted
        correct_step = 3  # the correct step to operate
        robust_facor = 8 # the radius of circle
        # is_test = True
        #
        # if is_test:
        #     for i in range(correct_step):
        #         init_polys = last_step_gcn_pred
        #         delta_x_y_cp_num = torch.zeros(gt_polys.shape[0], gt_polys.shape[1], gt_polys.shape[2]).to(device)
        #         cnn_feature = self.interpolated_sum(conv_layers, init_polys, self.psp_feature)  # [16, 40, 256]
        #         # input_feature: initialization
        #         input_feature = torch.cat((cnn_feature, init_polys, delta_x_y_cp_num),
        #                                   2)  # [16, 40, 256]+[:,:,2]+[:,:,2] -> (bs, 40, 260)
        #
        #         interacitve_gcn_pred = self.interactiveGCN[i].forward(input_feature, adjacent)
        #
        #         interactive_gcn_pred_poly = init_polys + interacitve_gcn_pred
        #         out_dict['pred_polys'].append(interactive_gcn_pred_poly)
        #         out_dict['gcn_layer'].append(interacitve_gcn_pred)
        #         last_step_gcn_pred = interactive_gcn_pred_poly
        #else:
        #totol_time = 0
        for i in range(correct_step):
            # find the worst f_i by manhattan distance
            #final_match: (bs, 1, pnum, 2) ; final_max_dis_id: (bs, 1)
            import time

            final_match, final_max_dis_id = losses.poly_match_interactive(40, last_step_gcn_pred, gt_polys, loss_type="L1")
            #idx = losses.poly_match_interactive(40, last_step_gcn_pred, gt_polys, loss_type="L1")   # idx: (bs, 1)
            #tic = time.time()

            # print("cost time match: ", toc - tic)
            ## change last_step_gcn_pred by final_max_dis_id and final_match
            gt_polys = final_match.squeeze(1).cuda()  # (bs, pnum, 2)

            ## visual last_step_gcn_pred and gt_polys
            # import numpy as np
            # pred_spline_numpy = last_step_gcn_pred[0].detach().cpu().numpy() # (cp_num,2)
            # to_axis = (pred_spline_numpy * 224).astype(np.int32)
            # CAM = np.zeros((224, 224, 3))
            #
            # gt_polys_numpy = gt_polys[0].cpu().numpy()
            # to_axis_gt = (gt_polys_numpy * 224).astype(np.int32)
            # for index, item in enumerate(to_axis):
            #     # if index>10:
            #     #     break
            #     if final_max_dis_id[0] == index:
            #     #if 0 == index:   #start point
            #         CAM[:, :, 0][item[1], item[0]] = 1  #
            #         CAM[:, :, 0][item[1], item[0]-1] = 1 #up
            #         CAM[:, :, 0][item[1], item[0]+1]=1 #down
            #         CAM[:, :, 0][item[1]-1, item[0]] = 1 #left
            #         CAM[:, :, 0][item[1]+1, item[0]] = 1 #right
            #         CAM[:, :, 2][to_axis_gt[index][1], to_axis_gt[index][0]] = 1  # red is gt
            #         CAM[:, :, 2][to_axis_gt[index][1], to_axis_gt[index][0]-1] = 1
            #         CAM[:, :, 2][to_axis_gt[index][1], to_axis_gt[index][0]+1] = 1
            #         CAM[:, :, 2][to_axis_gt[index][1]-1, to_axis_gt[index][0]] = 1
            #         CAM[:, :, 2][to_axis_gt[index][1]+1, to_axis_gt[index][0]] = 1
            #     else:
            #         CAM[:, :, 0][item[1], item[0]] = 1   #blue is pre
            #         CAM[:, :, 2][to_axis_gt[index][1], to_axis_gt[index][0]] = 1 #red is gt
            #
            # import cv2
            # #heatmap = cv2.applyColorMap(np.uint8(255 * CAM), cv2.COLORMAP_JET)
            # heatmap = np.float32(np.uint8(CAM*255)) / 255
            #     # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
            #     # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
            # cam = heatmap  #+ np.float32(img)*0.3
            # cam = cam / np.max(cam)
            # import os
            # len_th = len(os.listdir(""))
            # cv2.imwrite("" + str(len_th)+".jpg", np.uint8(255 * cam))
            # #cv2.imwrite("" + str(i) + "gray_pre" + str(len_th) + ".jpg", np.uint8(255 * CAM))



            ## correct the worst vertex and calc the delta (x, y) dimention
            delta_x_y_cp_num = torch.zeros(gt_polys.shape[0], gt_polys.shape[1], gt_polys.shape[2]).to(device)  # (bs, cp_num, 2)
            ## bellow code for robust expirement (reset gt_polys with radius prob by r: 1 2 3 4 5 6 7)

            for b in range(gt_polys.shape[0]):
                gt_polys[b, final_max_dis_id[b, 0],:] = torch.from_numpy(self.get_new_gt_vertex(gt_polys[b, final_max_dis_id[b, 0],:], robust_facor)).cuda()

            for b in range(gt_polys.shape[0]):
                #delta_x_y_cp_num[b, idx[b, 0], :] = torch.abs(gt_polys[b, idx[b, 0], :] - last_step_gcn_pred[b, idx[b, 0], :])
                delta_x_y_cp_num[b,final_max_dis_id[b,0],:] = last_step_gcn_pred[b, final_max_dis_id[b,0], :] - gt_polys[b, final_max_dis_id[b,0], :]   ## calc the delta pair
                last_step_gcn_pred[b, final_max_dis_id[b,0], :] = gt_polys[b, final_max_dis_id[b,0], :] ## correct the worst point

            init_polys = last_step_gcn_pred # (bs, 40, 2)

            ## bellow code for (bs, 40, 258)
            # F=cat(F(xi,yi), xi, yi) -> feature extracted
            cnn_feature = self.interpolated_sum(conv_layers, init_polys, self.psp_feature)  # [16, 40, 256]
            # input_feature: initialization
            # input_feature = torch.cat((delta_x_y_cp_num, init_polys, cnn_feature), 2)  # [16, 40, 256]+[:,:,2]+[:,:,2] -> (bs, 40, 260)
            input_feature = torch.cat((cnn_feature, init_polys, delta_x_y_cp_num), 2)

            interactive_gcn_pred = self.interactive_gnn[i].forward(input_feature, adjacent)  # (bs, 40, 2)
            ## set interactive_gcn_pred outside the radiou be zero
            for b in range(gt_polys.shape[0]):
                interactive_gcn_pred[b, 0:final_max_dis_id[b,0]-radius, :] = torch.zeros(1).cuda()
                interactive_gcn_pred[b, final_max_dis_id[b,0]+radius:, :] = torch.zeros(1).cuda()

            interactive_gcn_pred_poly = init_polys + interactive_gcn_pred
            # ## set others nodes that outside the dadius to be original
            # for b in range(gt_polys.shape[0]):
            #     interactive_gcn_pred_poly[b, 0: idx[b, 0] - radius, :] = init_polys[b, 0: idx[b, 0] - radius, :]
            #     interactive_gcn_pred_poly[b, idx[b, 0] + radius:, :] = init_polys[b, idx[b, 0] + radius:, :]

            out_dict['pred_polys'].append(interactive_gcn_pred_poly)
            out_dict['gcn_layer'].append(interactive_gcn_pred)
            last_step_gcn_pred=interactive_gcn_pred_poly
        #     toc = time.time()
        #     totol_time += (toc - tic)
        # print("cost time match: ", totol_time)
        # toc = time.time()
        # print("cost time: ", toc-tic)
        out_dict['adjacent'] = adjacent
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

    def get_new_gt_vertex(self, gt_vertex, robust_facor):
        '''
        bellow code all constructed with numpy
        :param gt_vertex: Size(2) -> [x, y]
        :param robust_facor: int , eg: 1,2,3,4,5,6,7
        :return: new_gt_vertex: Size(2)
        '''
        #print("before: ", (gt_vertex[0], gt_vertex[1]))
        gt_vertex = gt_vertex.cpu().numpy()
        import numpy as np
        to_axis =  (gt_vertex*224).astype(np.int32)
        rand_seed1 = np.random.rand(1)  # if > 0.5 , y = [sqrt(r^2-(x-a)^2)]+b; else: y = b - [sqrt(r^2-(x-a)^2)];
        rand_seed2 = np.random.rand(1) # x = a+-rand_seed2*r
        if rand_seed1 >= 0.5:
            y = np.sqrt(robust_facor**2-(to_axis[0]+rand_seed2*robust_facor-to_axis[0])**2) + to_axis[1]
            gt_vertex[0] = to_axis[0]+rand_seed2*robust_facor
            gt_vertex[1] = y
            #print("after: ", ((gt_vertex/224).astype(np.float32)[0], (gt_vertex/224).astype(np.float32)[1]))
            return (gt_vertex/224).astype(np.float32)
        else:
            y = to_axis[1] - np.sqrt(robust_facor ** 2 - (to_axis[0] - rand_seed2 * robust_facor - to_axis[0])**2)
            gt_vertex[0] = to_axis[0] - rand_seed2 * robust_facor
            gt_vertex[1] = y
            #print("after: ", ((gt_vertex / 224).astype(np.float32)[0], (gt_vertex / 224).astype(np.float32)[1]))
            return (gt_vertex/224).astype(np.float32)