import torch
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import warnings
import matplotlib

matplotlib.use("Agg")
import skimage.io as sio
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import sys

sys.path.append('root_path')
from Utils import utils
from ActiveSpline import ActiveSplineTorch
from DataProvider import data_provider
from Models.GNN import GNN_model
from Models.GNN import Interactive_gnn_model
import timeit
from scipy.misc import imsave
from PIL import Image, ImageDraw
from skimage import draw

torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ('==> Using Devices %s' % (device))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--reload', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    return args


def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'

    dataset_val = DataProvider(split='val', opts=opts['train_val'], mode='oracle_test')

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
                            shuffle=False, num_workers=opts['train_val']['num_workers'],
                            collate_fn=data_provider.collate_fn)

    return val_loader


def override_options(opts):
    opts['mode'] = 'test'
    opts['temperature'] = 0.0
    opts['dataset']['train_val']['skip_multicomponent'] = False
    opts.pop('encoder_reload', None)
    opts['dataset']['train']['ext_points'] = opts['ext_points']
    opts['dataset']['train_val']['ext_points'] = opts['ext_points']
    opts['dataset']['train']['p_num'] = opts['p_num']
    opts['dataset']['train_val']['p_num'] = opts['p_num']
    opts['dataset']['train']['cp_num'] = opts['cp_num']
    opts['dataset']['train_val']['cp_num'] = opts['cp_num']
    opts['dataset']['train']['ext_points_pert'] = opts['ext_points_pert']
    opts['dataset']['train_val']['ext_points_pert'] = opts['ext_points_pert']

    return opts


class Tester(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.output_dir = args.output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(self.opts['exp_dir'], 'preds')
        print '==> Clean output folder'
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        utils.create_folder(self.output_dir)
        self.opts = override_options(self.opts)
        self.val_loader = get_data_loaders(self.opts['dataset'], data_provider.DataProvider)
        self.spline = ActiveSplineTorch(self.opts['cp_num'], self.opts[u'p_num'], device=device)

        # self.model = GNN_model.Model(state_dim=self.opts['state_dim'],
        #                               n_adj=self.opts['n_adj'],
        #                               cnn_feature_grids=self.opts['cnn_feature_grids'],
        #                               coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
        #                               get_point_annotation=self.opts['get_point_annotation']
        #                               ).to(device)
        self.model = Interactive_gnn_model.interactiveGNN(state_dim=self.opts['state_dim'],
                                                         n_adj=self.opts['n_adj'],
                                                         cnn_feature_grids=self.opts['cnn_feature_grids'],
                                                         coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
                                                         get_point_annotation=self.opts['get_point_annotation'],
                                                         ).to(device)

        print '==> Reloading Models'
        self.model.reload(args.reload, strict=False)

    def process_outputs(self, data, output, save=True):
        """
        Process outputs to get final outputs for the whole image
        Optionally saves the outputs to a folder for evaluation
        """
        instances = data['instance']

        pred_spline = output['pred_polys']  # (16, 40, 2)
        pred_gcn = output['gcn_layer']  # (16, 40, 2)

        ## overlay pred_spline to 224*224 image

        # origin_img = output['origin_img']
        # img = torch.squeeze(origin_img)
        # img = data['img']
        # img = torch.squeeze(img)
        # img = img.cpu().numpy().transpose(1, 2, 0)  # (224, 224, 3)
        # img = (img-img.min())/(img.max()-img.min()) # normalize to 0~1
        # img_save = np.uint8(255 * img)
        # import os
        # # len_th = len(os.listdir(""))
        # imsave("" + str(instances[0]['img_path'][-17:-4]) + '.png', img_save, 'PNG')
        # len_th = len(os.listdir(""))
        #
        # instance_id = instances[0]['img_path'][-17:-4]
        #
        # for i in range(len(output['pred_polys_all'])):
        #     pred_spline_numpy = torch.squeeze(output['pred_polys_all'][i]).cpu().numpy() # (cp_num,2)
        #     to_axis = (pred_spline_numpy * 224).astype(np.int32)
        #     CAM = np.zeros((224, 224))
        #     for index, item in enumerate(to_axis):
        #         CAM[item[1] - 2, item[0] - 2] = 1
        #         CAM[item[1] - 2, item[0] - 1] = 1
        #         CAM[item[1] - 1, item[0] - 2] = 1
        #         CAM[item[1] - 1, item[0] - 1] = 1  # top-left
        #         CAM[item[1] - 2, item[0]] = 1
        #         CAM[item[1] - 1, item[0]] = 1  # top
        #         CAM[item[1] - 2, item[0] + 1] = 1
        #         CAM[item[1] - 2, item[0] + 2] = 1
        #         CAM[item[1] - 1, item[0] + 2] = 1
        #         CAM[item[1] - 1, item[0] + 1] = 1  # top-right
        #         CAM[item[1], item[0] - 2] = 1
        #         CAM[item[1], item[0] - 1] = 1  # left
        #         CAM[item[1], item[0] + 2] = 1
        #         CAM[item[1], item[0] + 1] = 1  # right
        #         CAM[item[1] + 1, item[0] - 2] = 1
        #         CAM[item[1] + 2, item[0] - 1] = 1
        #         CAM[item[1] + 2, item[0] - 2] = 1
        #         CAM[item[1] + 1, item[0] - 1] = 1  # bottom-left
        #         CAM[item[1] + 2, item[0]] = 1
        #         CAM[item[1] + 1, item[0]] = 1  # bottom
        #         CAM[item[1] + 1, item[0] + 2] = 1
        #         CAM[item[1] + 2, item[0] + 2] = 1
        #         CAM[item[1] + 2, item[0] + 1] = 1
        #         CAM[item[1] + 1, item[0] + 1] = 1
        #         CAM[item[1], item[0]] = 1
        #     import cv2
        #     heatmap = cv2.applyColorMap(np.uint8(255 * CAM), cv2.COLORMAP_JET)
        #     heatmap = np.float32(heatmap) / 255
        #     # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
        #     # cam = heatmap + np.float32(torch.squeeze(img).permute(1,2,0).detach().numpy())
        #     cam = 0.7*heatmap + np.float32(img) * 0.4   #0.5 should be smaller
        #     cam = cam / np.max(cam)
        #     import os
        #
        #     cv2.imwrite("" + str(instance_id) + "_" + str(i) + ".jpg", np.uint8(255 * cam))
        #     #cv2.imwrite("" + str(i) + "gray_pre" + str(len_th) + ".jpg", np.uint8(255 * CAM))

        # preds = self.spline.sample_point(pred_spline) # (16, 1280, 2)
        preds = pred_spline
        preds_numpy = preds.cpu().numpy()
        torch.cuda.synchronize()
        preds = preds.cpu().numpy()

        pred_spline = pred_spline.cpu()
        pred_spline = pred_spline.numpy()

        ## overlay pre_polygons on croped img
        # img_save = np.uint8(img)
        # pil_img = Image.fromarray(img_save, mode='RGB')
        # img_draw = ImageDraw.Draw(pil_img, 'RGBA')
        # # img_draw.polygon(polys_pred, outline='red') #pred polygons
        # # img_draw.polygon(polys_gt, outline='blue')  #GT polygons
        # predicted_poly = []
        # poly = preds[0]
        # poly = poly * 224#data['patch_w'][0] #to (224, 224)
        # # poly[:, 0] += data['starting_point'][0][0]
        # # poly[:, 1] += data['starting_point'][0][1]
        # predicted_poly.append(poly.tolist())
        # polys_pre = predicted_poly[0]
        # polys_pre_final = [tuple(item) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0]-data['starting_point'][0][0])*224/data['patch_w'][0],(item[1]-data['starting_point'][0][1])*224/data['patch_w'][0])) for item in polys_gt]
        #
        # # draw circle
        # #img_draw = ImageDraw.Draw(img_ob, 'RGBA')
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        #
        # ## width be bigger
        # polys_pre_final = [tuple((item[0]-1, item[1]-1)) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0]-1,
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0]-1)) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        #
        # polys_pre_final = [tuple((item[0] + 1, item[1] + 1)) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0] + 1,
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0] + 1)) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        # ##top
        # polys_pre_final = [tuple((item[0], item[1]- 1)) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0],
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0] - 1)) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        # ##bottom
        # polys_pre_final = [tuple((item[0], item[1]+ 1)) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0],
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0] + 1)) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        # ## top-right
        # polys_pre_final = [tuple((item[0]+1, item[1] - 1)) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0]+1,
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0] - 1)) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        # ## bottom-left
        # polys_pre_final = [tuple((item[0] - 1, item[1] + 1)) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0] - 1,
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0] + 1)) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        # ## left
        # polys_pre_final = [tuple((item[0] - 1, item[1])) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0] - 1,
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0])) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons
        # #img_draw.polygon(polys_gt_final, outline=(0, 0, 255))
        # ## right
        # polys_pre_final = [tuple((item[0] + 1, item[1])) for item in polys_pre]
        # polys_gt = instances[0]['components'][0]['poly']
        # polys_gt_final = [tuple(((item[0] - data['starting_point'][0][0]) * 224 / data['patch_w'][0] + 1,
        #                          (item[1] - data['starting_point'][0][1]) * 224 / data['patch_w'][0])) for item in
        #                   polys_gt]
        # img_draw.polygon(polys_pre_final, outline=(255, 0, 0))  # pred polygons

        ## bellow code for draw vertexPoly on croped img
        # img_draw.polygon(polys_pre_final, outline=(255, 193, 37))
        # imsave_ob = np.array(pil_img).astype(np.uint8)
        # for index, item in enumerate(polys_pre_final):
        #     if index % 32 == 0:
        #         #cv2.circle(imsave_ob, (int(item[0]),int(item[1])),2,(0,255,255),-1)
        #         rr,cc = draw.circle(item[1],item[0],2)
        #         draw.set_color(imsave_ob, [rr,cc], [0,255,255])
        # pil_img = Image.fromarray(imsave_ob, mode='RGB')
        # pil_img.save(''+str(instances[0]['img_path'][-17:-4])+'.png', 'PNG')

        # img_ob.save(imgpath[:-56] + '' + imgpath[-17:-4] + '_overlay_pre_gt.png','PNG')
        # pil_img.save(imgpath[:-56] + "es" + '/pre_overlay/' + imgpath[-29:-4] + '_overlay_pre_gt.png','PNG')

        polys = []
        for i, instance in enumerate(instances):
            poly = preds[i]
            poly = poly * data['patch_w'][i]  # to (224 224)
            poly[:, 0] += data['starting_point'][i][0]
            poly[:, 1] += data['starting_point'][i][1]

            pred_sp = pred_spline[i]
            pred_sp = pred_sp * data['patch_w'][i]
            pred_sp[:, 0] += data['starting_point'][i][0]
            pred_sp[:, 1] += data['starting_point'][i][1]

            instance['spline_pos'] = pred_sp.tolist()

            polys.append(poly)

            if save:
                img_h, img_w = instance['img_height'], instance['img_width']
                predicted_poly = []

                pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                utils.draw_poly(pred_mask, poly.astype(np.int))
                predicted_poly.append(poly.tolist())

                ## overlap pred_poly with original image
                img_path = str(instance['img_path'])
                # get gt_polys
                # polys_gt = data['instance'][0]['components'][0]['poly']
                polys_gt = instance['components'][0]['poly']
                # get polys_pre
                polys_pre = predicted_poly[0]
                polys_gt_final = [tuple(item) for item in polys_gt]
                polys_pre_final = [tuple(item) for item in polys_pre]
                if len(polys_pre_final) == 1:
                    print('error!! ', img_path[10:])
                    print('before optimize is: ', output['pred_polys'][i])
                ## <<overlay gt_pre>>
                else:
                    pass
                    ##for 30 test volumes
                    #utils.overlap_pre_gt('/home/lxj/work_station/' + img_path[10:], polys_pre_final, polys_gt_final)
                    ## for 5-cross-val experiments
                    #utils.overlap_pre_gt('/home/lxj/work_station/' + img_path[25:], polys_pre_final, polys_gt_final)
                    # utils.overlap_pre_gt('/home/lxj/' + img_path[10:], polys_pre_final, polys_gt_final) # only for citiscapse

                gt_mask = utils.get_full_mask_from_instance(
                    self.opts['dataset']['train_val']['min_area'],
                    instance)

                instance['my_predicted_poly'] = predicted_poly
                instance_id = instance['img_path'][-17:-4]
                # image_id = instance['image_id']

                pred_mask_fname = os.path.join(self.output_dir, '{}_pred.png'.format(instance_id))
                instance['pred_mask_fname'] = os.path.relpath(pred_mask_fname, self.output_dir)

                gt_mask_fname = os.path.join(self.output_dir, '{}_gt.png'.format(instance_id))
                instance['gt_mask_fname'] = os.path.relpath(gt_mask_fname, self.output_dir)

                instance['n_corrections'] = 0

                info_fname = os.path.join(self.output_dir, '{}_info.json'.format(instance_id))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sio.imsave(pred_mask_fname, pred_mask)
                    sio.imsave(gt_mask_fname, gt_mask)

                # print '==> dumping json'
                with open(info_fname, 'w') as f:
                    json.dump(instance, f, indent=2)

        return polys

    def test(self):
        print 'Starting testing'
        self.model.eval()

        # Leave LSTM in train mode
        times = []
        count = 0
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                # Forward pass

                if self.opts['get_point_annotation']:
                    img = data['img'].to(device)
                    annotation = data['annotation_prior'].to(device).unsqueeze(1)
                    img = torch.cat([img, annotation], 1)
                else:
                    img = data['img'].to(device)

                start = timeit.default_timer()
                test_gt_data = data['gt_poly'][0, 0:40, :]
                output = self.model(img, data['fwd_poly'], data['sampled_interactive'])  # , img_path=data['img_path']
                stop = timeit.default_timer()
                if count > 0:
                    times.append(stop - start)

                if self.opts['coarse_to_fine_steps'] > 0:
                    output['pred_polys_all'] = output['pred_polys']
                    output['pred_polys'] = output['pred_polys'][-1]
                    output['gcn_layer'] = output['gcn_layer'][-1]
                # Bring everything to cpu/numpy
                for k in output.keys():
                    if k == 'pred_polys': continue
                    if k == 'edge_logits': continue
                    if k == 'vertex_logits': continue
                    if k == 'gcn_layer': continue
                    if k == 'pred_polys_all': continue
                    # output[k] = output[k].cpu().numpy()
                output["origin_img"] = img  # cuda data
                self.process_outputs(data, output, save=True)
                del (output)
                if count > 0:
                    print sum(times) / float(len(times))
                count = count + 1


if __name__ == '__main__':
    args = get_args()
    tester = Tester(args)
    tester.test()
