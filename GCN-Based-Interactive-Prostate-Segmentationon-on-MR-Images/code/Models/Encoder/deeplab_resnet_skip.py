from DeepResNet import resnet101, Res_Deeplab
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchvision.transforms as transforms
import Utils.utils as utils
from torch.autograd import Variable
from scipy.ndimage.morphology import distance_transform_cdt
from DeepResNet import PSPModule, ASPPModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepLabResnet(nn.Module):
    def __init__(self, concat_channels=64, final_dim=128, pnum=40, use_attention_sampling=False, nInputChannels=3, classifier="", n_classes=1, cnn_feature_grids=None, concat_classifier_channel=True, use_only_last_feature=False, edge_annotation=False, pixel_wise=False):
        super(DeepLabResnet, self).__init__()
        self.nInputChannels = nInputChannels
        # Default transform for all torchvision models
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.use_only_last_feature= use_only_last_feature
        self.concat_classifier_channel = concat_classifier_channel
        self.cnn_feature_grids =cnn_feature_grids
        self.concat_channels = concat_channels
        self.final_dim = final_dim
        self.feat_size = 28

        self.image_feature_dim = 256
        self.n_classes = n_classes
        self.classifier = classifier
        self.resnet = resnet101(n_classes, nInputChannels=nInputChannels, classifier=classifier)

        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)

        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)

        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.res1_concat = nn.Sequential(concat2, bn2, relu2)
        self.res1_concat_up = nn.Sequential(concat2, bn2, relu2, up2)

        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        self.res2_concat = nn.Sequential(concat3, bn3, relu3)

        self.res2_concat_up = nn.Sequential(concat3, bn3, relu3, up3)

        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        self.res4_concat = nn.Sequential(concat4, bn4, relu4)
        self.res4_concat_up = nn.Sequential(concat4, bn4, relu4, up4)


        if use_only_last_feature:
            self.layer5_concat_channels = 64 * 5
        else:
            self.layer5_concat_channels = concat_channels

        concat5 = nn.Conv2d(512, self.layer5_concat_channels, kernel_size=3, padding=1, bias=False)
        bn5 = nn.BatchNorm2d(self.layer5_concat_channels)
        relu5 = nn.ReLU(inplace=True)

        self.res5_concat = nn.Sequential(concat5, bn5, relu5)

        if edge_annotation:
            self.edge_annotation_concat_channels = 64 * 5

            edge_annotation_cnn_tunner_1 = nn.Conv2d(512 + 2, self.edge_annotation_concat_channels, kernel_size=3, padding=1, bias=False)
            edge_annotation_cnn_tunner_bn_1 = nn.BatchNorm2d(self.edge_annotation_concat_channels)
            edge_annotation_cnn_tunner_relu_1 = nn.ReLU(inplace=True)

            edge_annotation_cnn_tunner_2 = nn.Conv2d(self.edge_annotation_concat_channels, self.edge_annotation_concat_channels, kernel_size=3,
                                                     padding=1, bias=False)
            edge_annotation_cnn_tunner_bn_2 = nn.BatchNorm2d(self.edge_annotation_concat_channels)
            edge_annotation_cnn_tunner_relu_2 = nn.ReLU(inplace=True)

            self.edge_annotation_concat = nn.Sequential(edge_annotation_cnn_tunner_1,
                                                        edge_annotation_cnn_tunner_bn_1,
                                                        edge_annotation_cnn_tunner_relu_1,
                                                        edge_annotation_cnn_tunner_2,
                                                        edge_annotation_cnn_tunner_bn_2,
                                                        edge_annotation_cnn_tunner_relu_2)


        # Different from original, original used maxpool
        # Original used no activation here
        conv_final_1 = nn.Conv2d(4*concat_channels, 512, kernel_size=3, padding=1, stride=2,
            bias=False)
        bn_final_1 = nn.BatchNorm2d(512)
        relu_final_1 = nn.LeakyReLU(inplace=True)
        conv_final_2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2, bias=False)
        bn_final_2 = nn.BatchNorm2d(1024)
        relu_final_2 = nn.LeakyReLU(inplace=True)
        #conv_final_3 = nn.Conv2d(128, final_dim, kernel_size=3, padding=1, bias=False)
        conv_final_3 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, bias=False)
        #bn_final_3 = nn.BatchNorm2d(final_dim)
        bn_final_3 = nn.BatchNorm2d(2048)
        relu_final_3 = nn.LeakyReLU(inplace=True)
        self.conv_final = nn.Sequential(conv_final_1, bn_final_1, relu_final_1, conv_final_2, bn_final_2,relu_final_2,
            conv_final_3, bn_final_3, relu_final_3)

        #self.final_PSP = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1)
        self.final_PSP = ASPPModule(2048)
        ## conv for final (b_s, 512, 28, 28,)
        final_conv1_cat = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        final_conv1_cat_bn = nn.BatchNorm2d(512)
        final_conv2_cat = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        final_conv2_cat_bn = nn.BatchNorm2d(512)

        self.conv_final_cat = nn.Sequential(final_conv1_cat, final_conv1_cat_bn, final_conv2_cat, final_conv2_cat_bn)

        if self.classifier != 'psp' :
            self.final_dim = 64 * 4
        else:
            self.final_dim = 64 * 5

        self.concat_global = nn.Conv2d(concat_channels, 1, kernel_size=3, padding=1, bias=False)
        self.bn_global = nn.BatchNorm2d(1)
        self.relu_glboal = nn.ReLU(inplace=True)

        self.res_global_concat = nn.Sequential(self.concat_global, self.bn_global, self.relu_glboal)
        self.global_fc = nn.Linear(28*28, self.final_dim)

        self.attention_sampling = use_attention_sampling

        if self.attention_sampling:

            # pnum, grid_size * grid_size
            self.embed_1 = Variable(torch.from_numpy(self.create_dt_mask(pnum, 112)), requires_grad=True).to(device)
            self.embed_2 = Variable(torch.from_numpy(self.create_dt_mask(pnum, 56)), requires_grad=True).to(device)
            self.embed_3 = Variable(torch.from_numpy(self.create_dt_mask(pnum, 28)), requires_grad=True).to(device)
            self.embed_4 = Variable(torch.from_numpy(self.create_dt_mask(pnum, 28)), requires_grad=True).to(device)

        if pixel_wise:
            self.final = nn.Conv2d(512, n_classes, kernel_size=1)

        self.prob_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(64, 2, kernel_size=3, padding=1, bias=False))

    def create_dt_mask(self, pnum, grid_size):
        batch_mask =  np.zeros([pnum, grid_size* grid_size], dtype=np.float32)  # Initialize your array of zeros

        for i in range(pnum):
            mask = np.zeros([grid_size, grid_size], dtype=np.float32)  # Initialize your array of zeros

            thera = 1.0 * i / pnum * 2 * 3.1416
            x = np.cos(thera)
            y = -np.sin(thera)

            x = (0.7 * x + 1) / 2
            y = (0.7 * y + 1) / 2

            mask[ np.clip(np.floor(y * (grid_size-1)), 0, grid_size-1).astype(int), np.clip(np.floor(x * (grid_size-1)),  0, grid_size-1).astype(int)]  = 1

            dt_mask = grid_size - distance_transform_cdt(1- mask, metric='taxicab').astype(np.float32)

            dt_mask[dt_mask < grid_size-15]=0


            batch_mask[i] = dt_mask.reshape(-1)


        return batch_mask

    def create_norm_mask(self, pnum, grid_size):
        # batch_mask =  np.random.normal(0.5,0.1,[pnum, grid_size* grid_size])  # Initialize your array of zeros

        batch_mask = torch.randn(pnum, grid_size* grid_size) + 1.

        return batch_mask



    def pixel_reload(self, path, strict=False):

        self.load_state_dict(torch.load(path)['state_dict'],
            strict=strict)


    def reload(self, path):
        print("================reload start!!====================")
        model_full = Res_Deeplab(self.n_classes, pretrained=True, reload_path=path).to(device)
        print "Reloading resnet from: ", path, " nInputChannels: ", self.nInputChannels
        self.resnet.load_pretrained_ms(model_full, nInputChannels=self.nInputChannels)



    def get_global_feature(self, feature):


        batch_size = feature.size()[0]

        global_feature = self.res_global_concat(feature.view(-1, self.cnn_feature_grids[3], self.cnn_feature_grids[3], 64).permute(0, 3, 1, 2))
        global_feature = global_feature.view(batch_size, -1)

        index = torch.from_numpy(np.zeros([batch_size,2])).to(device).float()

        return torch.cat( [self.global_fc(global_feature),index], 1).unsqueeze(1)

    def edge_annotation_cnn(self, feature):

        final_feature_map = self.edge_annotation_concat(feature)

        return final_feature_map.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[-1]**2, self.edge_annotation_concat_channels)

    def forward(self, x, final=False, return_final_res_feature=False):
        x = self.normalize(x)
        # Normalization

        if self.classifier != 'psp':
            conv1_f, layer1_f, layer2_f, layer3_f, layer4_f = self.resnet(x)
        else:
            # conv1_f:(b_s, 64, 112, 112); layer1_f: (16, 256, 56, 56); layer3_f:(16, 1024, 28, 28)
            # layer4_f: (16, 2048, 28, 28); layer5_f:(16, 512, 28, 28)
            #conv1_f, layer1_f, layer2_f, layer3_f, layer4_f , layer5_f = self.resnet(x)
            conv1_f, layer1_f, layer2_f, layer3_f, layer4_f= self.resnet(x)

        # torch.Size([1, 64, 112, 112])
        if not self.use_only_last_feature:

            conv1_f_gcn = self.conv1_concat(conv1_f).permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[0]**2, 64)
            # torch.Size([1, 64, 56, 56])
            layer1_f_gcn = self.res1_concat(layer1_f).permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[1]**2, 64)
            # torch.Size([1, 64, 28, 28])
            layer2_f_gcn = self.res2_concat(layer2_f).permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[2]**2, 64)
            # torch.Size([1, 64, 28, 28])
            layer4_f_gcn = self.res4_concat(layer4_f).permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[3]**2, 64)


        if self.classifier == 'psp':
            conv1_f_up = self.conv1_concat(conv1_f)  # (b_s, 64, 112, 112)
            layer1_f_up = self.res1_concat_up(layer1_f)  # (b_s, 64, 112, 112)
            layer2_f_up = self.res2_concat_up(layer2_f)  # (b_s, 64, 112, 112)
            layer4_f_up = self.res4_concat_up(layer4_f)  # (b_s, 64, 112, 112)
            concat_features = torch.cat((conv1_f_up, layer1_f_up, layer2_f_up, layer4_f_up),
                                        dim=1)  # (b_s, 256, 112, 112)

            # # get (b_s, 128, 28, 28) for RNN

            final_features = self.conv_final(concat_features)  # (2048, 28, 28)

            final_fea = self.final_PSP(final_features)  #(b_s, 512, 28, 28)

            x_prob = F.interpolate(final_fea, size=224, mode='bilinear', align_corners=True)
            x_prob = self.prob_conv(x_prob)  # (bs, 2, 224, 224)

            return x_prob, final_fea

            # layer5_f_ = layer5_f  #(384, 28, 28)
            # final_cat = torch.cat((final_features, layer5_f), dim=1)  # (b_s, 512 28, 28)
            # final_cat_psp = self.conv_final_cat(final_cat) # (b_s, 512, 28, 28)

            return

        #     ## only handle the to (bs, 256, 112, 112)
        #     conv1_f_up = self.conv1_concat(conv1_f)  # (b_s, 64, 112, 112)
        #     layer1_f_up = self.res1_concat_up(layer1_f) # (b_s, 64, 112, 112)
        #     layer2_f_up = self.res2_concat_up(layer2_f) # (b_s, 64, 112, 112)
        #     layer4_f_up = self.res4_concat_up(layer4_f) # (b_s, 64, 112, 112)
        #     concat_features = torch.cat((conv1_f_up, layer1_f_up, layer2_f_up, layer4_f_up), dim=1)  # (b_s, 256, 112, 112)
        #
        #     # # get (b_s, 128, 28, 28) for RNN
        #
        #
        #     final_features = self.conv_final(concat_features)  # (16, 128, 28, 28)
        #     final_feat = torch.cat((layer5_f, final_features), dim=1)
        #
        #     layer5_f_gcn = self.res5_concat(layer5_f).permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[-1]**2, self.layer5_concat_channels)
        #     if self.use_only_last_feature:
        #         return [layer5_f_gcn]
        #
        #     if return_final_res_feature:
        #         #print(conv1_f_gcn.shape, layer1_f_gcn.shape, layer2_f_gcn.shape, layer4_f_gcn.shape, layer5_f_gcn.shape, layer5_f.shape)
        #         return conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn, layer5_f_gcn, layer5_f
        #
        #     else:
        #         return conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn, layer5_f_gcn
        # if final:
        #     conv1_f_up = self.conv1_concat(conv1_f)
        #     layer1_f_up = self.res1_concat_up(layer1_f)
        #     layer2_f_up = self.res2_concat_up(layer2_f)
        #     layer4_f_up = self.res4_concat_up(layer4_f)
        #
        #     concat_features = torch.cat((conv1_f_up, layer1_f_up, layer2_f_up, layer4_f_up), dim=1)
        #
        #     final_features = self.conv_final(concat_features)
        #
        #     return final_features, conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn
        #
        #
        #
        # if return_final_res_feature:
        #     return conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn, layer4_f
        #
        # return conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn

    def pixel_wise_inference(self,x):
        x = self.normalize(x)

        conv1_f, layer1_f, layer2_f, layer3_f, layer4_f, layer5_f = self.resnet(x)

        return self.final(layer5_f)

    def sampling(self, ids, features):
        cnn_out_feature = []
        for i in range(ids.size()[1]):
            id =  ids[:, i, :]
            cnn_out = utils.gather_feature(id, features[i])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)

if __name__ == '__main__':
    model = DeepLabResnet()
    model(torch.randn(2, 3, 224, 224))
