#!/usr/bin/env python
# coding:utf-8
"""
@version: 
@author: xiaojianli
@contact:
@software: PyCharm
@time:  上午11:06
"""
import os, shutil
from PIL import Image, ImageDraw
from scipy import misc, ndimage
import numpy as np
import json
import math
import calcArea
from calcArea import GetAreaOfPolyGon, Point

class utils:
    def __init__(self):
        self.file_names = ''
        self.path_from = ''
        self.path_to = ''
        self.target_height = 512
        self.target_width = 512

    def copy_img(self,file_names, path_from, path_to):
        files = os.listdir(file_names)
        #files_prostate = os.listdir('')
        for file in files:
#            elem =255
#            img_ = misc.imread(path_from+file)
            #if file not in files_prostate:
            #do copy operation
            img_ = misc.imread(path_from+file)
                #img_shape = img_.shape
            misc.imsave(path_to+file, img_)
            #img = Image.open(path_from+file)
            # img = Image.open(path_from+('DL_Image'+str(file[8:])))
            # misc.imsave(path_to+'DL_Image'+str(file[8:]), img)

    def copy_file(self,dic_file, path_from, path_to):
        for item in dic_file:
            files = os.listdir(path_from+str(item)+'/')
            for file in files:
                shutil.copyfile(path_from+str(item)+'/'+file, path_to+file)

    #used for rename the name of image
    def rename(self, path_img):
        files = os.listdir(path_img)
        for file in files:
            img = misc.imread(path_img+file)
            misc.imsave(path_img+file[:-9]+'.png', img)
            os.remove(path_img+file)

    #used for produce GT
    def makeGT(self, img_path, to_path):
        files = os.listdir(img_path)
        for file in files:
            print('file is: ', file)
            img = misc.imread(img_path+file)
            img2array = np.array(img)
            #print('img is ', img[:,:,0])
            for index_x in range(img2array.shape[0]-1):
                x_flag = False
                x_point= [0]
                for index_y in range(img2array.shape[1]-1):

                    if x_flag == True and img[index_x, index_y, 0] == 255:
                        print((x_point[0], index_y))
                        print(img[index_x, x_point[0]:index_y, 0])
                        # change the pixel value to 255
                        img[index_x, x_point[0]:index_y, 0] = 255
                        img[index_x, x_point[0]:index_y, 1] = 255
                        img[index_x, x_point[0]:index_y, 2] = 255
                        print('changed')
                        print(img[index_x, x_point[0]:index_y, 0])
                        x_flag = False
                    else:

                        if img[index_x, index_y, 0] == 255 and img[index_x, index_y+1, 0] == 0:
                            x_flag = True
                            print('index_x is: ', index_x)
                            print('------------> True')
                            x_point[0] = index_y

            misc.imsave(to_path+file, img)

    def make_vedio_input(self, dic_directory, img_path):
        ## for json annotation
        # for item in dic_directory:
        #     os.mkdir(self.path_to+str(item))
        files = os.listdir(img_path)
        for file in files:
            # to_file = int(file[4:8]) #for 50 volumes
            to_file = int(file[8:12])
            if to_file in dic_directory:
                img = misc.imread(img_path+file)
                #misc.imsave(self.path_to+str(to_file)+'/'+file[:-4]+'.png', img)
                misc.imsave(self.path_to+'/'+file[:-4]+'.png', img)
            # for json annotation
            #     from_file = img_path+file
            #     shutil.copyfile(from_file, self.path_to+str(to_file)+'/'+file[:-4]+'json')

    ## this func used for getting the index of mid, base, apex prostate
    def getNumL_NumR(self, img_path):
        numF = []
        lenF = []
        numL = []
        numR = []
        files =  sorted(int(item) for item in os.listdir(img_path))
        numF.append(files)
        for file in files:
            flag_start = False
            images = os.listdir(img_path+'/'+str(file))

            images_temp = []
            images_final = []
            for item in images:
                images_temp.append(int(item[11:13]))

            images_temp = sorted(images_temp)

            for i in range(len(images)):
                if len(str(images_temp[i])) == 2:
                    images_final.append(images[i][:11]+str(images_temp[i])+images[i][13:])
                else:
                    images_final.append(images[i][:11] +'0'+str(images_temp[i]) + images[i][13:])
            start = 0
            lenF.append(len(images))
            for index, image in enumerate(images_final):
                img = Image.open(img_path+'/'+str(file)+'/'+image)
                img2array = np.array(img)
                if 255 in img2array[:,:,0] and flag_start==False:
                    flag_start = True
                    start = index
                if flag_start==True and 255 not in img2array[:,:,0]:
                        numL.append(start+1)
                        numR.append(index)
                        break
                else:
                    if index == len(images)-1:
                        numL.append(start+1)
                        numR.append(index+1)
                        break
        print(numF)
        print(lenF)
        print(numL)
        print(numR)
    # draw the polygon which comes from the method OXR on the original picture
    def test_polygon(self, img_path, label_path):
        images = os.listdir(img_path)
        for img in images:
            vertices2 = []
            image = Image.open(img_path+img)
            json_object_label = json.load(open(label_path+img[:-4]+'.json'))    #load the json format annotation
            objects = json_object_label['object']
            for obj in objects:
                for points in obj['polygon']:
                    vertex = (points[0], points[1])
                    vertices2.append(vertex)
                img_draw = ImageDraw.Draw(image, 'RGBA')
                img_draw.polygon(vertices2, outline='blue')
                image.save('/media/jacoob/My Passport/LXJ/scholar_data/PROMISE11'
                           '2_SUBMITION_SECOND/TRAIN_DATA/image_with_boundary/'+img[:-4]+'_gt.png', 'PNG')



    def test_polygon_CityScape(self, img_path, label_path):
        images = os.listdir(img_path)
        for img in images:
            areas = []
            image = Image.open(img_path + img)
            #json_object_label = json.load(open(label_path + img[:20] + '.json'))  # load the json format annotation of CityScape
            json_object_label = json.load(open(label_path + img[:-4] + '.json'))
            print(img[:-4])
            objects = json_object_label[0]["components"]
            if len(objects) == 0:
                print('pass')
                pass
            else:
                bbox_x_y = json_object_label[0]["bbox"]
                vertex_bbox = [(bbox_x_y[0], bbox_x_y[1]), (bbox_x_y[0]+bbox_x_y[2], bbox_x_y[1]+bbox_x_y[3])]
                for obj in objects:
                    vertices_bbox = []
                    vertices2 = []
                    for points in obj["poly"]:
                        vertex = (points[0], points[1])
                        vertices2.append(vertex)
                    img_draw = ImageDraw.Draw(image, 'RGBA')
                    img_draw.polygon(vertices2, outline='blue')

                #image.save(img_path + img[:-4] + '_gt.png', 'PNG')
                #for point in obj["bbox"]:
                #x0, y0, w, h = boj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]
                    vertices_bbox.append((obj["bbox"][0], obj["bbox"][1]))
                    vertices_bbox.append((obj["bbox"][0]+obj["bbox"][2], obj["bbox"][1]+obj["bbox"][3]))
                    img_draw = ImageDraw.Draw(image, 'RGBA')
                    img_draw.rectangle(vertices_bbox, outline='red')
                ##calc Area
                    point_x_y =[]
                    for i in range(len(vertices2)):
                        point_x_y.append(Point(vertices2[i][0], vertices2[i][1]))
                    areas.append(GetAreaOfPolyGon(point_x_y))
                img_draw = ImageDraw.Draw(image, 'RGBA')
                img_draw.rectangle(vertex_bbox, outline='yellow')
                image.save(img_path+img[:-4]+'_gt_bbox_png.png', 'PNG')
                print(img)
                print(areas)





    ##process the annotation file without prostate
    # each subfile in
    # find the 'left_start_prostate' and 'right_start_prostate' annotation file
    # make all the files before 'left_start_prostate' be 'left_start_prostate'; the right, vice versa
    def process_no_prostate(self, dic_directory):
        left_right_prostate = {}
        for item in dic_directory:
            files = os.listdir(self.file_names+str(item)+'/')
            images_temp = []
            images_final = []
            for item_file in files:
                images_temp.append(int(item_file[11:13]))

            images_temp = sorted(images_temp)

            for i in range(len(files)):
                if len(str(images_temp[i])) == 2:
                    images_final.append(files[i][:11] + str(images_temp[i]) + files[i][13:])
                else:
                    images_final.append(files[i][:11] + '0' + str(images_temp[i]) + files[i][13:])
            #files=images_final

            print(images_final)
            left_start_prostate=""
            right_start_prosate=""
            for file in images_final:
                json_content = json.load(open(self.file_names+str(item)+'/'+file))
                # train image size should be 512*512
                # test image size should be 320*320
                if len(json_content['object']) == 0:
                    pass
                else:
                    left_start_prostate = file
                    break
            for index in range(len(images_final)-1, -1, -1):
                json_content = json.load(open(self.file_names+str(item)+'/'+images_final[index]))
                if len(json_content['object']) == 0:
                    pass
                else:
                    right_start_prosate = images_final[index]
                    break
            # handle the annotation without prostate
            if left_start_prostate != "" and right_start_prosate != "":
                left_right_prostate[item]=[left_right_prostate, right_start_prosate]
                for file in images_final:
                    if str(file)!=left_start_prostate:
                        os.remove(self.file_names+str(item)+'/'+file)
                        shutil.copyfile(self.file_names+str(item)+'/'+left_start_prostate, self.file_names+str(item)+'/'+file)
                    else:
                        break
                for index in range(len(images_final)-1, 0, -1):
                    if images_final[index]!=right_start_prosate:
                        os.remove(self.file_names+str(item)+'/'+images_final[index])
                        shutil.copyfile(self.file_names+str(item)+'/'+right_start_prosate, self.file_names+str(item)+'/'+images_final[index])
                    else:
                        break
            #return left_right_prostate
    # def clean_label_without_prostate(self, dict_file,left_right_prosate, annotation_pah):
    #     for item in dict_file:
    #         path = annotation_pah+str(item)+'/'
    #         files = os.listdir(path)
    #         for file in files:
    #             content = json.load(open(path+file))
    #             objects = content['object']
    #             poly = objects[0]['polygon']
    #             ## calc bbox
    #             min_col = np.min(np.array(poly), axis=0)    #min_col[0], min_col[1]=min_x, min_y
    #             max_col = np.max(np.array(poly), axis=0)    #max_col[0], max_col[1]=max_x, max_y
    #             object_w = max_col[0]-min_col[0]
    #             object_h = max_col[1]-min_col[1]
    #             x0, y0 = min_col[0], min_col[1]
    #             ## calc area of objxt
    #             bbox = [x0, y0, object_w, object_h]
    #
    #             vertices2 = []
    #             for points in poly:
    #                 vertex = (points[0], points[1])
    #                 vertices2.append(vertex)
    #             print(vertices2)
    #             point_x_y = []
    #             for i in range(len(vertices2)):
    #                 point_x_y.append(Point(vertices2[i][0], vertices2[i][1]))
    #             print(file)
    #             area = round(GetAreaOfPolyGon(point_x_y), 1)
    #
    #             image_path = img_path+file[:-4]+'png'
    #             key_vlaue = [{
    #                 "img_path":image_path,
    #                 "img_width": content['imgHeight'],
    #                 "img_height": content['imgWidth'],
    #                 "label": objects[0]['label'],
    #                 "split": split,
    #                 "components": [
    #                 {
    #                     "bbox": bbox,
    #                    "poly": poly,
    #                     "area": area
    #                 }
    #                 ]
    #             }
    #             ]
    #             os.remove(path+file)
    #             with open(path+file, 'w') as file_object:
    #                 json.dump(key_vlaue, file_object, indent=4)  # indent=4 make perfect print
    #     return
    #

    # this function just same as the 'def handle_polyRNNpp_annotation_val()' function
    def handle_polyRNNpp_annotation(self, dict_file, annotation_path, img_path, split):
        """
        :param dict_file: all parent-directory name of json annotation file, [dict type]
        :param annotation_pah: the path to dict_file param , [str type]
        :param img_path: path to image
        :param split : should be "train" , "train_val" or "val"
        :return: the result json format annotation file should't change the image size
        :note  "bbox" : x0, y0, w, h
               "area" : area of annotation object
               this script handle all image annotation without changing the size of origin image (w*h)
        """
        for item in dict_file:
            path = annotation_path+str(item)+'/'
            files = os.listdir(path)
            for file in files:
                #image_path = img_path + file[:-4] + 'png'
                image_path = img_path + 'DL_Image'+file[8:-4]+'png'
                content = json.load(open(path+file))
                objects = content['object']
                ##handle label without prostate
                if len(objects) == 0:
                   key_vlaue = [{
                       "img_path": image_path,
                       "img_width": content['imgHeight'],
                       "img_height": content['imgWidth'],
                       "label":"prostate",
                       "split": split,
                       "bbox":[],
                       "components": []
                   }]
                   os.remove(path + file)
                   with open(path + file, 'w') as file_object:
                       json.dump(key_vlaue, file_object, indent=4)  # indent=4 make perfect print
                else:
                     poly = objects[0]['polygon']
                     ## calc bbox
                     min_col = np.min(np.array(poly), axis=0)    #min_col[0], min_col[1]=min_x, min_y
                     max_col = np.max(np.array(poly), axis=0)    #max_col[0], max_col[1]=max_x, max_y
                     object_w = max_col[0]-min_col[0]
                     object_h = max_col[1]-min_col[1]
                     x0, y0 = min_col[0], min_col[1]
                     ## calc area of objxt
                     bbox = [x0, y0, object_w, object_h]

                     vertices2 = []
                     for points in poly:
                         vertex = (points[0], points[1])
                         vertices2.append(vertex)
                     print(vertices2)
                     point_x_y = []
                     for i in range(len(vertices2)):
                         point_x_y.append(Point(vertices2[i][0], vertices2[i][1]))
                     print(file)
                     area = round(GetAreaOfPolyGon(point_x_y), 1)
                     key_vlaue = [{
                         "img_path":image_path,
                         "img_width": content['imgHeight'],
                         "img_height": content['imgWidth'],
                         "label": objects[0]['label'],
                         "split": split,
                         "bbox": bbox,
                         "components": [
                         {
                             "bbox": bbox,
                             "poly": poly,
                             "area": area
                         }
                         ]
                     }
                     ]
                     os.remove(path+file)
                     with open(path+file, 'w') as file_object:
                         json.dump(key_vlaue, file_object, indent=4)  # indent=4 make perfect print
        return


    def handle_polyRNNpp_annotation_val(self, dict_file, annotation_pah, img_path, split):
        """
        :param dict_file: all parent-directory name of json annotation file, [dict type]
        :param annotation_pah: the path to dict_file param , [str type]
        :param img_path: path to image
        :param split : should be "train" , "train_val" or "val"
        :return:
        :note  "bbox" : x0, y0, w, h
               "area" : area of annotation object
               this script handle all test image annotation (w*h)
        """
        for item in dict_file:
            path = annotation_pah+str(item)+'/'
            files = os.listdir(path)
            for file in files:
                image_path = img_path + file[:-4] + 'png'
                content = json.load(open(path+file))
                objects = content['object']
                ##handle label without prostate
                if len(objects) == 0:
                   key_vlaue = [{
                       "img_path": image_path,
                       "img_width": content['imgHeight'],
                       "img_height": content['imgHeight'],
                       "label":"prostate",
                       "split": split,
                       "bbox":[],
                       "components": []
                   }]
                   os.remove(path + file)
                   with open(path + file, 'w') as file_object:
                       json.dump(key_vlaue, file_object, indent=4)  # indent=4 make perfect print
                else:
                     poly = objects[0]['polygon']
                     ## calc bbox
                     min_col = np.min(np.array(poly), axis=0)    #min_col[0], min_col[1]=min_x, min_y
                     max_col = np.max(np.array(poly), axis=0)    #max_col[0], max_col[1]=max_x, max_y
                     object_w = max_col[0]-min_col[0]
                     object_h = max_col[1]-min_col[1]
                     x0, y0 = min_col[0], min_col[1]
                     ## calc area of objxt
                     bbox = [x0, y0, object_w, object_h]

                     vertices2 = []
                     for points in poly:
                         vertex = (points[0], points[1])
                         vertices2.append(vertex)
                     print(vertices2)
                     point_x_y = []
                     for i in range(len(vertices2)):
                         point_x_y.append(Point(vertices2[i][0], vertices2[i][1]))
                     print(file)
                     area = round(GetAreaOfPolyGon(point_x_y), 1)
                     key_vlaue = [{
                         "img_path":image_path,
                         "img_width": content['imgHeight'],
                         "img_height": content['imgWidth'],
                         "label": objects[0]['label'],
                         "split": split,
                         "bbox": bbox,
                         "components": [
                         {
                             "bbox": bbox,
                             "poly": poly,
                             "area": area
                         }
                         ]
                     }
                     ]
                     os.remove(path+file)
                     with open(path+file, 'w') as file_object:
                         json.dump(key_vlaue, file_object, indent=4)  # indent=4 make perfect print
        return

    def resize(self, img_path, new_img_path):
        images = os.listdir(img_path)
        for image in images:
            img = ndimage.imread(img_path + image, mode='RGB')
            img = misc.imresize(img, (self.target_height, self.target_width))
            os.remove(img_path+image)
            misc.imsave(new_img_path + image, img)
    def count_png(self, img_path):
        files = os.listdir(img_path)
        count = 0
        for file in files:
            print(file[-4:])
            if file[-4:] == '.png':
                count+=1
        print('final count is: ', count)
    def draw_vertexs(self, poly1, poly2, poly3):

        pass
    def vertical_flip(self, path, new_fliped_path):
        files = os.listdir(path)
        for img in files:
            # img_ob = ndimage.imread(path+img, mode='RGB')
            # im_v = img_ob.rotate()
            img_ob = Image.open(path+img)
            img_v = img_ob.transpose(Image.FLIP_TOP_BOTTOM)  # flip image vertical
            misc.imsave(new_fliped_path+img, img_v)
            print(new_fliped_path+img)

if __name__ == '__main__':
    ut = utils()
    ...