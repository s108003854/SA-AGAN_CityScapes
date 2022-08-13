
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint

# load and prepare training images
def load_real_samples(filename):
    # load compressed ararys
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]
# load dataset
[X1, X2] = load_real_samples('CityScapes_Training_DataSet.npz')
print('Loaded', X1.shape, X2.shape)

# load model
model = load_model('CityScapes-withoutSA/model_148750.h5')

# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]


# generate image from source
gen_image = model.predict(src_image)

def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']
    pyplot.figure(figsize=(20,20))
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 3, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show title
        pyplot.title(titles[i])
    pyplot.show()

import tensorflow as tf
def PSNR(gen_img, tar_img):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        y=sess.run(tf.image.psnr(gen_img,tar_img,max_val=1))
        return y[0]

# # plot all three images
# plot_images(src_image, gen_image, tar_image)
# gen_image = model.predict(src_image)
# print(PSNR(gen_image, tar_image))

# # select random example
# ix = randint(0, len(X1), 1)
# src_image, tar_image = X1[ix], X2[ix]
# gen_image = model.predict(src_image)
# print(PSNR(gen_image, tar_image))
# plot_images(src_image, gen_image, tar_image)

# performance=[]
# for index in range(11):
#     i = index*10
#     #src_image, tar_image = X1[i], X2[i]
#     src_image, tar_image = X1[i].reshape(1,256,256,3), X2[i].reshape(1,256,256,3)
#     gen_image = model.predict(src_image)
#     plot_images(src_image, gen_image, tar_image)
#     psnr = PSNR(gen_image, tar_image)
#     print(psnr)
#     performance.append(psnr)

# sum(performance)/len(performance)

def coor_get(r,g,b,image):
    coor=[]
    for i in range(256):
        for j in range(256):
            img=image[0][i][j]*127.5+127.5
            #s=abs(img[0]-r)+abs(img[1]-g)+abs(img[2]-b)
            s1=abs(img[0]-r);s2=abs(img[1]-g); s3=abs(img[2]-b)
            if(s1<=10):
                if(s2<=10):
                    if(s3<=10):
                        coor.append((i,j))
    return coor

def IOU(r,g,b,gen_image,tar_image):
    coor_gen=coor_get(r,g,b,gen_image)
    coor_tar=coor_get(r,g,b,tar_image)
    U=0
    for i in coor_gen:
        for j in coor_tar:
            if(i==j):
                coor_gen.remove(i)
                coor_tar.remove(j)
                U+=1
    TP=U;FP=len(coor_gen)-U;FN=len(coor_tar)-U
    s=TP+FP+FN
    if(s==0):
        s+=1
    IoU=TP/s
    return(IoU)

from collections import namedtuple
Label =  namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

result = {
#     'unlabeled':[]            ,
#     'ego vehicle':[]            ,
#     'rectification border':[]   ,
#     'out of roi':[]             ,
#     'static':[]                 ,
#     'dynamic':[]                ,
#     'ground':[]                 ,
    'road':[]                   ,
    'sidewalk':[]               ,
#     'parking':[]                ,
#     'rail track':[]             ,
    'building':[]               ,
    'wall':[]                   ,
    'fence':[]                ,
#     'guard rail':[]           ,
#     'bridge':[]               ,
#     'tunnel':[]               ,
    'pole':[]                 ,
#     'polegroup':[]            ,
    'traffic light':[]        ,
    'traffic sign':[]         ,
    'vegetation':[]           ,
    'terrain':[]              ,
    'sky':[]                  ,
    'person':[]               ,
    'rider':[]                ,
    'car':[]                  ,
    'truck':[]                ,
    'bus':[]                  ,
#     'caravan':[]              ,
#     'trailer':[]              ,
    'train':[]                ,
    'motorcycle':[]           ,
    'bicycle':[]              ,
#     'license plate':[]        
}

for i in range(len(result)):
    label=labels[i]
    for i in range(len(X1)):
        src_image, tar_image = X1[i].reshape(1,256,256,3), X2[i].reshape(1,256,256,3)
        gen_image = model.predict(src_image)
        mIOU = IOU(label.color[0], label.color[1],label.color[2], gen_image, tar_image)
        result[label.name].append(mIOU)

for i in range(len(result)):
    label=labels[i]
    myList = [value for value in result[label.name] if value != 0]
    print(label.name , round(sum(myList)/len(myList)*100,2))


