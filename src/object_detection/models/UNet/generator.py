import cv2
import copy
import numpy as np
from keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, random_rotate, correct_bounding_boxes

class BatchGenerator(Sequence):
    def __init__(self, 
        instances,  
        labels,        
        max_box_per_image=30,
        batch_size=1,
        #min_net_size=320,
        #max_net_size=608,    
        net_input_height=608,    
        net_input_width=608,    
        shuffle=True, 
        jitter=True, 
        allow_horizontal_flip=True, 
        allow_vertical_flip=True, 
        allow_90deg_rotate=True, 
        min_resize_scale=0.25, 
        max_resize_scale=2, 
        hue_adjust=18, 
        saturation_adjust=1.5, 
        brightness_adjust=18, 
        exposure_adjust=1.5, 
        norm=None,
        custom_mask_func=None
    ):
        self.instances             = instances
        self.batch_size            = batch_size
        self.labels                = labels
        self.max_box_per_image     = max_box_per_image
        #self.min_net_size          = min_net_size
        #self.max_net_size          = max_net_size
        self.shuffle               = shuffle
        self.jitter                = jitter
        self.allow_horizontal_flip = allow_horizontal_flip
        self.allow_vertical_flip   = allow_vertical_flip
        self.allow_90deg_rotate    = allow_90deg_rotate
        self.min_resize_scale      = min_resize_scale
        self.max_resize_scale      = max_resize_scale
        self.hue_adjust            = hue_adjust
        self.saturation_adjust     = saturation_adjust
        self.brightness_adjust     = brightness_adjust
        self.exposure_adjust       = exposure_adjust
        self.norm                  = norm
        self.mask_func             = self._gen_mask if custom_mask_func is None else custom_mask_func
        self.net_h                 = net_input_height  
        self.net_w                 = net_input_width

        if shuffle: np.random.shuffle(self.instances)
        
        self.flip_opts             = []
        if self.allow_horizontal_flip and self.allow_vertical_flip:
            self.flip_opts.append(0)
            self.flip_opts.append(1)
            self.flip_opts.append(2)
        elif self.allow_horizontal_flip:
            self.flip_opts.append(1)
        elif self.allow_vertical_flip:
            self.flip_opts.append(0)
        self.flip_opts.append(3) # Default
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))           

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))             # input images
        t_batch = np.zeros((r_bound - l_bound, net_h, net_w, 1))             # list of groundtruth masks
        
        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)

            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(img, obj['name'], 
                                (obj['xmin']+2, obj['ymin']+12), 
                                0, 1.2e-3 * img.shape[0], 
                                (0,255,0), 2)
                
                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1
            
            # assign output image to t_batch
            mask = self.mask_func(all_objs, net_h, net_w)
            if self.norm != None: 
                t_batch[true_box_index] = self.norm(mask)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(mask, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(mask, obj['name'], 
                                (obj['xmin']+2, obj['ymin']+12), 
                                0, 1.2e-3 * img.shape[0], 
                                (0,255,0), 2)
                
                t_batch[true_box_index] = mask
            
            # increase instance counter in the current batch
            true_box_index += 1

            '''# DEBUG
            cv2.imshow('image', img) 
            cv2.imshow('mask', mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
                
        return x_batch, t_batch

    def _get_net_size(self, idx):
        '''if idx%10 == 0:
            net_size = np.random.randint(self.min_net_size, self.max_net_size+1)
            print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size'''
        return self.net_h, self.net_w
    
    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name) # BGR image
        
        if image is None: print('Cannot find ', image_name)
        image = image[:,:,::-1] # RGB image
            
        image_h, image_w, _ = image.shape
        
        # determine the amount of scaling and cropping
        dw = self.jitter * image_w;
        dh = self.jitter * image_h;

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
        scale = np.random.uniform(self.min_resize_scale, self.max_resize_scale); # RIELCZ CHANGE: ORIGINALLY (0.25, 2)

        if (new_ar < 1):
            new_h = int(scale * net_h);
            new_w = int(new_h * new_ar); # RIELCZ CHANGE: ORIGINALLY new_w = int(net_h * new_ar);
        else:
            new_w = int(scale * net_w);
            new_h = int(new_w * new_ar); # RIELCZ CHANGE: ORIGINALLY new_h = int(net_w / new_ar);
            
        dx = int(np.random.uniform(0, net_w - new_w));
        dy = int(np.random.uniform(0, net_h - new_h));
        
        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
        
        # randomly distort hsv space
        im_sized = random_distort_image(im_sized, hue=self.hue_adjust, saturation=self.saturation_adjust, exposure=self.exposure_adjust, brightness=self.brightness_adjust)
        
        # randomly flip
        if (not self.allow_horizontal_flip) and (not self.allow_vertical_flip):
            flip = 3 # No flip by default
        else:
            flip = np.random.choice(self.flip_opts) # RIELCZ CHANGE
            im_sized = random_flip(im_sized, flip)
        
        # randomly rotate # RIELCZ CHANGE
        if self.allow_90deg_rotate:
            rotation = np.random.randint(4)
            im_sized = random_rotate(im_sized, rotation)
        else:
            rotation = 0 # No rotate
            
        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, rotation, image_w, image_h)
        
        return im_sized, all_objs   

    def _gen_mask(self, boxes, net_h, net_w):
        toReturn = np.zeros((net_h, net_w, 1))
        for i in range(len(boxes)):
            toReturn[boxes[i]['ymin']:boxes[i]['ymax']+1, boxes[i]['xmin']:boxes[i]['xmax']+1, :] = 255
        return toReturn

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)    

    def load_filename(self, i):
        return self.instances[i]['filename']

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.load_filename(i))