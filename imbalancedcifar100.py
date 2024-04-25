

import numpy as np
import torchvision
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.002, rand_number=0, train=True,task_num=10,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.task_num = task_num
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num 
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(max(int(num), 1))
        elif imb_type == 'exp_re':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(max(int(num), 1))
            img_num_per_cls.reverse()
        elif imb_type == 'exp_max':
            cls_per_group = cls_num//self.task_num 
            for cls_idx in range(cls_num):
                if (cls_idx+1)%cls_per_group==1:
                    # print(cls_idx)
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'exp_max_re':
            cls_per_group = cls_num//self.task_num 
            for cls_idx in range(cls_num):
                if (cls_idx+1)%cls_per_group==1:
                    # print(cls_idx)
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
            img_num_per_cls.reverse()    
            
        elif imb_type == 'exp_min':
            cls_per_group = cls_num//self.task_num
            for cls_idx in range(cls_num):
                if (cls_idx+1)%cls_per_group==1:
                    # print(cls_idx)
                    num = img_max * (imb_factor**((cls_idx+cls_per_group-1) / (cls_num - 1.0)))
                    # print(num)
                img_num_per_cls.append(int(num))
            
        elif imb_type == 'half':
            cls_per_group = cls_num // self.task_num
            ratio = 2
            num = 1  
            for cls_idx in range(cls_num):
                if num > img_max:  
                    num = img_max
                img_num_per_cls.append(int(num))
                if (cls_idx + 1) % cls_per_group == 0:  
                    num *= ratio
            img_num_per_cls.reverse()
                                    
        elif imb_type == 'half_re':
            cls_per_group = cls_num // self.task_num
            ratio = 2
            num = 1  
            for cls_idx in range(cls_num):
                if num > img_max:  
                    num = img_max
                img_num_per_cls.append(int(num))
                if (cls_idx + 1) % cls_per_group == 0:  
                    num *= ratio

                    
        elif imb_type == 'halfbal':
            cls_per_group = cls_num // self.task_num
            N = img_max * cls_per_group  
            
            total = 0
            for i in range(self.task_num):
                total += N / (2**i)
            print(total)    
            per_class_count = int(total / cls_num)
            img_num_per_cls.extend([per_class_count] * cls_num)
            
        elif imb_type == 'oneshot':
            img_num_per_cls.extend([1] * cls_num)
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'fewshot':
            for cls_idx in range(cls_num):
                if cls_idx<50:
                    num = img_max
                else:
                    num = img_max*0.01
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

