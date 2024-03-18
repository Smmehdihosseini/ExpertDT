import os
import platform
from glob import glob
import math
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from concurrent.futures import ThreadPoolExecutor, as_completed

if platform.system() == "Windows":
    OPENSLIDE_PATH = r'D:\Openslide\Openslide\openslide-win64-20231011\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
    DIV = "\\"
elif platform.system() == "Linux":
    import openslide
    DIV = "/"

from model.vgg16 import Vgg16

class ExpertDT:
    
    def __init__(self, crop_size=1000,
                 crop_resize=112,
                 level=0,
                 overlap=1,
                 batch_size=128,
                 weights_dir='_weights',
                 tree_pair_dir='_info/tree_pair_dict.json',
                 first_trained_layer = 11,
                 pruned_threshold=30,
                 save_plots=True):

        self.crop_size = crop_size
        self.crop_resize = crop_resize
        self.level = level
        self.overlap = overlap
        self.batch_size = batch_size
        self.weights_dir = weights_dir
        self.n_classes = 2
        self.first_trained_layer = first_trained_layer
        self.tree_pair_dir = tree_pair_dir
        self.node_pruned_threshold = pruned_threshold
        self.save_plots = save_plots

        self.stages_status = {
                                "Root": "Not Defined",
                                "Node": "Not Defined",
                                "Leaf1": "Not Defined",
                                "Leaf2": "Not Defined",
                            }

        self.load_tree_pair_dict()
        self.load_tree_weights()

    def load_tree_pair_dict(self):

        with open(f"{self.tree_pair_dir}", 'r') as file:
            self.tree_pair_dict = json.load(file)   

    def process_patch(self, args):

        x, y, _step_, bounds_y, bounds_x, _size_ = args
        if x * _step_ + self.crop_size > self.bounds_width or y * _step_ + self.crop_size > self.bounds_height:
            return None
        _top_ = _step_ * y + bounds_y
        _left_ = _step_ * x + bounds_x

        _patch_ = {'x': x, 'y': y, 'top': _top_, 'left': _left_, 'size': _size_}
        
        if not self.check_background(_patch_):
            _label_ = 1
        else:
            _label_ = 0
        _patch_['label'] = _label_

        return _patch_

    def init_patches(self, slide_dir):

        print(">>> Start Slide Analysis ...")

        self.slide = openslide.OpenSlide(slide_dir)
        downsample = self.slide.level_downsamples[self.level]
        self.bounds_width, self.bounds_height = self.slide.dimensions if 'openslide.bounds-width' not in self.slide.properties else (int(self.slide.properties['openslide.bounds-width']), int(self.slide.properties['openslide.bounds-height']))
        bounds_x, bounds_y = (0, 0) if 'openslide.bounds-x' not in self.slide.properties else (int(self.slide.properties['openslide.bounds-x']), int(self.slide.properties['openslide.bounds-y']))

        _step_ = int(self.crop_size / self.overlap)
        _size_ = math.floor(self.crop_size / downsample)
        _y_steps_ = int(math.ceil((self.bounds_height - self.crop_size) / _step_))
        _x_steps_ = int(math.ceil((self.bounds_width - self.crop_size) / _step_))

        patch_args = [(x, y, _step_, bounds_y, bounds_x, _size_) for y in range(_y_steps_) for x in range(_x_steps_)]

        self.patches_dict = []
        self.slide_array = np.zeros((_y_steps_, _x_steps_))

        print("[Step 1/4] >>> Cropping WSI into patches ...")

        with tqdm(total=len(patch_args)) as pbar:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.process_patch, arg) for arg in patch_args]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        self.patches_dict.append(result)
                        self.slide_array[result['y'], result['x']] = result['label']
                    pbar.update(1)

        self.patches_dict = list(self.patches_dict)    
    
    @staticmethod
    def is_background(region, threshold=210):

        if np.mean(region) > threshold:
            return True
        else:
            return False
        
    def check_background(self, patch):

        region = self.slide.read_region([patch['left'],
                                                    patch['top']],
                                                self.level,
                                                [patch['size'],
                                                    patch['size']])
        
        region = region.convert('RGB').resize(size=(self.crop_resize, self.crop_resize))

        return self.is_background(region)
    
    def _refine(self, input):
        
        central = input[input.size // 2]
        values, counts = np.unique(input, return_counts=True)
        max_freq = counts.max()
        modes = values[counts == max_freq]
        
        if central==0:
            return 0
        elif central in modes:
            return central
        else:
            return modes[0]
        
    def refine_output(self, input_crops, size=(3, 3)):

        refined_crops = ndimage.generic_filter(input_crops, self._refine, size=size)

        return refined_crops
    
    def load_tree_weights(self):

        stage_folders = [os.path.join(self.weights_dir, f) for f in os.listdir(self.weights_dir) if os.path.isdir(os.path.join(self.weights_dir, f))]

        self.final_weights = {}

        for stage in stage_folders:

            files = glob(os.path.join(stage, '*'))
            
            if files:

                files.sort(key=os.path.getmtime, reverse=True)
                latest_update = os.path.basename(files[0])
                self.final_weights[os.path.basename(stage)] = latest_update

    def check_stage_requirement(self, stage):

        if stage == "Node":
            if self.stages_status["Root"] != "Not Defined":
                return True
            else:
                return False
        
        elif stage == "Leaf1" or stage == "Leaf2" :
            if self.stages_status["Node"] != "Not Defined":
                return True
            else:
                return False
            
    def predict_root(self):
        
        vgg16 = Vgg16(input_shape=(self.crop_resize, self.crop_resize, 3),
                      n_classes=self.n_classes,
                      first_trained_layer=self.first_trained_layer)
        
        vgg16.load_weights(weights_path=f"{self.weights_dir}{DIV}Root{DIV}{self.final_weights['Root']}")

        self.root_array = self.slide_array.copy()

        image_batch = []
        patch_indices = []

        def process_batch(batch, indices):
            if batch:
                pred_labels = vgg16.predict(np.array(batch))
                pred_classes = np.argmax(pred_labels, axis=1)
                for pred_class, index in zip(pred_classes, indices):
                    self.root_array[self.patches_dict[index]['y'], self.patches_dict[index]['x']] = pred_class+1

        print("[Step 2/4] >>> Root Stage Analysis ...")

        for patch_index in tqdm(range(len(self.patches_dict))):

            root_condition = self.patches_dict[patch_index]['label'] != 0

            if root_condition:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                region = region.convert('RGB').resize(size=(self.crop_resize, self.crop_resize))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_indices.append(patch_index)

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_indices)
                    image_batch = []
                    patch_indices = []

        process_batch(image_batch, patch_indices)

        self.root_refined_array = self.refine_output(self.root_array)
        self.stages_status['Root'] = "Done"

        
    def predict_node(self):
        
        if not self.check_stage_requirement(stage="Node"):
            print("[WARNING] No Root Analysis Found ...")
            return

        vgg16 = Vgg16(input_shape=(self.crop_resize, self.crop_resize, 3),
                      n_classes=self.n_classes,
                      first_trained_layer=self.first_trained_layer)
        
        vgg16.load_weights(weights_path=f"{self.weights_dir}{DIV}Node{DIV}{self.final_weights['Node']}")

        self.node_array = self.root_refined_array.copy()

        print("[Step 3/4] >>> Node Stage Analysis ...")
        
        image_batch = []
        patch_info_batch = []

        def process_batch(batch, info_batch):
            if not batch:
                return
            
            pred_labels = vgg16.predict(np.array(batch))
            pred_classes = np.argmax(pred_labels, axis=1)
            
            for pred_class, info in zip(pred_classes, info_batch):
                y, x, patch_index = info
                if pred_class == 0:
                    node_label = 3
                else:
                    node_label = 2
                self.node_array[y, x] = node_label

        for patch_index in tqdm(range(len(self.patches_dict))):
            root_crop_label = self.root_refined_array[self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x']]

            if root_crop_label == 2:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                region = region.convert('RGB').resize(size=(self.crop_resize, self.crop_resize))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_info_batch.append((self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x'], patch_index))

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_info_batch)
                    image_batch = []
                    patch_info_batch = []

        process_batch(image_batch, patch_info_batch)

        self.node_refined_array = self.refine_output(self.node_array)

        unique_values, counts = np.unique(self.node_refined_array, return_counts=True)
        value_counts = dict(zip(unique_values, counts))

        self.node_count_1, self.node_count_2 = value_counts.get(2, 0), value_counts.get(3, 0)
        
        self.node_cert = 0

        if self.node_count_1+self.node_count_2 != 0:
            self.node_cert = round(abs(self.node_count_1 - self.node_count_2) / (self.node_count_1 + self.node_count_2) * 100, 2)

            if self.node_cert >= self.node_pruned_threshold:

                node_majority = 1 if self.node_count_1 > self.node_count_2 else 2
                print(f"--- >>> Node Confidence: : {self.node_cert}%, Node to Leaf {node_majority}")
                self.stages_status['Node'] = f'Leaf{node_majority}' 
            else:
                self.stages_status['Node'] = 'Pruned'
                print("--- >>> Node is Pruned! Connecting Root Directly to Leafs ...")
        else:
            self.stages_status['Node'] = 'Pruned'
            print("--- >>> Node is Pruned! Connecting Root Directly to Leafs ...")

    def predict_leaf1(self):

        vgg16 = Vgg16(input_shape=(self.crop_resize, self.crop_resize, 3),
                      n_classes=self.n_classes,
                      first_trained_layer=self.first_trained_layer)
        
        vgg16.load_weights(weights_path=f"{self.weights_dir}{DIV}Leaf1{DIV}{self.final_weights['Leaf1']}")

        self.leaf1_array = self.node_refined_array.copy()
        self.leaf1_array[self.leaf1_array == 3] = 1

        print("[Step 4/4] >>> Leaf Stage Analysis: Leaf 1 ...")
        
        image_batch = []
        patch_info_batch = []

        def process_batch(batch, info_batch):
            if not batch:
                return
            
            pred_labels = vgg16.predict(np.array(batch))
            pred_classes = np.argmax(pred_labels, axis=1)
            
            for pred_class, info in zip(pred_classes, info_batch):
                y, x, patch_index = info
                if pred_class == 0:
                    leaf1_label = 4
                else:
                    leaf1_label = 5

                self.leaf1_array[y, x] = leaf1_label

        for patch_index in tqdm(range(len(self.patches_dict))):

            node_crop_label = self.node_refined_array[self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x']]

            if node_crop_label == 2:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                
                region = region.convert('RGB').resize(size=(self.crop_resize, self.crop_resize))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_info_batch.append((self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x'], patch_index))

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_info_batch)
                    image_batch = []
                    patch_info_batch = []

        process_batch(image_batch, patch_info_batch)

        self.leaf1_refined_array = self.refine_output(self.leaf1_array)

        unique_leaf1, count_leaf1 = np.unique(self.leaf1_refined_array, return_counts=True)
        self.value_counts_leaf1 = dict(zip(unique_leaf1, count_leaf1))
    
    def predict_leaf2(self):

        vgg16 = Vgg16(input_shape=(self.crop_resize, self.crop_resize, 3),
                      n_classes=self.n_classes,
                      first_trained_layer=self.first_trained_layer)
        
        vgg16.load_weights(weights_path=f"{self.weights_dir}{DIV}Leaf2{DIV}{self.final_weights['Leaf2']}")

        self.leaf2_array = self.node_refined_array.copy()
        self.leaf2_array[self.leaf2_array == 2] = 1

        print("[Step 4/4] >>> Leaf Stage Analysis: Leaf 2 ...")
        
        image_batch = []
        patch_info_batch = []

        def process_batch(batch, info_batch):
            if not batch:
                return
            
            pred_labels = vgg16.predict(np.array(batch))
            pred_classes = np.argmax(pred_labels, axis=1)
            
            for pred_class, info in zip(pred_classes, info_batch):
                y, x, patch_index = info
                if pred_class == 0:
                    leaf2_label = 2
                else:
                    leaf2_label = 3

                self.leaf2_array[y, x] = leaf2_label

        for patch_index in tqdm(range(len(self.patches_dict))):

            node_crop_label = self.node_refined_array[self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x']]

            if node_crop_label == 3:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                
                region = region.convert('RGB').resize(size=(self.crop_resize, self.crop_resize))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_info_batch.append((self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x'], patch_index))

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_info_batch)
                    image_batch = []
                    patch_info_batch = []

        process_batch(image_batch, patch_info_batch)

        self.leaf2_refined_array = self.refine_output(self.leaf2_array)

        unique_leaf2, count_leaf2 = np.unique(self.leaf2_refined_array, return_counts=True)
        self.value_counts_leaf2 = dict(zip(unique_leaf2, count_leaf2))

    def predict_leaf(self):

        if not self.check_stage_requirement(stage="Node"):
            print("[WARNING] No Root Analysis Found ...")
            return
        
        if self.stages_status['Node'] == 'Pruned':

            self.predict_leaf1()
            self.predict_leaf2()

            value_counts_leaf = {**self.value_counts_leaf1, **self.value_counts_leaf2}

            ccrcc_count, prcc_count = value_counts_leaf.get(2, 0), value_counts_leaf.get(3, 0)
            chromo_count, onco_count = value_counts_leaf.get(4, 0), value_counts_leaf.get(5, 0)

            self.subtype_counts = {
                                'ccRCC': ccrcc_count,
                                'pRCC': prcc_count,
                                'CHROMO': chromo_count,
                                'ONCOCYTOMA': onco_count
                            }
            
            self.max_subtype = max(self.subtype_counts, key=self.subtype_counts.get)

        elif self.stages_status['Node'] == 'Leaf1':

            self.predict_leaf1()

            chromo_count, onco_count = self.value_counts_leaf1.get(4, 0), self.value_counts_leaf1.get(5, 0)

            self.subtype_counts = {
                                'ccRCC': 0,
                                'pRCC': 0,
                                'CHROMO': chromo_count,
                                'ONCOCYTOMA': onco_count
                            }
            
            self.max_subtype = max(self.subtype_counts, key=self.subtype_counts.get)

        elif self.stages_status['Node'] == 'Leaf2':

            self.predict_leaf2()

            ccrcc_count, prcc_count = self.value_counts_leaf2.get(2, 0), self.value_counts_leaf2.get(3, 0)

            self.subtype_counts = {
                                'ccRCC': ccrcc_count,
                                'pRCC': prcc_count,
                                'CHROMO': 0,
                                'ONCOCYTOMA': 0
                            }
            
            self.max_subtype = max(self.subtype_counts, key=self.subtype_counts.get)

    def predict(self, slide_dir):

        self.init_patches(slide_dir)
        self.predict_root()
        self.predict_node()
        self.predict_leaf()

        print(f">>> Results:")
        print(f">>> Slide Label: {self.max_subtype}")
        print(f">>> Subtype Count:\n {self.subtype_counts}\n")

    def save_figs(self, note_dir, save_dir, id, slidename):

        if self.save_plots:

            print(">>> Saving Figures ...")

            os.makedirs(f'{save_dir}{DIV}{note_dir}{DIV}{id}{DIV}{slidename}')

            stages_plot = ['Root', 'Node']

            if self.stages_status['Node']=='Pruned':
                stages_plot.append('Leaf1')
                stages_plot.append('Leaf2')
            else:
                stages_plot.append(self.stages_status['Node'])

            for stage in stages_plot:
                self.save_plot_stage(stage,
                                     name_dir=f'{save_dir}{DIV}{note_dir}{DIV}{id}{DIV}{slidename}{DIV}{slidename}_{stage}.png',
                                      dpi=300, alpha=1)

    def save_plot_stage(self, stage, name_dir, dpi=300, alpha=1):

        if stage=="Root":
            arrays = [self.root_array, self.root_refined_array]

        elif stage=="Node":
            arrays = [self.node_array, self.node_refined_array]

        elif stage=="Leaf1":
            arrays = [self.leaf1_array, self.leaf1_refined_array]

        elif stage=="Leaf2":
            arrays = [self.leaf2_array, self.leaf2_refined_array]

        stage_kwargs = PlotTree().PlotStage(stage=stage, alpha=alpha)

        sorted_keys = sorted(stage_kwargs['label_dict'].keys(), key=int)

        labels = [stage_kwargs['label_dict'][key] for key in sorted_keys]
        colors = [stage_kwargs['stage_colors'][int(key)] for key in sorted_keys]

        cmap = ListedColormap(colors)
        bounds = [float(key) - 0.5 for key in sorted_keys] + [float(sorted_keys[-1]) + 0.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(1, len(arrays), figsize=(20, 5))
        for ax, array, title in zip(axes, arrays, stage_kwargs['titles']):
            im = ax.imshow(array, cmap=cmap, norm=norm)
            ax.set_title(title)

        handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(name_dir, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

    def plot_stage_on_slide(self, stage, alpha=1):

        if stage=="Root":
            arrays = [self.root_array, self.root_refined_array]

        elif stage=="Node":
            arrays = [self.node_array, self.node_refined_array]

        elif stage=="Leaf1":
            arrays = [self.leaf1_array, self.leaf1_refined_array]

        elif stage=="Leaf2":
            arrays = [self.leaf2_array, self.leaf2_refined_array]

        stage_kwargs = PlotTree().PlotStage(stage=stage, alpha=alpha)

        sorted_keys = sorted(stage_kwargs['label_dict'].keys(), key=int)

        labels = [stage_kwargs['label_dict'][key] for key in sorted_keys]
        colors = [stage_kwargs['stage_colors'][int(key)] for key in sorted_keys]

        cmap = ListedColormap(colors)
        bounds = [float(key) - 0.5 for key in sorted_keys] + [float(sorted_keys[-1]) + 0.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(1, len(arrays), figsize=(20, 5))
        for ax, array, title in zip(axes, arrays, stage_kwargs['titles']):
            im = ax.imshow(array, cmap=cmap, norm=norm)
            ax.set_title(title)

        handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


class PlotTree:

    def __init__(self):
        
        self.plot_dict = {}

    def PlotStage(self, stage, alpha):

        self.colors_rgba = {
                                "Red": (1.0, 0.0, 0.0, alpha),
                                "Orange": (1.0, 0.5, 0.0, alpha),
                                "Yellow": (1.0, 1.0, 0.0, alpha),
                                "Lime Green": (0.5, 1.0, 0.0, alpha),
                                "Green": (0.0, 1.0, 0.0, alpha),
                                "Spring Green": (0.0, 1.0, 0.5, alpha),
                                "Cyan": (0.0, 1.0, 1.0, alpha), 
                                "Blue": (0.0, 0.0, 1.0, alpha),
                                "Purple": (0.5, 0.0, 1.0, alpha),
                                "Magenta": (1.0, 0.0, 1.0, alpha),
                                "Pink": (1.0, 0.75, 0.8, alpha),
                                "Dark Red": (0.6, 0.0, 0.0, alpha),
                                "White": (1.0, 1.0, 1.0, alpha)
                            }

        self.plot_dict = {
            
            "Root": {
                "titles": ['Root', 'Refined Root'],
                "stage_colors": [self.colors_rgba['White'], self.colors_rgba['Pink'], self.colors_rgba['Dark Red']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "2": "Tumor"}

            },

            "Node": {
                "titles": ['Node', 'Refined Node'],
                "stage_colors": [self.colors_rgba['White'], self.colors_rgba['Pink'], self.colors_rgba['Blue'], self.colors_rgba['Orange']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "2": "CHROMO/ONCO", "3": "ccRCC/pRCC"}

            },

            "Leaf1": {
                "titles": ['Leaf1', 'Refined Leaf1'],
                "stage_colors": [ self.colors_rgba['White'], self.colors_rgba['Pink'], "", "", self.colors_rgba['Purple'], self.colors_rgba['Magenta']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "4": "CHROMO", "5": "ONCOCYTOMA"}

            },

            "Leaf2": {
                "titles": ['Leaf2', 'Refined Leaf2'],
                "stage_colors": [self.colors_rgba['White'], self.colors_rgba['Pink'], self.colors_rgba['Lime Green'], self.colors_rgba['Cyan']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "2": "ccRCC", "3": "pRCC"}

            }
        }

        return self.plot_dict[stage]