import os
import platform
import numpy as np
import math
import random
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.ops import unary_union
from collections import defaultdict

if platform.system() == "Windows":
    OPENSLIDE_PATH = r'D:\Openslide\Openslide\openslide-win64-20231011\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
elif platform.system() == "Linux":
    import openslide

class CropList:
    """
    The class CropList implements a type which behaves like a python list, but reads data from disk only when required.
    This allows the usage of large WSI without being constrained by the amount of available memory.
    """
    def __init__(self, indexes, size=None):
        self.indexes = indexes
        self.size = size

    def __add__(self, b):
        return CropList(self.indexes + b.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        section = self.indexes[idx]
        region = openslide.OpenSlide(
            section['filepath_slide']).read_region([section['left'],
                                                    section['top']],
                                                   section['level'],
                                                   [section['size'],
                                                    section['size']])
        region = region.convert('RGB')
        region = region.resize(size=(self.size, self.size))
        return np.array(region)

    def shuffle(self):
        random.shuffle(self.indexes)
    
class CropIndexer:
    def __init__(self, type, crop_size, overlap=1):
        """
        # SectionManager provides an easy way to generate a cropList object.
        # This object is not tied to a particular slide and can be reused to crop many slides using the same settings.
        @param crop_size: crop_size
        @param overlap: overlap (%)
        """
        self.crop_default = crop_size
        self.level = 0
        self.overlap = int(1/overlap)
        self.type = type

    @staticmethod
    def parse_xml_mask(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        masks_points = []

        for mask in root.find('Annotations'):
            points = []
            label = mask.attrib['PartOfGroup']
            for coordinate in mask.find('Coordinates'):
                points.append((float(coordinate.attrib['X']),
                               float(coordinate.attrib['Y'])))

            masks_points.append({'label': label, 'points': points})

        return masks_points
    
    def calculate_intersection_area(self, patch_polygon, masks_points):
        intersection_areas = defaultdict(float)  # Use defaultdict to sum areas for the same label
        for mask in masks_points:
            mask_polygon = Polygon(mask['points'])
            if not mask_polygon.is_valid:
                mask_polygon = mask_polygon.buffer(0)  # Attempt to repair the polygon
            intersection_area = patch_polygon.intersection(mask_polygon).area
            intersection_areas[mask['label']] += intersection_area  # Sum areas for labels
        return intersection_areas

    def decide_label(self, intersection_areas):
        # Decide based on the maximum summed intersection area
        if not intersection_areas:
            return []
        max_label = max(intersection_areas, key=intersection_areas.get)
        return max_label if intersection_areas[max_label] > 0 else []
    
    def patch_label(self, patch, masks_points):      

        patch_polygon = Polygon([(patch['left'], patch['top']), 
                                (patch['left']+patch['size'], patch['top']), 
                                (patch['left']+patch['size'], patch['top']+patch['size']), 
                                (patch['left'], patch['top']+patch['size'])]).buffer(0)

        labels = []
        for mask in masks_points:
            if patch_polygon.intersects(Polygon(mask['points'])):
                labels.append(mask['label'])

        # Check if there are multiple labels loaded from annotations and check for the best option 
        if len(labels)>1:
            intersection_areas = self.calculate_intersection_area(patch_polygon, masks_points)
            labels = self.decide_label(intersection_areas)

        return labels
    
    @staticmethod
    def is_background(region, threshold=210):

        if np.mean(region) > threshold:
            return True
        else:
            return False
        
    def check_background(self, slide, patch):

        region = slide.read_region([patch['left'],
                                                    patch['top']],
                                                self.level,
                                                [patch['size'],
                                                    patch['size']])
        region = region.convert('RGB')
        region = region.resize(size=(self.crop_default, self.crop_default))

        return self.is_background(region)
        
    def crop(self, slide_dir, xml_dir=None):

        if self.type=='XML':
            masks_points = self.parse_xml_mask(xml_path=xml_dir)

        slide = openslide.OpenSlide(slide_dir)
        downsample = slide.level_downsamples[self.level]
        if 'openslide.bounds-width' in slide.properties.keys():
            bounds_width = int(slide.properties['openslide.bounds-width'])
            bounds_height = int(slide.properties['openslide.bounds-height'])
            bounds_x = int(slide.properties['openslide.bounds-x'])
            bounds_y = int(slide.properties['openslide.bounds-y'])
        else:
            # print("{} - No Bounds".format(slide_dir))
            bounds_width = slide.dimensions[0]
            bounds_height = slide.dimensions[1]
            bounds_x = 0
            bounds_y = 0

        _step_ = int(self.crop_default / self.overlap)
        _size_ = math.floor(self.crop_default / downsample)
        self.sections = []

        # N.B. Crops are considered in the 0 level
        for y in range(int(math.floor(bounds_height / _step_))):
            for x in range(int(math.floor(bounds_width / _step_))):
                
                if x * _step_ + self.crop_default > bounds_width or y * _step_ + self.crop_default > bounds_height:
                    continue
                
                # Top and left boundaries
                _top_ = bounds_y + _step_ * y
                _left_ = bounds_x + _step_ * x
                
                # Create patch dictionary
                _patch_ = {'top': _top_, 'left':_left_, 'size': _size_, 'augmented': False}

                if self.type=='XML':
                    _patch_['label'] = self.patch_label(patch=_patch_, masks_points=masks_points)
                    if len(_patch_['label'])>0:
                        _patch_['label'] = _patch_['label'][0]
                        self.sections.append(_patch_)
                elif self.type=='SLIDE':
                    if not self.check_background(slide, _patch_):
                        self.sections.append(_patch_)
                             
        return list(self.sections)