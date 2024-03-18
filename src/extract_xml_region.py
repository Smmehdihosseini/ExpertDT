import numpy as np
import os
import cv2
import openslide
import lxml.etree as ET
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imsave
from skimage.transform import resize
from PIL import Image
import lxml.etree as ET
import time
import numpy as np

cwd = os.getcwd()
save_dir = cwd + '/extracted/'
size_thresh = None # not saved is uder this px area
final_image_size = 1200 # final_image_size
white_background = True # mask the structure
extract_one_region = True # include only one structure per patch (ignore other structures in ROI)

WSIs_ = glob(cwd + '/*.svs')
WSIs = []
XMLs = []

for WSI in WSIs_:
    xml_ = glob(WSI.split('.')[0] + '.xml')
    if xml_ != []:
        print('including: ' + WSI)
        XMLs.append(xml_[0])
        WSIs.append(WSI)

"""
location (tuple) - (x, y) tuple giving the top left pixel in the level 0 reference frame
size (tuple) - (width, height) tuple giving the region size

"""

def xml_to_mask(xml_path, location, size, downsample_factor=1, verbose=0):

    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculate region bounds
    bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0], 'y_max' : location[1] + size[1]}

    IDs = regions_in_mask(xml_path=xml_path, root=root, bounds=bounds, verbose=verbose)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # find regions in bounds
    Regions = get_vertex_points(root=root, IDs=IDs, verbose=verbose)

    # fill regions and create mask
    mask = Regions_to_mask(Regions=Regions, bounds=bounds, IDs=IDs, downsample_factor=downsample_factor, verbose=verbose)
    if verbose != 0:
        print('done...\n')

    return mask


def regions_in_mask(xml_path, root, bounds, verbose=1):
    # find regions to save
    IDs = []
    mtime = os.path.getmtime(xml_path)

    write_minmax_to_xml(xml_path, tree)

    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']

        for Region in Annotation.findall("./*/Region"): # iterate on all region

            for Vert in Region.findall("./Vertices"): # iterate on all vertex in region

                # get minmax points
                Xmin = np.int32(Vert.attrib['Xmin'])
                Ymin = np.int32(Vert.attrib['Ymin'])
                Xmax = np.int32(Vert.attrib['Xmax'])
                Ymax = np.int32(Vert.attrib['Ymax'])

                # test minmax points in region bounds
                if bounds['x_min'] <= Xmax and bounds['x_max'] >= Xmin and bounds['y_min'] <= Ymax and bounds['y_max'] >= Ymin:
                    # save region Id
                    IDs.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
                    break
    return IDs

def get_vertex_points(root, IDs, verbose=1):
    Regions = []

    for ID in IDs: # for all IDs

        # get all vertex attributes (points)
        Vertices = []

        for Vertex in root.findall("./Annotation[@Id='" + ID['annotationID'] + "']/Regions/Region[@Id='" + ID['regionID'] + "']/Vertices/Vertex"):
            # make array of points
            Vertices.append([int(float(Vertex.attrib['X'])), int(float(Vertex.attrib['Y']))])

        Regions.append(np.array(Vertices))

    return Regions

def Regions_to_mask(Regions, bounds, IDs, downsample_factor, verbose=1):
    downsample = int(np.round(downsample_factor**(.5)))

    if verbose !=0:
        print('\nMAKING MASK:')

    if len(Regions) != 0: # regions present
        # get min/max sizes
        min_sizes = np.empty(shape=[2,0], dtype=np.int32)
        max_sizes = np.empty(shape=[2,0], dtype=np.int32)
        for Region in Regions: # fill all regions
            min_bounds = np.reshape((np.amin(Region, axis=0)), (2,1))
            max_bounds = np.reshape((np.amax(Region, axis=0)), (2,1))
            min_sizes = np.append(min_sizes, min_bounds, axis=1)
            max_sizes = np.append(max_sizes, max_bounds, axis=1)
        min_size = np.amin(min_sizes, axis=1)
        max_size = np.amax(max_sizes, axis=1)

        # add to old bounds
        bounds['x_min_pad'] = min(min_size[1], bounds['x_min'])
        bounds['y_min_pad'] = min(min_size[0], bounds['y_min'])
        bounds['x_max_pad'] = max(max_size[1], bounds['x_max'])
        bounds['y_max_pad'] = max(max_size[0], bounds['y_max'])

        # make blank mask
        mask = np.zeros([ int(np.round((bounds['y_max_pad'] - bounds['y_min_pad']) / downsample)), int(np.round((bounds['x_max_pad'] - bounds['x_min_pad']) / downsample)) ], dtype=np.uint8)

        # fill mask polygons
        index = 0
        for Region in Regions:
            # reformat Regions
            Region[:,1] = np.int32(np.round((Region[:,1] - bounds['y_min_pad']) / downsample))
            Region[:,0] = np.int32(np.round((Region[:,0] - bounds['x_min_pad']) / downsample))
            # get annotation ID for mask color
            ID = IDs[index]
            cv2.fillPoly(mask, [Region], int(ID['annotationID']))
            index = index + 1

        # reshape mask
        x_start = np.int32(np.round((bounds['x_min'] - bounds['x_min_pad']) / downsample))
        y_start = np.int32(np.round((bounds['y_min'] - bounds['y_min_pad']) / downsample))
        x_stop = np.int32(np.round((bounds['x_max'] - bounds['x_min_pad']) / downsample))
        y_stop = np.int32(np.round((bounds['y_max'] - bounds['y_min_pad']) / downsample))
        # pull center mask region
        mask = mask[ y_start:y_stop, x_start:x_stop ]

    else: # no Regions
        mask = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) / downsample)), int(np.round((bounds['x_max'] - bounds['x_min']) / downsample)) ])

    return mask

def write_minmax_to_xml(xml_path, tree, time_buffer=10):
    # function to write min and max verticies to each region
    # parse xml and get root

    root = tree.getroot()

    try:
        # has the xml been modified to include minmax
        modtime = np.float64(root.attrib['modtime'])
        # has the minmax modified xml been changed?
        assert os.path.getmtime(xml_path) < modtime + time_buffer

    except:

        for Annotation in root.findall("./Annotation"): # for all annotations
            annotationID = Annotation.attrib['Id']

            for Region in Annotation.findall("./*/Region"): # iterate on all region

                for Vert in Region.findall("./Vertices"): # iterate on all vertex in region
                    Xs = []
                    Ys = []
                    for Vertex in Vert.findall("./Vertex"): # iterate on all vertex in region
                        # get points
                        Xs.append(np.int32(np.float64(Vertex.attrib['X'])))
                        Ys.append(np.int32(np.float64(Vertex.attrib['Y'])))

                    # find min and max points
                    Xs = np.array(Xs)
                    Ys = np.array(Ys)

                    # modify the xml
                    Vert.set("Xmin", "{}".format(np.min(Xs)))
                    Vert.set("Xmax", "{}".format(np.max(Xs)))
                    Vert.set("Ymin", "{}".format(np.min(Ys)))
                    Vert.set("Ymax", "{}".format(np.max(Ys)))

        root.set("modtime", "{}".format(time.time()))
        xml_data = ET.tostring(tree, pretty_print=True)
        #xml_data = Annotations.toprettyxml()
        f = open(xml_path, 'w')
        f.write(xml_data.decode())
        f.close()


def get_num_classes(xml_path):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotation_num = 0
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotation_num += 1

    return annotation_num + 1

def main():
    # go though all WSI
    for idx, XML in enumerate(XMLs):
        bounds, masks = get_annotation_bounds(XML,1)
        basename = os.path.basename(XML)
        basename = os.path.splitext(basename)[0]

        print('opening: ' + WSIs[idx])
        pas_img = openslide.OpenSlide(WSIs[idx])

        for idxx, bound in enumerate(bounds):
            if extract_one_region:
                mask = masks[idxx]
            else:
                mask=(xml_to_mask(XML,(bound[0],bound[1]), (final_image_size,final_image_size), downsample_factor=1, verbose=0))

            if size_thresh == None:
                PAS = pas_img.read_region((int(bound[0]),int(bound[1])), 0, (final_image_size,final_image_size))
                PAS = np.array(PAS)[:,:,0:3]

            else:
                size=np.sum(mask)
                if size >= size_thresh:
                    PAS = pas_img.read_region((bound[0],bound[1]), 0, (final_image_size,final_image_size))
                    PAS = np.array(PAS)[:,:,0:3]

            if white_background:
                for channel in range(3):
                    PAS_ = PAS[:,:,channel]
                    PAS_[mask == 0] = 255
                    PAS[:,:,channel] = PAS_

            subdir = '{}/{}/'.format(save_dir,basename)
            make_folder(subdir)
            imsave(subdir + basename + '_' + str(idxx) + '.jpg', PAS)


def get_annotation_bounds(xml_path, annotationID=1):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    Regions = root.findall("./Annotation[@Id='" + str(annotationID) + "']/Regions/Region")

    bounds = []
    masks = []
    for Region in Regions:
        Vertices = Region.findall("./Vertices/Vertex")
        x = []
        y = []

        for Vertex in Vertices:
            x.append(int(np.float32(Vertex.attrib['X'])))
            y.append(int(np.float32(Vertex.attrib['Y'])))

        x_center = min(x) + ((max(x)-min(x))/2)
        y_center = min(y) + ((max(y)-min(y))/2)

        bound_x = x_center-final_image_size/2
        bound_y = y_center-final_image_size/2
        bounds.append([bound_x, bound_y])

        points = np.stack([np.asarray(x), np.asarray(y)], axis=1)
        points[:,1] = np.int32(np.round(points[:,1] - bound_y ))
        points[:,0] = np.int32(np.round(points[:,0] - bound_x ))
        mask = np.zeros([final_image_size, final_image_size], dtype=np.int8)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask)

    return bounds, masks

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory


if __name__ == '__main__':
    main()
