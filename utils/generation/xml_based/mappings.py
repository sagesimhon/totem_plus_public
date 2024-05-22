import time

import mitsuba as mi

from config import mi_variant

mi.set_variant(mi_variant)
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label, center_of_mass
from scipy.spatial.distance import cdist

import math
import os


class Coords:
    """
    PARSES XML TO EXTRACT PARAMETERS FOR WORLD TO IMAGE CONVERSIONS
    """
    def __init__(self, fname, path_to_xmls, is_reddot=True):
        if not fname or not path_to_xmls:
            raise ValueError('Must provide fname and path to xmls')
        self.fname = fname  # filename of xml in consideration
        self.path_to_xmls = path_to_xmls  # path to xml directory containing the xml 'fname'
        self.path_to_file = os.path.join(path_to_xmls, fname)
        self.wc, self.r = self.get_totem_coordinates(self.read_xml_file())
        self.fov, self.near, self.far, self.res_x, self.res_y, self.origin, self.target, self.up = self.get_xml_params()
        if is_reddot:
            self.wc_reddot = self.get_reddot_coordinates(self.read_xml_file())
        self.pixel_coords = self.world_to_image(self.wc, self.fov, self.near, self.far, self.res_x, self.res_y, self.origin, self.target, self.up)
        # print("Totem center pixel coords: ", self.pixel_coords)

        # Get bounding box around totem
        self.wc_p_r = np.array([self.wc[0] + self.r, self.wc[1] + self.r, self.wc[2]])  # np.array([0.2, 0.2, 0])
        self.wc_m_r = np.array([self.wc[0] - self.r, self.wc[1] - self.r, self.wc[2]])  # np.array([-0.2, -0.2, 0])

        self.ic_p_r = self.world_to_image(self.wc_p_r, self.fov, self.near, self.far, self.res_x, self.res_y, self.origin, self.target, self.up)
        self.ic_m_r = self.world_to_image(self.wc_m_r, self.fov, self.near, self.far, self.res_x, self.res_y, self.origin, self.target, self.up)

        self.bounding_box = [self.ic_m_r, self.ic_p_r]

    def read_xml_file(self):
        with open(self.path_to_file, 'r') as f:
            xml_string = f.read()
        return xml_string

    @staticmethod
    def get_totem_coordinates(xml_string):
        root = ET.fromstring(xml_string)
        sphere = root.find("shape[@id='totem']")
        c = sphere.find("point[@name='center']")
        x = c.attrib['x']
        y = c.attrib['y']
        z = c.attrib['z']
        r = sphere.find("float[@name='radius']")
        r = r.attrib['value']
        return np.array([float(x), float(y), float(z)]), float(r)

    @staticmethod
    def get_reddot_coordinates(xml_string):
        root = ET.fromstring(xml_string)
        reddot = root.find("shape[@id='reddot']")
        to_world = reddot.find("./transform[@name='to_world']")
        location = to_world.find("./translate").attrib['value']
        return location

    @staticmethod
    def to_homogeneous(a):
        b = np.zeros((a.shape[0] + 1,) + a.shape[1:])
        b[:a.shape[0], ...] = a
        b[-1, -1, ...] = 1.0
        return b

    def get_xml_params(self):
        # Load the XML file
        xml_file = self.path_to_file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get the camera parameters
        sensor = root.find("sensor")
        fov = sensor.find("float[@name='fov']").attrib["value"]
        near = sensor.find("float[@name='near_clip']").attrib["value"]
        far = sensor.find("float[@name='far_clip']").attrib["value"]
        film = sensor.find("film")
        res_x = film.find("integer[@name='width']").attrib["value"]
        res_y = film.find("integer[@name='height']").attrib["value"]
        to_world = sensor.find("transform[@name='to_world']")
        lookat = to_world.find("lookat")
        origin = np.array([float(x) for x in lookat.attrib["origin"].split(",")])
        target = np.array([float(x) for x in lookat.attrib["target"].split(",")])
        up = np.array([float(x) for x in lookat.attrib["up"].split(",")])

        return float(fov), float(near), float(far), float(res_x), float(res_y), origin, target, up

    @staticmethod
    def world_to_image(world_coord, fov, near, far, resx, resy, origin, target, up):
        # Compute the view matrix
        forward = target - origin
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        view_matrix = np.eye(4)
        view_matrix[0, :-1] = right
        view_matrix[1, :-1] = up
        view_matrix[2, :-1] = -forward
        view_matrix[0, 3] = -np.dot(right, origin)
        view_matrix[1, 3] = -np.dot(up, origin)
        view_matrix[2, 3] = np.dot(forward, origin)

        # Compute the projection matrix
        aspect_ratio = resx / resy
        fov_rad = np.deg2rad(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = f / aspect_ratio
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (far + near) / (near - far)
        projection_matrix[3, 2] = -1.0
        projection_matrix[2, 3] = (2.0 * far * near) / (near - far)

        # Compute the transformation matrix
        world_coord = np.append(world_coord, 1)
        trans_matrix = np.dot(projection_matrix, np.dot(view_matrix, world_coord))

        # Convert to image coordinates
        x = ((trans_matrix[0] / trans_matrix[3]) + 1) * resx / 2
        y = ((trans_matrix[1] / trans_matrix[3]) + 1) * resy / 2

        return (x, y)

    # # Get params of this scene
    # fov, near, far, res_x, res_y, origin, target, up = get_xml_params(fname)

    # Get pixel coords of totem
    # pixel_coords = world_to_image(wc, fov, near, far, res_x, res_y, origin, target, up)
    # print(pixel_coords)
    #
    # # Get bounding box around totem
    # wc_p_r = np.array([wc[0]+r,wc[1]+r, wc[2]]) #np.array([0.2, 0.2, 0])
    # wc_m_r = np.array([wc[0]-r, wc[1]-r, wc[2]]) #np.array([-0.2, -0.2, 0])
    #
    # ic_p_r = world_to_image(wc_p_r, fov, near, far, res_x, res_y, origin, target, up)
    # ic_m_r = world_to_image(wc_m_r, fov, near, far, res_x, res_y, origin, target, up)
    #
    # bounding_box = [ic_m_r, ic_p_r]

# TODO
# 0. Bounding box automation wont work for other shapes (is unique to sphere), and potentially other camera angles. must consider all 4 corners, but those in the world might not be the same as those in the image - will have to think through this
# 1. Check that bounding box is actually correct. The r_x L and R should be symmetric but they are not. Do this by imshowing just the BBOX cropped image
# 2. THIS Totem center may be incorrect - check. Disparity with image (0, 1023 going down, 0, 1023 going right)
# Totem bbox is incorrect..
# in image x: Totem is contained between 400 and 600 but 428 to 608 seems incorrect a bit?
# in image y: Totem is contained between 800 and 1000 ish but we got 77 to 210.
# and THIS disparity between image coords in image and image coords in conversion, probably..
# y and x are also inverted in image
# 3. rounding issue with bbox - why are they floats? the conversion by chatgpt should give ints

class RedDot:
    """
    Handles finding uv_tot/uv_cam mapping for a red dot in a scene with a single totem.
    """

    def __init__(self, fname, output_dir, xml_dir, tot_bbox, resx, resy, spp=None):
        self.fname = fname  # filename of xml file representing the scene in consideration
        self.xml_dir = xml_dir  # path to xml directory containing the xml given by {fname}
        self.exp_dir = output_dir  # should = parent dir of xml_dir
        self.path_to_file = os.path.join(xml_dir, fname)
        self.resx = resx
        self.resy = resy

        self.tot_bbox = tot_bbox
        self.spp = spp

        self.image = self.get_img()
        # self.show_img(save_plots=True)

        self.centroids = self.get_centroids()  # (2, 2)
        self.uv_cam, self.uv_tot = self.label_centroids()  # each has shape (1, 2)

    def get_img(self):
        scene1 = mi.load_file(self.path_to_file)
        if self.spp and self.spp > 0:
            image = mi.render(scene1, spp=self.spp)
        else:
            image = mi.render(scene1)
        return image

    def show_img(self, show_bbox=False, show_lights=False, save_plots=False):
        plt.axis("off")
        plt.imshow(self.image ** (1.0 / 2.2))
        plt.title("FULL SCENE")
        if save_plots:
            subname = "renderings/render_" + self.fname[:-4] + "_fullscene" + ".png"
            name = os.path.join(self.exp_dir, subname)
            plt.savefig(name)
            plt.close()
        # plt.show()

        if show_bbox:
            xmin, ymin = self.tot_bbox[0]
            xmax, ymax = self.tot_bbox[1]

            cut = np.array(self.image)
            ymin_for_crop = int(self.resy - ymax)
            ymax_for_crop = math.ceil(self.resy - ymin)
            xmin_for_crop = int(xmin)
            xmax_for_crop = math.ceil(xmax)
            cut = cut[ymin_for_crop:ymax_for_crop, xmin_for_crop:xmax_for_crop]
            plt.imshow(cut ** (1.0 / 2.2))
            plt.title("BOUNDING BOX AROUND TOTEM")
            if save_plots:
                subname = "renderings/render_" + self.fname[:-4] + "_bbox" + ".png"
                name = os.path.join(self.exp_dir, subname)
                plt.savefig(name)
                plt.close()
            # plt.show()

        if show_lights:
            fname_with_lights = self.fname.replace('.xml', '_LO.xml')
            path_to_fname_with_lights = f'{self.xml_dir}/' + fname_with_lights
            img_lights = mi.render(mi.load_file(path_to_fname_with_lights))
            plt.imshow(img_lights ** (1.0 / 2.2))
            plt.title("LIGHTS ON")
            if save_plots:
                subname = "renderings/render_" + self.fname[:-4] + "_LO" + ".png"
                name = os.path.join(self.exp_dir, subname)
                plt.savefig(name)
                plt.close()
            # plt.show()

    def get_centroids(self, plot_markers=True):
        # Convert drjit.ArrayBase to NumPy array
        np_img = np.array(self.image)
        # print("image array of size", np_img.shape)

        # Find the connected components of non-black pixels
        mask = np.any(np_img > 0.05, axis=2)  # Create a binary mask of non-black pixels, avoiding small off-black pixels (not sure what their origin is)
        labels, num_labels = label(mask)  # Label connected components

        # Calculate the centroid of each component
        centroids = []
        for i in range(1, num_labels + 1):  # Skip label 0, which corresponds to the black pixels
            indices = np.argwhere(labels == i)  # Find the indices of pixels in the component
            centroid = center_of_mass(mask, labels, i)  # Calculate the centroid
            centroids.append(centroid)

        # Filter centroids based on distance
        n = 15  # Distance threshold TODO change to just consider all contiguous clusters?
        filtered_centroids = []
        for i, c1 in enumerate(centroids):
            is_far_enough = all(cdist([c1], [c2]) > n for j, c2 in enumerate(centroids) if j != i)
            if is_far_enough:
                filtered_centroids.append(c1)

        # Convert centroids from float to integer coordinates
        filtered_centroids = [(int(round(c[0])), int(round(c[1]))) for c in filtered_centroids]

        np_img = np.array(self.image)
        if len(filtered_centroids) == 2:
            pass
        elif len(filtered_centroids) == 3:
            # Filter out the higher order reflection
            sumz = [np.sum(np_img[y, x]) for y, x in filtered_centroids]
            for i, c in enumerate(filtered_centroids):
                y, x = c
                if np.sum(np_img[y, x]) == min(sumz):
                    filtered_centroids.pop(i)
            print("popped a centroid!")
            if len(filtered_centroids) != 2:
                print(f'centroids are: {filtered_centroids}')
                raise AssertionError("incorrect implementation!")
        elif len(filtered_centroids) > 3:
            print(f'centroids are: {filtered_centroids}')
            raise AssertionError("MORE THAN 3 filtered centroids!!")
        elif len(filtered_centroids) == 1:
            print(f'centroids are: {filtered_centroids}')
            raise AssertionError("ONLY 1 filtered centroid :(")
        else:
            raise AssertionError("something else is up :/")

        np_img_plotting = np.copy(np_img)
        if plot_markers:
            for i, c in enumerate(filtered_centroids):  # y, x = + to down , + to right
                y, x = c
                np_img_plotting[y - 5:y + 5, x - 5:x + 5] = [1, i, 1]

            plt.imshow(np_img_plotting ** (1.0 / 2.2))
            num_c = len(filtered_centroids)
            plt.title(f"{num_c} Centroids Found excluding those popped")

            subname = "renderings/render_" + self.fname[:-4] + "_centroids" + ".png"
            name = os.path.join(self.exp_dir, subname)

            plt.savefig(name)
            plt.close()

        # Return the result
        # assert len(filtered_centroids)==2, "Not exactly 2 filtered centroids found! Check the image manually."
        return filtered_centroids

    def label_centroids(self):
        c0, c1 = self.centroids

        # IMPORTANT STEP: invert coordinates from y,x to x,y
        c0 = c0[1], c0[0]
        c1 = c1[1], c1[0]

        c0_lbl = 'Cam'
        c1_lbl = 'Cam'

        xmin, ymin = self.tot_bbox[0]
        xmax, ymax = self.tot_bbox[1]
        ymin_for_crop = int(self.resx - ymax)
        ymax_for_crop = math.ceil(self.resy - ymin)
        xmin_for_crop = int(xmin)
        xmax_for_crop = math.ceil(xmax)

        # Check c0
        x, y = c0
        if x >= xmin_for_crop and x <= xmax_for_crop and y >= ymin_for_crop and y <= ymax_for_crop:
            c0_lbl = 'Totem'

        # Check c1
        x, y = c1
        if x >= xmin_for_crop and x <= xmax_for_crop and y >= ymin_for_crop and y <= ymax_for_crop:
            c1_lbl = 'Totem'

        assert c1_lbl != c0_lbl, "c0 and c1 have same label!"
        if c0_lbl == 'Totem' and c1_lbl == 'Cam':
            return c1, c0
        elif c0_lbl == 'Cam' and c1_lbl == 'Totem':
            return c0, c1
        else:
            raise AssertionError("c0 and c1 have same label and first assertion isn't working!")

####################################################################################################

