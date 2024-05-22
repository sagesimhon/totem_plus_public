import time

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import copy
import numpy as np
import mitsuba as mi

from config import mi_variant

mi.set_variant(mi_variant)
import math
import shutil
import glob
from utils.generation.xml_based.mappings import Coords

def generate_coords(xrange, yrange, zrange, resx, resy, n=999):

    # n = 5 means 125 coords. n = cubed root of number of coordinates to generate (5 intervals in each dim)

    # coord = np.array([0.0, -0.5, -0.78]) # initial coordinate

    # min_y = 0
    n = n/999
    xmin, xmax = xrange
    ymin, ymax = yrange
    zmin, zmax = zrange
    x = np.linspace(xmin, xmax, num=math.ceil(n*resx/2*(xmax-xmin))+1) # evenly spaced intervals for x-axis
    y = np.linspace(ymin, ymax, num=math.ceil(n*resy/2*(ymax-ymin))+1) # evenly spaced intervals for y-axis
    z = np.linspace(zmin, zmax, num=math.ceil(n*max(resx, resy)/2*(zmax-zmin))+1) # evenly spaced intervals for z-axis

    # create meshgrid of x, y, and z coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # flatten the meshgrid coordinates and add the initial coordinate to get final coordinates
    coords = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    print("Coords shape:", coords.shape)
    return coords

def generate_files(ranges, og_file, path_to_xmls, n=None, show=False):
    path_to_og_file = os.path.join(path_to_xmls,og_file)
    xml_parser = Coords(og_file, path_to_xmls, is_reddot=False)

    coords = generate_coords(ranges[0], ranges[1], ranges[2], xml_parser.res_x, xml_parser.res_y, n)
    print(f"Generating {coords.shape} xml files for each reddot pixel location...")
    # read in the original XML file
    tree = ET.parse(path_to_og_file)
    root = tree.getroot()

    # set the ID of the object to modify
    obj_id = 'reddot'

    # loop over each coordinate and modify the XML file
    for i, coord in enumerate(coords):
        # make a copy of the root element for the new XML file
        # make a copy of the root element for the new XML file
        new_root = copy.deepcopy(root)

        # loop over each element in the original XML file and modify the relevant object
        for elem in new_root.iter():
            if elem.attrib.get('id') == obj_id:
                for child in elem.findall('transform'):
                    if child.attrib.get('name') == 'to_world':
                        for translate in child.findall('translate'):
                            translate.set('value', ' '.join(str(c) for c in coord))
            # new_root.append(elem)

        # write out the modified XML file
        new_tree = ET.ElementTree(new_root)
        num = str(i).zfill(7)
        new_tree.write(os.path.join(f'{path_to_xmls}', f'mappings_{num}.xml'))

    if show: # TODO fix + clean this?
        new_root = copy.deepcopy(root)
        for i, coord in enumerate(coords):
            # make a copy of the root element for the new XML file

            # loop over each element in the original XML file and modify the relevant object
            for elem in new_root.iter():
                if elem.attrib.get('id') == obj_id:
                    for child in elem.findall('transform'):
                        if child.attrib.get('name') == 'to_world':
                            for translate in child.findall('translate'):
                                new_elem = copy.deepcopy(elem)  # Make a copy of elem
                                new_translate = copy.deepcopy(translate)
                                new_translate.set('value', ' '.join(str(c) for c in coord))
                                new_obj_id = f"{obj_id}_{i}"
                                new_elem.attrib['id'] = new_obj_id
                                new_elem.append(new_translate)
                                new_root.append(new_elem)
                                break
        # write out the modified XML file
        new_tree = ET.ElementTree(new_root)
        new_tree.write(os.path.join(f'{path_to_xmls}', 'see_full_range.xml'))
        scene = mi.load_file(os.path.join(f'{path_to_xmls}', 'see_full_range.xml'))
        im = mi.render(scene,spp=128)
        plt.imshow(np.array(im))
        plt.savefig(os.path.join(f'{path_to_xmls}', 'see_full_range.png'))
        plt.show()
        print(f"Sucessfully finished generating files! Entering main for loop...")


def generate_lights_on_versions(file_start,path_to_xmls,xml_string):
    # Write corresponding files that show illuminated scene with walls and red dot visible (note: ~doubles runtime of file generation, avoid unless debugging)

    # Define the XML code to be added
    xml_code = xml_string
    # Loop through all files in the current directory that start with "test"
    for filename in os.listdir(path_to_xmls):
        if filename.startswith(file_start) and filename.endswith('.xml'):
            # Copy the file and rename it with "_LO" added to its name
            new_filename = filename.replace('.xml', '_LO.xml')
            path_to_new_file = os.path.join(path_to_xmls,new_filename)
            shutil.copyfile(os.path.join(path_to_xmls,filename), path_to_new_file)

            # Parse the copied XML file
            tree = ET.parse(path_to_new_file)
            root = tree.getroot()

            # Add the XML code to the root element
            root.append(ET.fromstring(xml_code))

            # Write the updated XML to the copied file
            tree.write(path_to_new_file)

def append_xmlstrings(fileglob, path_to_xmls, xml_strings):
    # Iterate over the files that match the criteria
    start = time.time()
    for i, filename in enumerate(glob.glob(os.path.join(path_to_xmls, fileglob))):
        # Load the existing XML
        tree = ET.parse(filename)
        root = tree.getroot()
        for xml_string in xml_strings:
            # Append the new XML string
            root.append(ET.fromstring(xml_string))

            # Write the updated XML to the file
            tree.write(filename)
        print(f'Processed file {filename}  -- {i}: {(time.time() - start)/(i + 1)}')


def format_file_additions():
    xml_string_light = '''
        <!--     Light -->
            <shape type="obj" id="light">
                <string name="filename" value="meshes/cbox_luminaire.obj"/>
                <transform name="to_world">
                    <translate x="0" y="-0.01" z="0"/>
                </transform>
                <ref id="white"/>
                <emitter type="area">
                    <rgb name="radiance" value="18.387, 13.9873, 6.75357"/>
                </emitter>
            </shape>
        '''
    xml_string_walls_1 = '''
        <!-- Walls -->
        <shape type="obj" id="floor">
        <string name="filename" value="meshes/cbox_floor.obj"/>
        <ref id="brown"/>
        </shape> 
        '''
    xml_string_walls_2 = '''    
        <shape type="obj" id="ceiling">
        <string name="filename" value="meshes/cbox_ceiling.obj"/>
        <ref id="brown"/>
        </shape> 
        '''

    xml_string_walls_3 = '''
            <shape type="obj" id="back">
                <string name="filename" value="meshes/cbox_back.obj" />
                <ref id="brown" />
            </shape>
    '''

    xml_string_walls_4 = '''
            <shape type="obj" id="greenwall">
                <string name="filename" value="meshes/cbox_greenwall.obj" />
                <ref id="brown" />
            </shape>
    '''

    xml_string_walls_5 = '''
            <shape type="obj" id="redwall">
                <string name="filename" value="meshes/cbox_redwall.obj" />
                <ref id="brown" />
            </shape>
    '''

    xml_string_light_point = '''
        <emitter type="point">
            <spectrum name="intensity" value="3"/> <!-- Adjust the intensity value -->
            <point name="position" x="0" y="0" z="1.0"/>
        </emitter>
    '''
    xml_strings_all_walls = [xml_string_walls_1, xml_string_walls_2, xml_string_walls_3, xml_string_walls_4, xml_string_walls_5]

    return xml_string_light_point, xml_strings_all_walls


def top_level_generation(xml_starter, out_path, folder, xrange,yrange,zrange, n, refresh=True):
    path_to_folder = os.path.join(out_path, folder)
    path_to_xmls = os.path.join(path_to_folder, 'XMLs')
    print("XML PATH and folder path: ", path_to_xmls, path_to_folder, flush=True)
    print(f"Refresh={refresh}", flush=True)

    num = 1
    tried_once = False
    current_path_to_folder = path_to_folder
    while os.path.exists(path_to_folder):
        if refresh:
            print(f"WARNING: Folder '{folder}' already exists, will create new folder with slightly modified name...", flush=True)
        current_folder = folder
        try:
            folder = folder.split('_')[0] + '_' + str(int(folder.split('_')[1][0]) + 1) + folder[5:]
        except:
            if tried_once:
                folder = folder[:-1] + str(num)
            else:
                folder = folder + f"_{num}"
                tried_once = True
        num += 1
        current_path_to_folder = path_to_folder
        path_to_folder = os.path.join(out_path, folder)
        print(f"Trying '{folder}' instead of {current_folder}.", flush=True)

    if not refresh and os.path.exists(current_path_to_folder):
        path_to_xmls = os.path.join(current_path_to_folder, 'XMLs')
        print("No refresh --- using existing folder path: ", path_to_xmls, current_path_to_folder, flush=True)
        return current_path_to_folder, path_to_xmls

    os.makedirs(path_to_folder)
    path_to_xmls = os.path.join(path_to_folder, 'XMLs')
    os.makedirs(path_to_xmls)
    path_to_renderings = os.path.join(path_to_folder, 'renderings')
    os.makedirs(path_to_renderings)
    print(f"Folder '{folder}' created successfully!", flush=True)

    generate_files([xrange, yrange, zrange], xml_starter, path_to_xmls, n=n)
    # xml_string_light_point, xml_strings_all_walls = format_file_additions()
    # generate_lights_on_versions(fstart, path_to_xmls, xml_string_light_point)
    # append_xmlstrings(f'{fstart}*LO.xml', path_to_xmls, xml_strings_all_walls)

    return path_to_folder, path_to_xmls
