import xml.etree.ElementTree as ET
import os


def generate_unwarp_file(path_to_xmls, res, im_file, path_to_ims):
    starter_file = 'unwarp_tmp.xml'
    path_to_og_file = os.path.join(path_to_xmls, starter_file)

    # Read in the original XML file
    tree = ET.parse(path_to_og_file)
    root = tree.getroot()

    # Set the ID of the object to modify
    obj_id = 'im_to_unwarp'

    # Loop over each element in the original XML file and modify the relevant object
    found_flag = False
    for elem in root.findall('.//shape[@id="{}"]'.format(obj_id)):
        print(f"Found element with ID '{obj_id}'")
        for child in elem.findall('.//bsdf[@type="diffuse"]/texture[@type="bitmap"]/string[@name="filename"]'):
            print(f"Before: filename={child.get('value')}")
            child.set('value', os.path.join(path_to_ims,im_file))
            print(f"After: filename={child.get('value')}")
            found_flag = True

    if not found_flag:
        print(f"Element with ID '{obj_id}' not found in the XML.")

    # Write out the modified XML file
    path_to_new_file = os.path.join(path_to_xmls, 'unwarp_tmp.xml')
    tree.write(path_to_new_file)

    print("XML file modified and saved successfully.")