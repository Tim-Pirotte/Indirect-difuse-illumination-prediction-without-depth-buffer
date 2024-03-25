import bpy
from PIL import Image
from datetime import datetime
import random
import math
import os    


def find_image_by_filename(filename):
    """
    Find an image in Blender's data by filename.

    Args:
    - filename (str): The filename to search for.

    Returns:
    - bpy.types.Image or None: The image object if found, else None.
    """
  
    for img in bpy.data.images:
        if img.filepath == filename:
            return img
    return None


def add_lighting():
    """
    Add environmental lighting to the scene. HDRI's are from polyhaven
    """
  
    node_environment = bpy.context.scene.world.node_tree.nodes.get("Environment Texture")
    
    hdri_path = "C:/Users/timpi/Documents/hdri/"
    hdris = ["meadow_4k.exr",
     "poly_haven_studio_4k.exr",
     "brown_photostudio_02_4k.exr",
     "christmas_photo_studio_05_4k.exr",
     "dikhololo_night_4k.exr",
     "farm_sunset_4k.exr",
     "industrial_workshop_foundry_4k.exr",
     "kloofendal_48d_partly_cloudy_puresky_4k.exr",
     "little_paris_eiffel_tower_4k.exr",
     "misty_farm_road_4k.exr",
     "netball_court_4k.exr",
     "spree_bank_4k.exr",
     "wrestling_gym_4k.exr"]
     
    selected_hdri_filename = random.choice(hdris)
    selected_hdri_path = os.path.join(hdri_path, selected_hdri_filename)
    
    image = find_image_by_filename(selected_hdri_path)
    
    if not image:
        # Image not found, load it into Blender
        image = bpy.data.images.load(selected_hdri_path)
        
    node_environment.image = image
    return

def change_object_location():
    """
    Randomly change the location and rotation of objects in the scene.
    """
  
    for obj in bpy.context.scene.objects:
        if obj.name == 'Plane':
            if random.random() < 0.3:
                obj.hide_render = True
            else:
                obj.hide_render = False
            continue
        if obj.type != 'CAMERA' and obj.type != 'LIGHT':
            x = random.uniform(-25.0, 25.0)
            y = random.uniform(0.0, 50.0)
            z = random.uniform(0.0, 20.0)

            # Set the object's location
            obj.location = (x, y, z)

            # Generate random rotation angles in radians
            rot_x = random.uniform(0, 2 * math.pi)
            rot_y = random.uniform(0, 2 * math.pi)
            rot_z = random.uniform(0, 2 * math.pi)

            # Set the object's rotation
            obj.rotation_euler = (rot_x, rot_y, rot_z)
        

def change_material():
    """
    Randomly change the materials of objects in the scene.
    """
  
    for obj in bpy.context.scene.objects:
        if obj.type != 'CAMERA' and obj.type != 'LIGHT': 
            
            r = random.uniform(0.0, 1.0)
            g = random.uniform(0.0, 1.0)
            b = random.uniform(0.0, 1.0)
            new_color = (r, g, b, 1.0)
            
            roughness = random.uniform(0.0, 1.0)
            metallic = random.uniform(0.0, 0.95)
            clearcoat = random.uniform(0.0, 1.0)
            transmission = random.uniform(0.0, 1.0)
            
            material = obj.data.materials[0]
            
            bsdf_shader = material.node_tree.nodes["Principled BSDF"]
            
            # Socket 18 is clearcoat
            bsdf_shader.inputs[18].default_value = 0
            
            bsdf_shader.inputs["Base Color"].default_value = new_color
            bsdf_shader.inputs["Roughness"].default_value = roughness
            bsdf_shader.inputs["Metallic"].default_value = metallic
            if random.random() < 0.1:
                bsdf_shader.inputs[18].default_value = clearcoat
            
            if obj.name == 'Plane':
                bsdf_shader.inputs["Roughness"].default_value = 1
                bsdf_shader.inputs[18].default_value = 0
            

def change_focal_length():
    """
    Randomly change the focal length of the active camera in the scene.
    """ 
    active_camera = bpy.context.scene.camera
    active_camera.data.lens = random.uniform(20, 80)
        
        
def render_scene():
    """
    Render the scene using both Eevee and Cycles render engines.
    """
    
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.filepath = os.path.join('path_to_save_folder', 'eevee')
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    for node in tree.nodes:
        if node.type == 'COMPOSITE':
            node.mute = True
            
    bpy.ops.render.render(write_still=True)
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.filepath = os.path.join('path_to_save_folder', 'cycles')
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    for node in tree.nodes:
        if node.type == 'COMPOSITE':
            node.mute = False
            
    bpy.ops.render.render(write_still=True)
    
    
def combine_renders():
    """
    Combine rendered images from Eevee and Cycles into a single image.
    """
    
    image_eevee = Image.open(r'path_to_save_folder\eevee.png')
    image_cycles = Image.open(r'path_to_save_folder\cycles.png')
    
    combined_image = Image.new('RGB', (image_eevee.width + image_cycles.width, image_eevee.height))
    combined_image.paste(image_eevee, (0, 0))
    combined_image.paste(image_cycles, (image_eevee.width, 0))
    
    combined_image.save(r'path_to_save_folder' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png')
    
    return
        
        
number_of_renders = 10000

for n in range(number_of_renders):
    # Loop to generate multiple renders of the scene
    add_lighting()
    change_object_location()
    change_material()
    change_focal_length()
    cleanup_particle_systems()
    render_scene()
    combine_renders()
