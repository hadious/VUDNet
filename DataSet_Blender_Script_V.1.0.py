# DataSet generation: creating 2D_images and corresponding 3D-point-cloud. V.1.0

import bpy
import numpy as np
from mathutils import Matrix
from bpy_extras.object_utils import world_to_camera_view
#import bmesh
#from bmesh.ops import spin
#import math

'''
array1=np.loadtxt("Z:\Personal\Claire\CWCCalibration\createMeshBlenderPython_23092022\outFDP1.csv",delimiter=",",dtype=float)
z=array1[:,0]
r=array1[:,1]
#create vertices list 
vertices=[]
edges=[]
for ctr1 in range(0,array1.shape[0]):
    newVert=[r[ctr1],0,z[ctr1]]
    vertices.append(newVert)
for ctr1 in range(0,array1.shape[0]-1):
    newEdge=[ctr1,ctr1+1]
    edges.append(newEdge)
    
faces=[]
new_mesh = bpy.data.meshes.new('new_mesh')
new_mesh.from_pydata(vertices,edges,faces)
new_mesh.update()
new_object=bpy.data.objects.new('new_object',new_mesh)
new_collection=bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(new_collection)
new_collection.objects.link(new_object)
#get r for first frame, set up initial mesh
verts=new_object.data.vertices
sk_basis=new_object.shape_key_add(name='Basis',from_mix=False)
sk_basis.interpolation='KEY_LINEAR'
new_object.data.shape_keys.use_relative = True

prevKey='Basis'
me=new_object.data
me.shape_keys.key_blocks[prevKey].value=1
me.shape_keys.key_blocks[prevKey].keyframe_insert(data_path="value",frame=0)
me.shape_keys.key_blocks[prevKey].value=0
me.shape_keys.key_blocks[prevKey].keyframe_insert(data_path="value",frame=1)


    
for ctr in range(1,array1.shape[1]):

    #create new shape key
    keyName='Deform'+str(ctr)
    sk=new_object.shape_key_add(name=keyName,from_mix=False)
    sk.interpolation='KEY_LINEAR'
    #set each successive set of vertices to a new shape key
    for i in range(len(verts)):
        sk.data[i].co.x*=array1[i,ctr]
    me.shape_keys.key_blocks[keyName].value=0
    me.shape_keys.key_blocks[keyName].keyframe_insert(data_path="value",frame=0)
    
for ctr in range(1,array1.shape[1]-1):
    #set this shape key to be a magnitude of 1 in this frame
    keyName='Deform'+str(ctr)
    nextKeyName='Deform'+str(ctr+1)
    me.shape_keys.key_blocks[keyName].value=1
    me.shape_keys.key_blocks[prevKey].value=0
    me.shape_keys.key_blocks[nextKeyName].value=0
    me.shape_keys.key_blocks[keyName].keyframe_insert(data_path="value",frame=ctr)
    me.shape_keys.key_blocks[prevKey].keyframe_insert(data_path="value",frame=ctr)
    me.shape_keys.key_blocks[nextKeyName].keyframe_insert(data_path="value",frame=ctr)
        
    #store name of this key
    prevKey=str(keyName)

'''
#Added By Hadi
#################################################################################################

# camera movement to provide 2d images:
#____________________________________________________________________________________
# Set the number of frames and the output directory
'''
num_frames = 150
output_directory = "Desktop/GF/DataSet/2D_Images"

# Get the existing camera object
cam_obj = bpy.data.objects["Center Side View"]

# Loop through the frames and set keyframes for the camera's location and rotation
for i in range(num_frames):
    # Set the current frame
    bpy.context.scene.frame_set(i)
    
    # Set the camera's location and rotation ?!
    cam_obj.location = (0, 0, i)
    cam_obj.rotation_euler = (0, 0, 0)
    
    # Add keyframes for the camera's location and rotation
    cam_obj.keyframe_insert(data_path="location", index=-1)
    cam_obj.keyframe_insert(data_path="rotation_euler", index=-1)
    
    # Render the image with the point cloud
    bpy.context.scene.render.filepath = f"{output_directory}/frame_{i:04d}.png"
    bpy.ops.render.render(write_still=True)
    
'''
#_________________________________________________________________________________

# Building correspondece 3d Cloud point:
#____________________________________________________________________________________

# Set the number of frames and the output directory
'''
num_frames = 150
output_directory = "Desktop/GF/DataSet/CloudPoints"

# Get the active camera object
cam_obj = bpy.context.scene.camera

# Get the scene's camera matrix
cam_mat = cam_obj.matrix_world.inverted()

# Create a list to store the point clouds for each frame
point_clouds = []

# Loop through the frames and set keyframes for the camera's location and rotation
for i in range(num_frames):
    # Set the current frame
    bpy.context.scene.frame_set(i)

    # Get the 3D coordinates of each visible object
    point_cloud = []
    for obj in bpy.context.visible_objects:
        if obj.type == 'MESH':
            # Get the object's vertices in world space
            vertices = [v.co for v in obj.data.vertices]
            vertices = [obj.matrix_world @ v for v in vertices]
            
            # Project the vertices onto the camera image plane
            for v in vertices:
                v_cam = cam_mat @ v
                if v_cam.z > 0:
                    x = int(bpy.context.scene.render.resolution_x * v_cam.x / v_cam.z)
                    y = int(bpy.context.scene.render.resolution_y * v_cam.y / v_cam.z)
                    point_cloud.append([v.x, v.y, v.z])

    # Save the point cloud to a .npy file
    point_cloud = np.array(point_cloud)
    np.save(f"{output_directory}/frame_{i:04d}.npy", point_cloud)
    
    # Add the point cloud to the list
    point_clouds.append(point_cloud)

    # Set the camera's location and rotation
    cam_obj.location = (0, 0, i)
    cam_obj.rotation_euler = (0, 0, 0)
    
    # Add keyframes for the camera's location and rotation
    cam_obj.keyframe_insert(data_path="location", index=-1)
    cam_obj.keyframe_insert(data_path="rotation_euler", index=-1)
    
    # Render the image
    bpy.context.scene.render.filepath = f"{output_directory}/frame_{i:04d}.png"
    bpy.ops.render.render(write_still=True)

# Save the list of point clouds to a single .npy file
np.save(f"{output_directory}/point_clouds.npy", point_clouds)
'''
#____________________________________________________________________________________

# Try#1:
#____________________________________________________________________________________

'''
num_frames = 150

# Set the output file paths
image_path_template = "Desktop/GF/DataSet/2D_Images/image_{:04d}.png"
point_cloud_path_template = "Desktop/GF/DataSet/point_clouds/point_cloud_{:04d}.npy"

# Set the camera location and orientation
bpy.data.objects["Camera"].location = (0, 0, 0)
bpy.data.objects["Camera"].rotation_euler = (0, 0, 0)

# Get the cylinder object and its diameter
cylinder = bpy.data.objects["new_object.002"]
diameter = cylinder.dimensions[0]

# Animate the camera rotation
for i in range(num_frames):
    angle = i * (2 * np.pi / num_frames)
    bpy.data.objects["Camera"].rotation_euler = (0, 0, angle)
    # Set the camera's look-at point to the center of the cylinder
    bpy.data.objects["Empty"].location = (0, 0, 0)
    bpy.data.objects["Camera"].track_to("Empty", "Z", 1)
    bpy.context.scene.render.filepath = image_path_template.format(i)
    bpy.ops.render.render(write_still=True)
    # Save the corresponding 3D point cloud
    num_points = 10000
    points = []
    for j in range(num_points):
        # Generate a random point on the surface of the cylinder
        theta = np.random.uniform(0, 2 * np.pi)
        x = diameter / 2 * np.cos(theta)
        y = diameter / 2 * np.sin(theta)
        z = np.random.uniform(-cylinder.dimensions[2] / 2, cylinder.dimensions[2] / 2)
        points.append((x, y, z))
    point_cloud = np.array(points)
    np.save(point_cloud_path_template.format(i), point_cloud)
'''
#____________________________________________________________________________________

# Try#2:
#____________________________________________________________________________________

'''
# set the camera to a fixed position inside the cylinder
bpy.data.objects['Center Top Side View'].location = (0, 0, 1.5)
bpy.data.objects['Center Top Side View'].rotation_euler = (0, 0, 0)

# set the number of images you want to generate
num_images = 150

# set the fixed angle by which you want to rotate the camera (we can change it according to information overlay then maybe!!!)
angle_step = 2*math.pi/num_images

for i in range(num_images):
    # rotate the camera by the fixed angle around the cylinder
    bpy.data.objects['Center Top Side View'].rotation_euler = (0, 0, i*angle_step)
    
    # render the image
    bpy.ops.render.render()
    
    # save the rendered image as a PNG file
    filepath = 'Desktop/GF/DataSet/2D_Images/' + 'image_' + str(i) + '.png'
    bpy.data.images['Render Result'].save_render(filepath)
    
    # save the visible section of the cylinder as a numpy array
    point_cloud = []
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("new_object.002"):
            mesh = obj.to_mesh(bpy.context.scene, True, 'PREVIEW')
            for vert in mesh.vertices:
                point_cloud.append([vert.co[0], vert.co[1], vert.co[2]])
            bpy.data.meshes.remove(mesh)
    
    point_cloud = np.array(point_cloud)
    np.save('Desktop/GF/DataSet/Point_Cloud/' + 'pointcloud_' + str(i) + '.npy', point_cloud)
#____________________________________________________________________________________
'''
#____________________________________________________________________________________

# Try#3:
#____________________________________________________________________________________
'''
# Set output path and resolution
output_path = "Desktop/GF/DataSet/"
resolution_x = 640 #???
resolution_y = 480 #???

# Set camera and scene
scene = bpy.context.scene
camera = scene.camera

# Set number of frames and angles
num_frames = 150
num_angles = 360 # ???

# Set camera rotation increment
rot_increment = 360.0 / num_angles

for frame in range(num_frames):
    # Set the current frame
    scene.frame_set(frame)
    
    for angle in range(num_angles):
        # Set the current camera rotation
        rot_z = np.deg2rad(angle * rot_increment)
        camera.rotation_euler = (0, 0, rot_z)
        
        # Render the image
        filename = f"frame_{frame}_angle_{angle}.png"
        bpy.context.scene.render.filepath = output_path + filename
        bpy.ops.render.render(write_still=True)
        
        # Get the corresponding 3D point cloud
        camera_matrix = camera.matrix_world.inverted()
        coords_3d = [(camera_matrix @ v.co)[:3] for v in bpy.context.active_object.data.vertices]
        coords_3d = np.array(coords_3d)

        # Save the 3D point cloud to a .npy file
        np.save(output_path + f"frame_{frame}_angle_{angle}.npy", coords_3d)

#____________________________________________________________________________________
'''
'''
output_file = "output.npy"

# Get the active camera object
camera = bpy.context.scene.camera

# Get the dimensions of the rendered image
render_width = bpy.context.scene.render.resolution_x
render_height = bpy.context.scene.render.resolution_y

# Set the number of angles to capture
num_angles = 360

# Set the number of frames to capture
num_frames = 150

# Create an array to store the results
results = np.zeros((num_angles * num_frames, render_height, render_width, 4), dtype=np.uint8)
point_clouds = np.zeros((num_angles * num_frames, render_height * render_width, 3), dtype=np.float32)

# Loop over the frames
for frame in range(num_frames):
    
    # Set the current frame
    bpy.context.scene.frame_set(frame)

    # Loop over the angles
    for angle in range(num_angles):

        # Calculate the camera rotation
        camera.rotation_euler[2] = angle * 2 * np.pi / num_angles

        # Render the image
        bpy.ops.render.render()

        # Get the rendered image as a numpy array
        rendered_image = bpy.data.images['Render Result'].pixels[:] 

        # Convert the image to uint8 format and store it in the results array
        results[frame * num_angles + angle] = (255 * np.array(rendered_image)).astype(np.uint8).reshape(render_height, render_width, 4)

        # Get the 3D coordinates of the visible points
        depth_buffer = bpy.data.images['Viewer Node'].pixels[:]
        point_cloud = []
        for y in range(render_height):
            for x in range(render_width):
                # Calculate the 3D coordinates of the point
                depth = depth_buffer[(render_height - y - 1) * render_width + x]
                clip_coords = np.array([(x - render_width/2)/render_width, (y - render_height/2)/render_height, depth * 2.0 - 1.0, 1.0])
                view_coords = np.linalg.inv(camera.matrix_world) @ clip_coords
                world_coords = np.linalg.inv(bpy.context.scene.render.film_transformation) @ view_coords
                world_coords /= world_coords[3]
                if depth > 0.0:
                    point_cloud.append(world_coords[:3])
        point_clouds[frame * num_angles + angle] = np.array(point_cloud)

# Save the results to a file
np.save(output_file, (results, point_clouds))
'''

# Final Version:
#___________________________________________________________________________________________________
# Set the output directory for the images and point clouds
output_dir = "Desktop/GF/DataSet"

# Set the number of views to capture
num_views = 36 # every 10 degree a rotation ?

# Set the resolution of the images
res_x =  bpy.context.scene.render.resolution_x  #or 640 or ?
res_y =   bpy.context.scene.render.resolution_y #or 480 or ?

# Get the active scene and camera
scene = bpy.context.scene
camera = scene.camera

# Get the total number of frames in the animation
num_frames = scene.frame_end - scene.frame_start + 1 # 150 ?

# Loop through each frame
for frame in range(scene.frame_start, scene.frame_end+1):
    
    # Set the scene to the current frame
    scene.frame_set(frame)
    
    # Calculate the angular step between views
    angular_step = 360.0 / num_views
    
    # Loop through each view
    for view in range(num_views):
        
        # Set the camera rotation for the current view
        camera.rotation_euler[2] = np.radians(view * angular_step)
        
        
        image_filename = output_dir + "/frame{}_view{}.png".format(frame, view)
        bpy.context.scene.render.filepath = image_filename
        bpy.ops.render.render(write_still=True)
        
        # Get the 3D coordinates of the visible vertices
        coords = []
        for obj in scene.objects:
            if obj.type == 'MESH' and obj.hide_render == False:
                mesh = obj.data
                matrix_world = obj.matrix_world
                for vert in mesh.vertices:
                    vert_world = matrix_world @ vert.co
                    vert_cam = world_to_camera_view(scene, camera, vert_world)
                    if (0.0 <= vert_cam.x <= 1.0 and
                        0.0 <= vert_cam.y <= 1.0 and
                        vert_cam.z > 0.0):
                        coords.append(vert_world)
                        
        # Convert the list of coordinates to a numpy array
        coords = np.array(coords)
        
        # Save the 3D coordinates to a .npy file
        np.save(output_dir + "/frame{}_view{}_cloud.npy".format(frame, view), coords)


#___________________________________________________________________________________________________

#################################################################################################
