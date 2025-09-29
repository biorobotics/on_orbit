import pickle
import numpy as np
import trimesh
from collision import boundary_volume_hierarchy
import timeit
import random

if __name__=="__main__":
    meshes_path="../../../urdf/meshes"
    mesh=trimesh.load_mesh(meshes_path+"/full_cv.stl")
    print("Loaded mesh")
    with open(meshes_path+"/arrayrsstree_fullcvSTL_np1dot23.pkl","rb") as fh:
        array_rsstree=pickle.load(fh)
    print("Loaded RSS Tree from pickle file")
    print("The first call to a distance compute function is slow because it is compiled JIT. Calling is_distance_lte_array now")
    start=timeit.default_timer()
    close=boundary_volume_hierarchy.is_distance_lte_array(np.array([-1,1,-2.0]),array_rsstree,mesh.triangles,.01)
    end=timeit.default_timer()
    print(f"First call took {end-start} seconds")
    print("Subsequent calls are fast")
    n=100
    start=timeit.default_timer()
    for i in range(n):
        close=boundary_volume_hierarchy.is_distance_lte_array(np.array([random.random()*10,random.random()*10,random.random()*10]),array_rsstree,mesh.triangles,.01)
    end=timeit.default_timer()
    print(f"Next {n} calls took an average of {(end-start)/n} seconds")

    