
from numpy.typing import ArrayLike
from numba import njit
import numpy as onp
from collision.Math import norm,cuda_dot,cuda_vec3_minus

@njit
def stable_triangle_area(vertices:ArrayLike):
    '''
    given the vertices of a triangle, compute its area robustly to numerical problems following a formulat given by W. Kahan in "Miscalculating Area and Angles of a Needle-like Triangle"

    Parameters: vertices : (3,n) float array
                    the vertices of the triangle
    '''
    one=norm(vertices[1]-vertices[0])
    two=norm(vertices[2]-vertices[0])
    three=norm(vertices[2]-vertices[1])
    if one<two:
        a=two
        b=one
    else:
        a=one
        b=two
    if b<three:
        c=b#b may not be two anymore
        b=three
        if a<b:
            temp=a
            a=b
            b=temp
    else:
        c=three
    return stable_triangle_area_from_edges(a,b,c)

@njit
def stable_triangle_area_from_edges(a,b,c):
    '''
    given non-increasing edge lengths a>=b>=c stably compute triangle area
    '''
    return onp.sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4

@njit
def brute_force_point2mesh_cpu(point,triangles):
    tol=1e-12
    return onp.sqrt(min([point_to_triangle_squared_distance(t[0],t[1],t[2],point,tol) for t in triangles]))

@njit
def point_to_triangle_squared_distance(A,B,C,query,tol):
    '''
    given vertices of a triangle and a query point compute the shortest distance
    '''
    #First thing is to check if the query point is closest to vertices A or B
    #check vertex a
    abx,aby,abz=cuda_vec3_minus(B,A)
    acx,acy,acz=cuda_vec3_minus(C,A)
    apx,apy,apz=cuda_vec3_minus(query,A)
    d1=cuda_dot(abx,aby,abz,apx,apy,apz)
    d2=cuda_dot(acx,acy,acz,apx,apy,apz)
    if d1<=tol and d2<=tol:
        #||e||^2=||ap||^2
        return apx*apx+apy*apy+apz*apz
    
    #check vertex b
    bpx,bpy,bpz=cuda_vec3_minus(query,B)
    d3=cuda_dot(abx,aby,abz,bpx,bpy,bpz)
    d4=cuda_dot(acx,acy,acz,bpx,bpy,bpz)
    if d3>=-tol and d4<=d3:
        #||e||^2=||bp||^2
        return bpx*bpx+bpy*bpy+bpz*bpz
    
    #values computed for checking A and B can be used to check if closest to edge AB
    vc=d1*d4-d3*d2
    if vc<=tol and d1>=-tol and d3<=tol:
        #||e||^2=||ap||^2-v^2||ab||^2 for v s.t. A+vAB is the nearest point
        v=d1/(d1-d3)
        return apx*apx+apy*apy+apz*apz-v*v*(abx*abx+aby*aby+abz*abz)
    
    #now check if closest to vertex C
    cpx,cpy,cpz=cuda_vec3_minus(query,C)
    d5=cuda_dot(abx,aby,abz,cpx,cpy,cpz)
    d6=cuda_dot(acx,acy,acz,cpx,cpy,cpz)
    if d6>=-tol and d5<=d6:
        #||e||^2=||cp||^2
        return cpx*cpx+cpy*cpy+cpz*cpz
    
    #now check if closest to edge AC
    vb=d5*d2-d1*d6
    if vb<=tol and d2>=-tol and d6<=tol:
        #||e||^2=||ap||^2-w^2||ac||^2 for w s.t. A+wAC is the nearest point
        w=d2/(d2-d6)
        return apx*apx+apy*apy+apz*apz-w*w*(acx*acx+acy*acy+acz*acz)
    
    #check if closest to edge BC
    va=d3*d6-d5*d4
    if va<=tol and (d4-d3)>=-tol and (d5-d6)>=-tol:
        #||e||^2=||bp||^2-w^2||bc||^2 for w s.t. B+wBC is the nearest point
        w=(d4-d3)/((d4-d3)+(d5-d6))
        bcx,bcy,bcz=cuda_vec3_minus(C,B)
        return bpx*bpx+bpy*bpy+bpz*bpz-w*w*(bcx*bcx+bcy*bcy+bcz*bcz)
    
    #must be inside the face
    denom=va+vb+vc
    v=vb/denom
    w=vc/denom
    ex=apx-v*abx-w*acx
    ey=apy-v*aby-w*acy
    ez=apz-v*abz-w*acz
    return ex*ex+ey*ey+ez*ez

@njit
def closest_point_on_triangle(A,B,C,query,tol):
    '''
    compute the coordinates of the point in a triangle closest to a single 3D point
    
    Parameters: A : (3,) float array
                    first vertex
                B : (3,) float array
                    second vertex
                C : (3,) float array
                    third vertex
                query : (3,) float array
                    the query point
                tol : float
                    values smaller than this in absolute value are considered 0
    Returns:    x : float
                    first coordinate of closest point
                y : float
                    second coordinate of closest point
                z : float
                    third coordinate of closest point
    
    transcription of ClosestPtPointTriangle from Real-Time Collision Detection by Christer Ericson (2004)
    '''    
    #First thing is to check if the query point is closest to vertices A or B
    #check vertex a
    abx,aby,abz=cuda_vec3_minus(B,A)
    acx,acy,acz=cuda_vec3_minus(C,A)
    apx,apy,apz=cuda_vec3_minus(query,A)
    d1=cuda_dot(abx,aby,abz,apx,apy,apz)
    d2=cuda_dot(acx,acy,acz,apx,apy,apz)
    if d1<=tol and d2<=tol:
        return A[0],A[1],A[2]
    
    #check vertex b
    bpx,bpy,bpz=cuda_vec3_minus(query,B)
    d3=cuda_dot(abx,aby,abz,bpx,bpy,bpz)
    d4=cuda_dot(acx,acy,acz,bpx,bpy,bpz)
    if d3>=-tol and d4<=d3:
        return B[0],B[1],B[2]
    
    #values computed for checking A and B can be used to check if closest to edge AB
    vc=d1*d4-d3*d2
    if vc<=tol and d1>=-tol and d3<=tol:
        v=d1/(d1-d3)
        return A[0]+v*abx,A[1]+v*aby,A[2]+v*abz
    
    #now check if closest to vertex C
    cpx,cpy,cpz=cuda_vec3_minus(query,C)
    d5=cuda_dot(abx,aby,abz,cpx,cpy,cpz)
    d6=cuda_dot(acx,acy,acz,cpx,cpy,cpz)
    if d6>=-tol and d5<=d6:
        return C[0],C[1],C[2]
    
    #now check if closest to edge AC
    vb=d5*d2-d1*d6
    if vb<=tol and d2>=-tol and d6<=tol:
        w=d2/(d2-d6)
        return A[0]+w*acx,A[1]+w*acy,A[2]+w*acz

    #check if closest to edge BC
    va=d3*d6-d5*d4
    if va<=tol and (d4-d3)>=-tol and (d5-d6)>=-tol:
        w=(d4-d3)/((d4-d3)+(d5-d6))
        bcx,bcy,bcz=cuda_vec3_minus(C,B)
        return B[0]+w*bcx,B[1]+w*bcy,B[2]+w*bcz
    
    #must be inside the face
    denom=va+vb+vc
    v=vb/denom
    w=vc/denom
    return A[0]+v*abx+w*acx,A[1]+v*aby+w*acy,A[2]+v*abz+w*acz