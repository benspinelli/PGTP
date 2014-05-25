bl_info = {
    "name": "PGTP",
    "author": "Ben Spinelli",
    "version": (0, 4, 1),
    "blender": (2, 7, 0),
    "location": "View3D > Add > Mesh",
    "description": ("Adds a tree mesh using a space colonization algorithm."),
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Add Mesh"}

# Copyright 2013 Ben Spinelli

# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or    
# (at your option) any later version.                                  
#                                                                      
# This program is distributed in the hope that it will be useful,      
# but WITHOUT ANY WARRANTY; without even the implied warranty of       
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        
# GNU General Public License for more details.                         
#                                                                      
# You should have received a copy of the GNU General Public License    
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bpy
import bmesh
import mathutils
import math

# Main function. Parameters are passed to it in a dictionary
def add_tree(parameters):
    mesh = bpy.data.objects[parameters['mesh']]
    p = parameters

    # Generate random points
    points = getPoints(p['points'], mesh, p['seed'], False)

    ###########################################
    # For optional custom mesh, un-comment this code
    # (filling in the desired values).
    # Then click Run Script to register the new plugin,
    # which will be accesible from Menu>Add.
    #
##    p['mesh'] = 'CustomMeshNameHere'
##    p['seed'] = 0
##    p['points'] = 1000
##    p['iters'] = 16
##    p['length'] = 0.3
##    p['length2'] = 0.1
##    p['angle'] = 90
##    p['influence'] = 1.0
##    p['kill'] = 0.3
##    p['rootLoc'] = mathutils.Vector((0, 0, 0))
##    p['radius'] = 0.005
##    p['faces'] = 6
##
##    mesh = bpy.data.objects[p['mesh']]
##    points = getPoints(p['points'], mesh, p['seed'], True)
    #
    ###########################################
    #TODO: test this

    # Generate the tree data structure
    root = getTree(points, p['iters'], p['length'], p['length2'], p['angle'],
                   p['influence'], p['kill'], p['rootLoc'])

    # Generate and return the mesh data to Blender to be drawn
    return getMesh(root, p['radius'], p['faces'])

#################################################
# Attraction Point Class 
#################################################


# Used to keep track of the treeNode closed to each attPoint for efficient
# contruction of the influence dictionary and pruning
class AttPoint(object):
    def __init__(self, coord):
        self.coord = coord

    def __repr__(self):
        return "(%f, %f, %f)" % (self.coord[0], self.coord[1], self.coord[2])

    def __hash__(self):
        return hash(repr(self))

    def setNode(self, node):
        self.treeNode = node

    def nodeDist(self, node):
        return distance(self.coord, node.coord)

#################################################
# Random Point Generation
#################################################

# Returns a set of n random points bounded by ob (assumed to have closed mesh)
def getPoints(n, ob, seed, custom=False):
    points = set()

    # Get bounding box (not used for custom mesh, but by calculating it here
    # we only have to calculate it exactly once in all cases)
    bounds = (ob.location[0] + .5 * ob.dimensions[0],
              ob.location[0] - .5 * ob.dimensions[0],
              ob.location[1] + .5 * ob.dimensions[1],
              ob.location[1] - .5 * ob.dimensions[1],
              ob.location[2] + .5 * ob.dimensions[2],
              ob.location[2] - .5 * ob.dimensions[2])

    # Find n points within bounds (mesh or ellipsoid, depending on custom)
    found = 0
    lcgVal = 0
    while found < n:
        point = randomPoint(seed, lcgVal, bounds)
        if (custom and isInMesh(point, ob)) or \
           (not custom and isInEllipsoid(point, bounds)):
            points.add(AttPoint(point))
            found += 1
        # Always increment lcgVal to prevent an infinite loop
        lcgVal += 1
    return points

def randomPoint(seed, n, bounds):
    rand = lcg(seed, n)
    # Use least 10 bits of rand for x, middle 10 for y, and top 11 for z
    # This gives branches 2.1 billion places to start/end
    randx = (0x000003FF & rand) / 0x3FF
    randy = ((0x000FFC00 & rand) >> 10) / 0x3FF
    randz = ((0x7FF00000 & rand) >> 20) / 0x7FF

    x = bounds[0] + randx * (bounds[1] - bounds[0])
    y = bounds[2] + randy * (bounds[3] - bounds[2])
    z = bounds[4] + randz * (bounds[5] - bounds[4])
    return (x, y, z)

# Linear congruential generator (borrows constants from glibc)
def lcg(x, n):
    a = 1103515245
    c = 12345
    m = 2 ** 31

    for iteration in range(n):
        x = (a * x + c) % m
    return x

# Modified version of pointInsideMesh function by Atom and Aothmos
#   http://www.blenderartists.org/forum/showthread.php?195605-Detecting-if-a-
#   point-is-inside-a-mesh-2-5-API
def isInMesh(point, ob):
    point = mathutils.Vector(point)

    # Only need to test on one axis
    axis = mathutils.Vector((1, 0, 0))

    # Ray_cast is in object_space:  http://www.blender.org/documentation/
    # 250PythonDoc/bpy.types.Object.html#bpy.types.Object.ray_cast
    mat = mathutils.Matrix(ob.matrix_world)
    mat.invert()
    orig = mat * point
    count = 0
    loop = True
    while loop:
        location, normal, index = ob.ray_cast(orig, orig + axis * 10000.0)
        if index == -1:
            loop = False
        else:
            count += 1
            orig = location + axis * 0.00001
    if count % 2 == 0:
        return False
    return True

# Tests if (x, y, z) is in or on the largest ellipsoid within the bounds
# This is used as the default bound if a custom mesh is not specified
def isInEllipsoid(point, bounds):
    (x, y, z) = point
    return 1 >= ((x ** 2 / bounds[0] ** 2) + (y ** 2 / bounds[2] ** 2) +
                 (z ** 2 / bounds[4] ** 2))

#################################################
# Node Class
#################################################


# Basic graph (tree) node with a 3D coordinate, one parent, and children
# Nodes are additionally labled with the radius of the tree at that point
class Node(object):
    def __init__(self, coord, parent, children=[]):
        self.coord = coord
        self.parent = parent
        self.children = children
        self.radius = None

    def __repr__(self):
    # Not a true repr, but simplified to avoid recursive listing of nodes
        return "<%f, %f, %f>" % (self.coord[0], self.coord[1], self.coord[2])

    def __eq__(self, other):
    # Equality is defined as pairwise equality of all fields
        if type(other) != Node or self.coord != other.coord or \
           self.parent != other.parent or self.children != other.children or \
           self.radius != other.radius:
            return False
        return True

    def __hash__(self):
        return hash(repr(self))

    def addChildren(self, children):
        newChildren = []
        for child in children:
            new = Node(child, self, [])
            newChildren.append(new)
            self.children.append(new)
        return newChildren

    def vector(self):
    # Returns the mathutils vector from parent to self
    # The vector from the root is defined to be unit along the z-axis
        if self.parent is None:
            return mathutils.Vector((0, 0, 1))

        x = self.coord[0] - self.parent.coord[0]
        y = self.coord[1] - self.parent.coord[1]
        z = self.coord[2] - self.parent.coord[2]
        return mathutils.Vector((x, y, z))

    def setRadius(self, default):
    # A nodes radius is set such that its cross-sectional area is equal to the
    # sum of the cross-sectional areas of its children.
        if self.children == []:
            self.radius = default
        else:
            area = 0
            for child in self.children:
                if child.radius is None:
                    child.setRadius(default)
                area += math.pi * (child.radius ** 2)
            self.radius = math.sqrt(area / math.pi)

    def childrenSet(self):
    # Returns set of self and all descendants
        nodes = set([self])
        for child in self.children:
            nodes = nodes.union(child.childrenSet())
        return nodes

    def childCoordSet(self):
    # Returns a set of own coord and all descendants coords
        children = self.childrenSet()
        coords = set()
        for child in children:
            coords.add(child.coord)
        return coords

#################################################
# Tree Generation - Space Colonization Algorithm
#################################################

# Generates the tree data structure - see interface section for description
# of variables
def getTree(attPoints, iters, length, length2, angle, influence, kill, rootLoc):
    root = Node((rootLoc.x, rootLoc.y, rootLoc.z), None, [])
    treeNodes = set([root.coord])
    # Now that we've created the root, set it as closest node to each attPoint
    for attPoint in attPoints:
        attPoint.setNode(root)

    iter = 0
    while iter < iters and attPoints != set():
        curLength = length + (length2 - length) * (iter / (iters - 1))
        attPoints = iterTree(attPoints, curLength, angle, influence, kill, root)
        iter += 1
    return root

# Iterates the tree through one full step of the space-colonization algorithm
def iterTree(attPoints, length, angle, influence, kill, root):
    # Calculate current nodes and their coordinates
    treeNodes = root.childrenSet()
    treePoints = root.childCoordSet()

    # Calculate the new nodes and find out where they were added
    getNewPoints(getInfluences, attPoints, treeNodes, influence, length, angle)

    # Prune the attraction points
    attPoints = prunePoints(attPoints, kill)
    return attPoints

# Calculates and returns the set of new node coordinates, adding the nodes
# to the tree structure in the process
def getNewPoints(inflFunc, attPoints, treeNodes, influence, length, angle):
    nodeInfluences = inflFunc(attPoints, treeNodes, influence)
    for node in nodeInfluences:
        normal = getNormVector(node.coord, nodeInfluences[node])

        # Check angle and add new node
        if not (normal == mathutils.Vector((0, 0, 0)) or
                vectorAngle(normal, node.vector()) > angle):
            coord = mathutils.Vector(node.coord) + normal * length
            new = node.addChildren([(coord.x, coord.y, coord.z)])
            # Update attPoints' closest node
            for attPoint in attPoints:
                for child in new:
                    if attPoint.nodeDist(child) < \
                       attPoint.nodeDist(attPoint.treeNode):
                        attPoint.setNode(child)
    return None

# Calculates the average of the normalized vectors beween coord and each
# member of attPoints
def getNormVector(coord, attPoints):
    node = mathutils.Vector(coord)
    vector = mathutils.Vector((0, 0, 0))
    for point in attPoints:
        v = mathutils.Vector(point.coord) - node
        v.normalize()
        vector += v
    vector.normalize()
    return vector

# Returns the smallest angle (in degrees) between vectors u and v
def vectorAngle(u, v):
    if u.magnitude == 0 or v.magnitude == 0:
        return 0
    # Deal with rounding error to prevent domain issues with acos
    quot = (u * v) / (u.magnitude * v.magnitude)
    if quot <= -1 or quot >= 1:
        return 0
    return math.degrees(math.acos((u * v) / (u.magnitude * v.magnitude)))

# Returns a dictionary mapping each tree node to the set of points in
# attPoints that are closer to it than any other tree node and within a
# distance influence of it
def getInfluences(attPoints, treeNodes, influence):
    influences = dict([(node, set()) for node in treeNodes])
    for point in attPoints:
        closest = point.treeNode 
        closestDist = point.nodeDist(point.treeNode)
        if closestDist < influence:
            influences[closest] = influences[closest].union([point])
    return influences

# Returns the distance between the points (given as triples)
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 +
                     (p1[2] - p2[2]) ** 2)

# Removes all members of attPoints that are within distance kill of a member
# of treePoints
def prunePoints(attPoints, kill):
    prune = []
    for attPoint in attPoints:
        if attPoint.nodeDist(attPoint.treeNode) < kill:
            prune.append(attPoint)
    return attPoints.difference(prune)

#################################################
# Mesh Creation (Vert and Face Calculation)
#################################################

# Takes the tree data structure and returns the verts and faces for its mesh
def getMesh(root, radius, faces):
    cylinders = []

    # Recursively set radius of all nodes
    root.setRadius(radius)
    for node in root.childrenSet():
        for child in node.children:
            # For branches with no children, close the end of the mesh
            if child.children == []:
                top = circle(child.coord, 0, faces)
                bottom = circle(node.coord, node.radius, faces)
                cylinders.append(tube(top, bottom))
            # Otherwise, make a tube
            else:
                top = circle(child.coord, child.radius, faces)
                bottom = circle(node.coord, node. radius, faces)
                cylinders.append(tube(top, bottom))

    verts, faces = combineMeshes(cylinders)
    return verts, faces

# Makes a list of coordinates for a circle outlined by verts number of points
def circle(center, radius, verts):
    (x, y, z) = center
    s = 2 * math.pi / verts
    return [(x + radius * math.cos(n * s), y + radius * math.sin(n * s), z)
            for n in range(verts)]

# Returns the vertices and faces for the tube between two circles
def tube(top, bottom):
    length = len(top)
    verts = bottom + top
    # The verts go in a circle around the bottom then top, so the four verts
    # that make up each face will follow this pattern
    faces = [(n, (n + 1) % length, length + (n + 1) % length, length + n)
             for n in range(length)]
    return verts, faces

# Takes a list of separate verts-faces tuples and combines them into one
def combineMeshes(meshes):
    verts = []
    faces = []
    for mesh in meshes:
        # Add the offset to all faces in this mesh
        offset = len(verts)
        for face in mesh[1]:
            faces.append(tuple([f + offset for f in face]))
        # Add the new verts
            verts.extend(mesh[0])
    return verts, faces

#################################################
# Interface
#################################################

from bpy.props import FloatProperty, IntProperty, BoolProperty
from bpy.props import FloatVectorProperty, StringProperty


class AddTree(bpy.types.Operator):
    '''Add a procedurally generated tree'''
    bl_idname = "mesh.primitive_tree_add"
    bl_label = "Add Tree"
    bl_options = {'REGISTER', 'UNDO'}

    #Tree properties
    boundingMesh = StringProperty(
            name="Canopy Bound Mesh Name",
            description="Name of the mesh that will bound the canopy.",
            default='Cube'
            )

    seed = IntProperty(
            name="Seed",
            description="Seed for attraction point generation.",
            min=0,
            default=0,
            )

    attractionPoints = IntProperty(
            name="Attraction Points",
            description=("Number of Attraction Points - " +
                         "Increase for more branches."),
            min=0, max=32000,
            default=1000,
            )

    iterations = IntProperty(
            name="Iterations",
            description=("Iterations of the algorithm."),
            min=1, max=100,
            default=16,
            )

    branchLength = FloatProperty(
            name="Initial Branch Segment Length",
            description="Length of branch segments at base of tree.",
            min=0.01, max=10.0,
            default=0.3,
            )

    branchLengthEnd = FloatProperty(
            name="Final Branch Segment Length",
            description="Length of branch segments at tips of tree.",
            min=0.01, max=10.0,
            default=0.1,
            )

    angleLimit = IntProperty(
            name="Angle Limit",
            description="Maximum angle (degrees) between branch segments.",
            min=0, max=180,
            default=90,
            )

    influenceRadius = FloatProperty(
            name="Influence Radius",
            description=("Max distance at which attraction points " +
                         "influence the branching direction."),
            min=0.01, max=100.0,
            default=1.0,
            )

    killRadius = FloatProperty(
            name="Kill Radius",
            description=("Distance from tree at which attraction " +
                         "points are removed."),
            min=0.01, max=100.0,
            default=0.3,
            )

    rootLocation = FloatVectorProperty(
            name="Root Location",
            description="Location of the base of the tree.",
            subtype='TRANSLATION',
            )

    radius = FloatProperty(
            name="Tip Radius",
            description="Radius of the smallest branches.",
            default=0.005,
            )

    branchFaces = IntProperty(
            name="Branch Faces",
            description=("N-gon used to trace each branch."),
            min=3,
            default=6,
            )

    # Generic transform properties
    view_align = BoolProperty(
            name="Align to View",
            default=False,
            )
    location = FloatVectorProperty(
            name="Location",
            subtype='TRANSLATION',
            )
    rotation = FloatVectorProperty(
            name="Rotation",
            subtype='EULER',
            )

#################################################
# Blender Functions
#################################################

    # Note: This function is a method of the AddTree class
    def execute(self, context):
        args = {"mesh" : self.boundingMesh, "seed" : self.seed,
                "points" : self.attractionPoints, "iters" : self.iterations,
                "length" : self.branchLength, "length2" : self.branchLengthEnd,
                "angle" : self.angleLimit, "influence" : self.influenceRadius,
                "kill" : self.killRadius, "rootLoc" : self.rootLocation,
                "radius" : self.radius, "faces" : self.branchFaces}
        verts_loc, faces = add_tree(args)

        mesh = bpy.data.meshes.new("Tree")
        bm = bmesh.new()

        for v_co in verts_loc:
            bm.verts.new(v_co)
        for f_idx in faces:
            bm.faces.new([bm.verts[i] for i in f_idx])

        bm.to_mesh(mesh)
        mesh.update()

        # Add the mesh as an object into the scene with this utility module
        from bpy_extras import object_utils
        object_utils.object_data_add(context, mesh, operator=self)

        return {'FINISHED'}

def menu_func(self, context):
    self.layout.operator(AddTree.bl_idname, icon='MESH_CUBE')

def register():
    bpy.utils.register_class(AddTree)
    bpy.types.INFO_MT_mesh_add.append(menu_func)

def unregister():
    bpy.utils.unregister_class(AddTree)
    bpy.types.INFO_MT_mesh_add.remove(menu_func)

if __name__ == "__main__":
    register()
