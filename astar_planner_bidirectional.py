#!/usr/bin/env python3

import rospy
import numpy as np
import tf

from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Quaternion, Pose, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path, Odometry
from visualization_msgs.msg import Marker


class Node_end():
    """
    A node class for A* Pathfinding, stores nodes for searching from end point

        @parameter parent: parent node
        @parameter position: position on map
        @parameter g: cost from start position to current position
        @parameter h: heuristic cost from current position to goal
        @parameter f: sum of g and h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class Node_start():
    """
    A node class for A* Pathfinding, stores nodes for searching from start point

        @parameter parent: parent node
        @parameter position: position on map
        @parameter g: cost from start position to current position
        @parameter h: heuristic cost from current position to goal
        @parameter f: sum of g and h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class Bidirectional_Astar_Planner():
    """
    Independent Astar_Planner function class, which can find a path from start point to end point
    """

    def check_obstacle(self, start, end):
        """
        This function is used to check if there is an obstacle between start point and end point

            @parameter start: position of start point
            @parameter end: position of end point

            @return True: if there is an obstacle between start and end
            @return False: if there is no obstacle between start and end
        """

        # get the difference between start and end in x,y axis
        disx = -(start[0] - end[0])
        disy = -(start[1] - end[1])

        # The circumstance that difference in x axis is bigger
        if abs(disx) > abs(disy):

            # disx is larger than 0
            if disx > 0:
                for i in range(disx):
                    x = start[0] + i
                    y = int(start[1] + i * disy / disx)
                    if self.map[x][y] > 50:
                        return True
                return False

            # disx is smaller than 0
            else:
                for i in range(-disx):
                    x = start[0] - i
                    y = int(start[1] + i * disy / (-disx))
                    if self.map[x][y] > 50:
                        return True
                return False

        # The circumstance that difference in y axis is bigger
        else:

            # disy is larger than 0
            if disy > 0:
                for i in range(disy):
                    x = int(start[0] + i * disx / disy)
                    y = start[1] + i
                    if self.map[x][y] > 50:
                        return True
                return False

            # disy is smaller than 0
            else:
                for i in range(-disy):
                    x = int(start[0] + i * disx / (-disy))
                    y = start[1] - i
                    if self.map[x][y] > 50:
                        return True
                return False

    def get_key_point(self, path):
        """
        This function is used to delete non-neccessary point in path

            @parameter path: path represented by points

            @return: path with only key point
        """
        # set new path begin at path[0]
        new_path = [path[0]]

        # determine if the moving direction changes, if it does not change, delete the point in middle
        length_path = len(path)
        for i in range(2, length_path - 1):
            vector1 = (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
            vector2 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            if vector1 != vector2:
                new_path.append(path[i - 1])

        # at last, add the last element in path to new path
        new_path.append(path[length_path - 1])
        return new_path

    def Path_smoothing(self, path):
        """
        This is a function using floyed method to smooth path. To make a path looks more realistic.

            @parameter path: path represented by points

            @return path: smoothed path
        """
        # First merge nodes that the direction do not change, keep key nodes only
        path = self.get_key_point(path)

        # Second using Floyed method to smooth path
        l = len(path)
        i = 0

        # if the path only contains two key points, return the path
        if l == 2:
            return path

        # apply path smoothing function
        while True:
            while not self.check_obstacle(path[i], path[i + 2]):
                path.pop(i + 1)
                l = len(path)
                if i == l - 2:
                    break
            i += 1
            if i > l - 3:
                break
        return path

    def Path_augment(self, path):
        """
        This is a function to augment path which consists of only key points to dense path

            @patameter path: path represented by points

            @return path: dense path consists of continuous points
        """
        # set a new path
        new_path = []

        # main function of path augment
        length = len(path)
        i = 0
        while True:

            # break rule
            if i == length - 1:
                break

            # difference in x,y axis
            disx = -(path[i][0] - path[i + 1][0])
            disy = -(path[i][1] - path[i + 1][1])

            # if the two key points can directly connected, then pass
            if abs(disy) == 1 and abs(disy) == 1:
                pass

            # if there must be other grids between two key points
            # The circumstance that difference in x axis is bigger
            if abs(disx) > abs(disy):
                # disx is larger than 0
                if disx > 0:
                    for j in range(disx):
                        x = path[i][0] + j
                        y = int(path[i][1] + j * disy / disx)
                        new_path.append((x, y))
                # disx is smaller than 0
                else:
                    for j in range(-disx):
                        x = path[i][0] - j
                        y = int(path[i][1] + j * disy / (-disx))
                        new_path.append((x, y))
            # The circumstance that difference in y axis is bigger
            else:
                # disy is larger than 0
                if disy > 0:
                    for j in range(disy):
                        x = int(path[i][0] + j * disx / disy)
                        y = path[i][1] + j
                        new_path.append((x, y))
                # disy is smaller than 0
                else:
                    for j in range(-disy):
                        x = int(path[i][0] + j * disx / (-disy))
                        y = path[i][1] - j
                        new_path.append((x, y))
            i += 1
        return new_path

    def check_direction(self, node_child, node_parent):
        """
        check the direction of next step

            @parameter node_child: current node
            @parameter node_parent: previous node

            @return 0: if the direction does not change(the same as previous step)
            @return 1: if the direction changes(differ from previous step)
        """
        node_grand = node_parent.parent
        if not node_grand:
            return 0
        vector1 = (node_child.position[0] - node_parent.position[0], node_child.position[1] - node_parent.position[1])
        vector2 = (node_parent.position[0] - node_grand.position[0], node_parent.position[1] - node_grand.position[1])
        if vector1 == vector2:
            return 0
        return 5

    def getMinNode(self, input_list):
        """
        try to find the node with minimal f in openlist

            @parameter input_list: node list

            @return currentNode: the node with minimal f value
        """
        currentNode = input_list[0]
        for node in input_list:
            if node.f < currentNode.f:
                currentNode = node
        return currentNode

    def pointInCloseList(self, position, closed_list):
        """
        determine if a position is in closelist

            @parameter position: the position of points you want check
            @parameter closed_list: closed_list of points

            @return True: if the position is in closelist
            @return False: if the position is not in closelist
        """
        for node in closed_list:
            if node.position == position:
                return True
        return False

    def pointInOpenList(self, position, open_list):
        """
        determine if a position is in openlist

            @parameter position: the position of points you want check
            @parameter open_list: open_list of points

            @return True: if the position is in openlist
            @return False: if the position is not in openlist
        """
        for node in open_list:
            if node.position == position:
                return node
        return None

    def check_intersection(self, open_start, open_end):
        """
        find intersection part of two openlist

            @parameter open_start: openlist from the start point
            @parameter open_end: openlist from the end point

            @return: the intersection part of two nodelist
        """
        for node in open_start:
            append = self.pointInOpenList(node.position, open_end)
            if append:
                self.intersect.append(append.position)
        return self.intersect

    def search_start(self, minF, offsetX, offsetY):
        """
        search action for next step and add this node to openlist

            @parameter minF: currentNode with minimal f value in openlist
            @parameter offsetX: the offset in x direction
            @parameter offsetY: the offset in y direction

            @return: add the node for next step to openlist
        """

        node_pos = (minF.position[0] + offsetX, minF.position[1] + offsetY)

        # if the offset is out of boundary
        if node_pos[0] > self.map_width - 1 or node_pos[0] < 0 or node_pos[1] > self.map_height - 1 or node_pos[1] < 0:
            return

        # if the offset is valid
        elif self.map[node_pos[0]][node_pos[1]] > 98:
            return

        # if the node is in closed set, then pass
        elif self.pointInCloseList(node_pos, self.closed_list_start):
            return

        else:
            # if it is not in openlist, add it to openlist
            currentNode = self.pointInOpenList(node_pos, self.open_list_start)
            if not currentNode:
                currentNode = Node_start(minF, node_pos)
                currentNode.g = minF.g + np.sqrt(offsetX * offsetX + offsetY * offsetY)
                dx = abs(node_pos[0] - self.endnode.position[0])
                dy = abs(node_pos[1] - self.endnode.position[1])
                turn_cost = self.check_direction(currentNode, minF)
                # closed-form distance
                # currentNode.h =  dx + dy + (np.sqrt(2) - 2) * min(dx, dy) + self.map[node_pos[0]][node_pos[1]]
                # euclidean distance
                currentNode.h = dx + dy + self.map[node_pos[0]][node_pos[1]] * 0.9 + turn_cost
                # real distance
                # currentNode.h =  np.sqrt(dx * dx + dy * dy) + self.map[node_pos[0]][node_pos[1]]
                currentNode.f = currentNode.g + currentNode.h
                self.open_list_start.append(currentNode)
                return
            # if it is in openlist, determine if g of currentnode is smaller
            else:
                action_cost = np.sqrt(offsetX * offsetX + offsetY * offsetY)
                if minF.g + action_cost < currentNode.g:
                    currentNode.g = minF.g + action_cost
                    currentNode.parent = minF
                    return

    def search_end(self, minF, offsetX, offsetY):
        """
        search action for next step and add this node to openlist

            @parameter minF: currentNode with minimal f value in openlist
            @parameter offsetX: the offset in x direction
            @parameter offsetY: the offset in y direction

            @return: add the node for next step to openlist
        """

        node_pos = (minF.position[0] + offsetX, minF.position[1] + offsetY)

        # if the offset is out of boundary
        if node_pos[0] > self.map_width - 1 or node_pos[0] < 0 or node_pos[1] > self.map_height - 1 or node_pos[1] < 0:
            return

        # if the offset is valid
        elif self.map[node_pos[0]][node_pos[1]] > 98:
            return

        # if the node is in closed set, then pass
        elif self.pointInCloseList(node_pos, self.closed_list_end):
            return

        else:
            # if it is not in openlist, add it to openlist
            currentNode = self.pointInOpenList(node_pos, self.open_list_end)
            if not currentNode:
                currentNode = Node_end(minF, node_pos)
                currentNode.g = minF.g + np.sqrt(offsetX * offsetX + offsetY * offsetY)
                dx = abs(node_pos[0] - self.startnode.position[0])
                dy = abs(node_pos[1] - self.startnode.position[1])
                turn_cost = self.check_direction(currentNode, minF)
                # closed-form distance
                # currentNode.h =  dx + dy + (np.sqrt(2) - 2) * min(dx, dy) + self.map[node_pos[0]][node_pos[1]]
                # euclidean distance
                currentNode.h = dx + dy + self.map[node_pos[0]][node_pos[1]] * 0.9 + turn_cost
                # real distance
                # currentNode.h =  np.sqrt(dx * dx + dy * dy) + self.map[node_pos[0]][node_pos[1]]
                currentNode.f = currentNode.g + currentNode.h
                self.open_list_end.append(currentNode)
                return
            # if it is in openlist, determine if g of currentnode is smaller
            else:
                action_cost = np.sqrt(offsetX * offsetX + offsetY * offsetY)
                if minF.g + action_cost < currentNode.g:
                    currentNode.g = minF.g + action_cost
                    currentNode.parent = minF
                    return

    def bi_astar(self, gridmap, map_width, map_height, start, end):
        """
        main function of astar search

            @parameter gridmap: the input map
            @parameter map_width: the width of input map
            @parameter map_height: the height of input map
            @parameter start: start point's position
            @parameter end: end point's position

            @return: a global path
        """

        # Initialize endnode and startnode
        self.startnode = Node_start(None, start)
        self.startnode.g = self.startnode.h = self.startnode.f = 0
        self.endnode = Node_end(None, end)
        self.endnode.g = self.endnode.h = self.endnode.f = 0
        self.map = gridmap
        self.map_width = map_width
        self.map_height = map_height

        # Initialize open and closed list
        self.open_list_start = [self.startnode]  # store f of next possible step
        self.closed_list_start = []  # store f of minimal path
        self.open_list_end = [self.endnode]
        self.closed_list_end = []
        self.intersect = []
        start = True

        # try to find the path with minimal cost
        while True:

            # check if open list is empty
            length_open_list_start = len(self.open_list_start)
            length_open_list_end = len(self.open_list_end)
            if length_open_list_start == 0 or length_open_list_end == 0:
                path = []
                return path

            # find the node with minimal f in openlist
            minF_start = self.getMinNode(self.open_list_start)
            minF_end = self.getMinNode(self.open_list_end)

            # add this node to closed_list and delete this node from open_list
            self.closed_list_start.append(minF_start)
            self.open_list_start.remove(minF_start)
            self.closed_list_end.append(minF_end)
            self.open_list_end.remove(minF_end)

            # apply search to add node for next step in 8 directions
            self.search_end(minF_end, 0, 1)
            self.search_end(minF_end, 1, 0)
            self.search_end(minF_end, 0, -1)
            self.search_end(minF_end, -1, 0)
            self.search_end(minF_end, 1, 1)
            self.search_end(minF_end, 1, -1)
            self.search_end(minF_end, -1, 1)
            self.search_end(minF_end, -1, -1)

            self.search_start(minF_start, 0, 1)
            self.search_start(minF_start, 1, 0)
            self.search_start(minF_start, 0, -1)
            self.search_start(minF_start, -1, 0)
            self.search_start(minF_start, 1, 1)
            self.search_start(minF_start, 1, -1)
            self.search_start(minF_start, -1, 1)
            self.search_start(minF_start, -1, -1)

            self.intersect = self.check_intersection(self.open_list_start, self.open_list_end)
            if self.intersect:
                # get the intersection position with minimal f value
                minpos = self.intersect[0]
                current_f = self.pointInOpenList(minpos, self.open_list_start).f + self.pointInOpenList(minpos,
                                                                                                        self.open_list_end).f
                for pos in self.intersect:
                    node_start = self.pointInOpenList(pos, self.open_list_start)
                    node_end = self.pointInOpenList(pos, self.open_list_end)
                    f = node_start.f + node_end.f
                    if f < current_f:
                        current_f = f
                        minpos = pos

                # generate path
                path = []
                current = self.pointInOpenList(minpos, self.open_list_end)
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                path = path[1:]
                path = path[::-1]
                current = node_start
                while current is not None:
                    path.append(current.position)
                    current = current.parent

                # apply path smoothing function
                path = self.Path_smoothing(path)

                # apply path augment function
                path = self.Path_augment(path)

                # return path
                return path[::-1]
