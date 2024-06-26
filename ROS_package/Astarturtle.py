#!/usr/bin/env python3
# Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
import heapq
import csv
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist



class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.velocities = []  # Initialize empty list for velocities
        self.orientation = []  # Initialize empty list for orientation
        self.index = 0  # Index to keep track of which command to publish
        self.add_on_set_parameters_callback(self.on_shutdown)
        # Note: Don't start the timer here
    def on_shutdown(self):
        print("Shutting down, stopping the TurtleBot.")
        self.stop_turtlebot()

    def stop_turtlebot(self):
        stop_msg = Twist()  # Creating a default Twist message which has linear.x = 0 and angular.z = 0
        self.publisher_.publish(stop_msg)

    def start_moving(self):
        self.timer_period = 1  # seconds
        # Only start the timer after velocities and orientation are set
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        print(f"Publishing at index: {self.index}")
        if self.index < len(self.velocities):
            msg = Twist()
            msg.linear.x = float(self.velocities[self.index])
            # Convert orientation from radians to angular velocity if needed
            msg.angular.z = float(self.orientation[self.index])
            print(f"Linear X: {msg.linear.x}, Angular Z: {msg.angular.z}")
            self.publisher_.publish(msg)
            self.index += 1
        else:
            print("Stopping timer")
            self.stop_turtlebot()
            self.timer.cancel()


# Node class 
class createNode :
    def __init__ (self, pos, theta, parent, cost, euclidean):
        """
        Initialize node attributes.
        pos: Position coordinates (x, y)
        theta: Orientation angle
        parent: Parent node
        cost: Cost to reach this node
        euclidean: Euclidean distance to the goal
        """
        self.pos = pos
        self.theta = theta
        self.parent = parent
        self.cost = cost
        self.euclidean = euclidean
        self.total_cost = cost + euclidean

    # Point of the Node
    def getPos(self):
        return self.pos

    # Orientation of the Node
    def getTheta(self):
        return self.theta

    # Parent Node
    def getParent(self):
        return self.parent

    # # Comparison method for priority queue
    def __lt__(self, other):
        return self.total_cost < other.total_cost

# Define image dimensions
width, height = 6000, 2000

# Define colors in BGR format
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)  # For the clearance
red = (0, 0, 255)  # For the obstacle space
blue = (255, 0, 0)

#Robot Parameters
robot_radius = 38
wheel_distance = 356
dt = 0.3

clearance = 200

# Create a blank black image
screen = np.zeros((2000, 6000, 3), dtype=np.uint8)

# Function to draw the environment with obstacles and clearance areas
def draw_scene(clearance):
    """
    Draw the environment with obstacles, clearance areas, and boundaries.
    """
    # Draw the rectangle with clearance
    clearance_rectangles = [
        (1500 - clearance, 0 - clearance, 250 + (2 * clearance), 1000 + (2 * clearance)),  # Left vertical
        (2500 - clearance, 1000 - clearance, 250 + (2 * clearance), 1000 + (2 * clearance)),  # Right vertical
    ]
    for rect in clearance_rectangles:
        cv2.rectangle(screen, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), green,
                      -1)

    # Draw circle with clearance
    cv2.circle(screen, (4200, 800), 600 + clearance, green, -1)

    # Draw the rectangle without clearance (red color)
    no_clearance_rectangles = [
        (1500, 0, 250, 1000),  # Left vertical
        (2500, 1000, 250, 1000),  # Right vertical
    ]
    for rect in no_clearance_rectangles:
        cv2.rectangle(screen, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), red, -1)

    # Draw the circle without clearance (red color)
    cv2.circle(screen, (4200, 800), 600, red, -1)

    # Draw the lines at clearance distance from each corner
    lines = [
        (0, 0, clearance, height),  # Left line
        (0, 0, width, clearance),  # Top line
        (width - clearance, 0, clearance, height),  # Right line
        (0, height - clearance, width, clearance)  # Bottom line
    ]
    for line in lines:
        cv2.rectangle(screen, (line[0], line[1]), (line[0] + line[2], line[1] + line[3]), green, -1)

    return screen

# Function to validate user input for start and end points
def get_valid_input(prompt, robot_radius, clearance, ask_orientation=True):
    """
    Get valid input from user for start and end points.
    """
    while True:
        try:
            x = int(input(prompt + " X coordinate: "))
            y = int(input(prompt + " Y coordinate: "))
            y = height - y
            if ask_orientation:
                theta = int(input(prompt + " angle (theta): "))
            else:
                theta = None
            if not (0 <= x <= 6000 and 0 <= y <= 2000):
                raise ValueError("Coordinates out of bounds")
            elif point_inside_circle(x, y):
                raise ValueError("Coordinates within hexagon obstacle space")
            elif point_inside_rectangle(x, y, robot_radius):
                raise ValueError("Coordinates within rectangle obstacle space")
            return x, y, theta  # Adjust y-coordinate
        except ValueError as e:
            print("Invalid input:", e)

# Function to check if a point is inside the circular obstacle space
def point_inside_circle(x, y):
    """
    Check if the point lies inside the circle
    """
    circle_center = (4200, 800)
    circle_radius = 600 + clearance + robot_radius
    return (x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2 <= circle_radius ** 2

# Function to check if a point is inside the rectangular obstacle spac
def point_inside_rectangle(x, y, robot_radius):
    """
    Check if point lies inside the rectangular obstacle space.
    """
    rectangle_boundaries = [
        (1500 - clearance - robot_radius, 0, 1750 + clearance + robot_radius,
         1000 + robot_radius + clearance),  # Rectangle 1
        (2500 - clearance - robot_radius, 1000 - clearance - robot_radius, 2750 + clearance + robot_radius,
         2000),  # Rectangle 2
        (0, 0, robot_radius + clearance, height),  # Left boundary
        (0, 0, width, robot_radius + clearance),  # Top boundary
        (width - robot_radius - clearance, 0, width, height),  # Right boundary
        (0, height - robot_radius - clearance, width, height)  # Bottom boundary
    ]

    # Check if the point lies inside any of the rectangles or boundaries
    for rect in rectangle_boundaries:
        if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            return True  # Point is inside the rectangle
    return False  # Point is not inside any rectangle


# Get start and end points
#start_point = get_valid_input("Enter start", robot_radius, clearance, ask_orientation=True)
start_x, start_y, start_theta = 500, 1000, 0
end_point = get_valid_input("End point", robot_radius, clearance, ask_orientation=False)
wL = 30
wR = 30
end_x, end_y = end_point[:2]
initial = (start_x, start_y)
final = (end_x, end_y)
initial_orientation = start_theta

# Display the screen
def display_image():
    """
    Display the generated image.
    """
    flipped_screen = cv2.flip(screen, 0)  # Flip the image vertically
    plt.imshow(flipped_screen)
    plt.gca().invert_yaxis()
    plt.show()

# Function to check if a point is valid (not inside any obstacle or outside map boundaries)
def is_valid_point(point):
    """
    Check if a point is valid (not inside any obstacle or outside map boundaries).
    """
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        return False

    # Check if the point is inside any obstacle space or clearance area
    if point_inside_circle(point[0], point[1]) or \
            point_inside_rectangle(point[0], point[1], robot_radius):
        return False

    # If none of the above conditions are met, the point is valid
    return True

# Function to calculate the incremental movement based on wheel speeds
def increment(wL, wR, theta):
    """
    Calculate incremental movement based on wheel speeds.
    """
    wL = wL * (math.pi / 30)
    wR = wR * (math.pi / 30)

    Xn = 0.5*robot_radius*(wL+wR)*(math.cos(theta))*dt
    Yn = 0.5*robot_radius*(wL+wR)*(math.sin(theta))*dt

    Thetan = (robot_radius/wheel_distance)*(wR - wL)*dt

    return Xn, Yn, Thetan

# Function to move the robot based on wheel speeds
def move(left, right, node, end_point):
    """
    Move the robot based on wheel speeds.
    """
    current_node = node
    node_list = []

    for i in range(8):
        current_point = current_node.getPos()
        current_orientation = current_node.getTheta()

        Xn, Yn, Thetan = increment(left,right, current_orientation)

        new_x = int(round(current_point[0] + Xn))
        new_y = int(round(current_point[1] + Yn))
        new_orientation = current_orientation + Thetan

        move_cost = math.sqrt((Xn)**2 + (Yn)**2)
        new_point = (new_x,new_y)

        if not is_valid_point(new_point):
            return

        new_node = createNode(pos=new_point,theta=new_orientation, parent= current_node, cost= current_node.cost + move_cost, euclidean= euclidean(new_point,end_point))
        node_list.append(new_node)
        current_node = new_node

    all_points = []
    for sub_node in node_list:
        all_points.append(sub_node.getPos())

    for p in range(len(all_points)-1):
        cv2.line(screen, all_points[p], all_points[p+1], white, 3)

    return node_list

# Function to get possible actions (movements) from a given node
def get_actions(node, wL, wR, final):
    """
    Get possible actions (movements) from a given node.
    """
    actions = []
    for left, right in [(0, wL), (wL, 0), (wL, wL), (0, wR), (wR, 0), (wR, wR), (wL, wR), (wR, wL)]:
        actions.append(move(left, right, node, final))
    return actions

# Check if point is in the goal threshold
def reached_goal(point,goal, threshold):
    x , y = point[0] ,point[1]
    circle_center_x, circle_center_y, = goal[0], goal[1]
    radius = threshold
    distance = math.sqrt((x - circle_center_x) ** 2 + (y - circle_center_y) ** 2)
    return distance <= radius

# Calculate the euclidean value = Euclidean distance
def euclidean(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# def save_values_to_csv(velocities,orientation, file_name='values.csv'):
#     with open(file_name, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Velocity','Orientation'])  # Writing header
#         for vel,angle in zip(velocities,orientation):
#             writer.writerow([vel, angle])
# A* Algorithm
def a_star(start_position, end_position, start_orientation):
    # Initialize lists
    open_nodes = []
    closed_nodes = set()
    visited = set()

    start_node = createNode(start_position, start_orientation, None, 0, euclidean(start_position, end_position))

    heapq.heappush(open_nodes, (start_node.total_cost, start_node))
    visited.add(start_position)
    closed_nodes.add(start_position)

    iteration_count = 0
    while open_nodes:
        iteration_count += 1

        current_node = heapq.heappop(open_nodes)[1]  # Pop node with lowest total cost
        current_position = current_node.getPos()

        # Check if current position is within the goal threshold
        if reached_goal(current_position, end_position, 100):
            print("Goal reached")
            path = []
            theta = []
            path_with_angle = []
            while current_node is not None:
                path.append(current_node.getPos())
                path_with_angle.append((current_node.getPos(), current_node.getTheta()))
                theta.append(current_node.getTheta())
                screen[current_node.getPos()[1], current_node.getPos()[0]] = (255, 255, 0)
                current_node = current_node.getParent()

            return path[::-1], theta[::-1]

        actions = get_actions(current_node, wL, wR, end_position)

        # Move in all directions
        for child_node in actions:
            if child_node is not None:
                new_node = child_node[len(child_node)-1]
                new_position = new_node.getPos()
                if new_position not in closed_nodes:
                    if new_position not in visited:
                        heapq.heappush(open_nodes, (new_node.total_cost, new_node))
                        visited.add(new_position)
                        closed_nodes.add(new_position)

                        if iteration_count % 3000 == 0:
                            resized_screen = cv2.resize(screen, (1800, 600))
                            flip_screen = cv2.flip(resized_screen, 0)
                            cv2.imshow("exploration", resized_screen)
                            cv2.waitKey(1)

                        draw_exploration(new_node)
                    else:
                        if new_position not in visited or visited[new_position].total_cost > new_node.total_cost:
                            heapq.heappush(open_nodes, (new_node.total_cost, new_node))
                            visited.add(new_position)  # Add new_position to visited set


# Draw the optimal path
def draw_path(path):
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i + 1]
        cv2.line(screen, start_point, end_point, green, 8)

# Draw the exploration of the nodes
def draw_exploration(explored_node):
    if explored_node.getParent() is not None:
        cv2.line(screen, explored_node.getPos(), explored_node.getParent().getPos() , white, 4)

# Function to calculate and return velocity for each point in the path
def get_velocity(path):
    velocity = []
    for i in range(1, len(path)):
        Xn = (path[i][0] - path[i-1][0])/3.5
        Yn = (path[i][1] - path[i-1][1])/3.5

        velocity_ = (math.sqrt(Xn**2 + Yn**2) / dt)/1000
        velocity.append(0)
        velocity.append(velocity_)

    return velocity

def orientation_val(path):
    
    ang_vel = [0]
    prev_ang = 0
    
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        dt = 2
        
        if dx == 0 or dy == 0:
            ang = prev_ang
        else:
            ang = math.atan(dy/dx)
        
        ang_change = (ang - prev_ang) / dt  
        
        ang_vel.append(0)
        ang_vel.append(ang_change)
        
        
        prev_ang = ang
        
        
    return ang_vel

def main(args=None):

    rclpy.init(args=args)
    turtlebot_controller = TurtleBotController()
    draw_scene(clearance)
    print("Start search")
    path, orientation = a_star(initial, final, initial_orientation)

    orientation = orientation_val(path)
    orientation = [-angle for angle in orientation]

    velocities = get_velocity(path)


    turtlebot_controller.velocities = velocities
    turtlebot_controller.orientation = orientation

        # Visualize start and goal nodes
    cv2.circle(screen, (initial[0], initial[1]), 10, red, -1)  # Start node in green
    cv2.circle(screen, (final[0], final[1]), 10, red, -1)  # Goal node in green

    draw_path(path)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('exploration_video.avi', fourcc, 20.0, (screen.shape[1], screen.shape[0]))

    # Iterate through the frames and write to video
    for frame in range(len(screen)):
        out.write(screen[frame])

    out.release()  # Release the VideoWriter object

    # Read and display the saved video
    cap = cv2.VideoCapture('exploration_video.avi')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Exploration Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
            break
        time.sleep(10)

    cap.release()
    cv2.destroyAllWindows()

    display_image()


    turtlebot_controller.start_moving()

    rclpy.spin(turtlebot_controller)

    turtlebot_controller.destroy_node()
    rclpy.shutdown()

# Main function
if __name__ == "__main__":

    # save_values_to_csv(velocities, orientation)

    main()


