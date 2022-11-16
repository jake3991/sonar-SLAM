#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Joy
from scipy.spatial.transform import Rotation
from gazebo_msgs.msg import ModelStates, LinkStates

class control():
    '''class to be used as a sub class for 3D mapping
    '''
    def __init__(self):

        scene = "penns_landing"

        # penns landing
        if scene == "penns_landing":
            self.depth_set_point = -1
            self.waypoints = np.array([ [300, 0, -75],
                                        [300,-18,0],
                                        [250,-18,80],
                                        [50,-5,80],
                                        [-30,-10,80],
                                        [-30,10,-80],
                                        [225,10,-80],
                                        [300,0,-80],
                                        ])

        # RFAL land
        elif scene == "RFAL_land":
            self.depth_set_point = -1
            self.waypoints = np.array([ [-20, 0, 80],
                                        [12, 0, 80],
                                        [12, 25, 0],
                                        [0, 25, 80],
                                        [-5, 22, 80],
                                        [-30, 19, 80],
                                        [0,19,-80]
                                        ])
        # plane
        elif scene == "plane":
            self.depth_set_point = -7
            self.waypoints = np.array([ [-10, -5, -20],
                                        [0, -20, -45],
                                        [15, -20, -100],
                                        [15, -9, -175],
                                        [25, -20, -100],
                                        [25, 20, -190],
                                        [15, 9, 85],
                                        [15, 20, -170],
                                        [0, 20, 85],
                                        [-10, 5, 20],
                                        [-10, 0, 0]
                                        ])

        #class object counter for waypoints
        self.currentWaypoint = 0        

        #joystick command publisher
        self.command = rospy.Publisher("rexrov/joy", Joy, queue_size = 100)

        #set up the subsriber
        self.sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.callback)


    def rotationMtx(self,angle):
        '''Generate a rotation matrix for a single angle, yaw
        '''

        mtx = np.array([[ np.cos(angle), np.sin(angle) ],
                        [ -np.sin(angle), np.cos(angle) ]])

        return mtx

    def generateRotationCommand(self,setPoint,yaw):
        '''Takes the desired angle and the current angle.  Outputs the rotation command with the correct sign
        '''

        rotationCmd = 0

        if yaw > 0 and setPoint > 0:
            rotationCmd = yaw - setPoint

        elif yaw < 0 and setPoint > 0:

            #get the possible rotation commands
            rotationCmd = np.array([yaw - setPoint,  - yaw - setPoint])

            #get the smallest command
            idx = np.argmin(abs(rotationCmd))
            rotationCmd = rotationCmd[idx]

        elif yaw > 0 and setPoint < 0:

            #get the possible rotation commands
            rotationCmd = np.array([yaw - setPoint,  - ((180. - yaw + (180 + setPoint))) ])

            #get the smallest command
            idx = np.argmin(abs(rotationCmd))
            rotationCmd = rotationCmd[idx]

        elif yaw < 0 and setPoint < 0:

            rotationCmd = abs(setPoint) - abs(yaw)

        elif yaw == 0:

            rotationCmd = - setPoint

        elif setPoint == 0:

            rotationCmd = yaw

        if abs(rotationCmd) < 1.0:
            rotationCmd = 0

        return rotationCmd

    def commandRobot(self,setPoints,gazebo_pose):
        '''Generates and publishes control commands for a given waypoint
        '''

        #get the pose from the gazebo link message
        idx = gazebo_pose.name.index('rexrov::rexrov/horizontal_sonar/forward_sonarhoriz_link')
        pose_ = gazebo_pose.pose[idx]

        #get the true pose from gazebo
        quat = [0, 0, 0, 0]
        quat[0] = pose_.orientation.x
        quat[1] = pose_.orientation.y
        quat[2] = pose_.orientation.z
        quat[3] = pose_.orientation.w
        eulerAngles = list(Rotation.from_quat(quat).as_euler('xyz', degrees=True))
        yaw = - eulerAngles[2]

        #generate a rotation command
        rotationCmd = self.generateRotationCommand(setPoints[2], yaw)
        rotationCmd = np.round(rotationCmd)

        #saturate the rotation commands
        if rotationCmd < 0.:
            rotationCmd = -1.0
        if rotationCmd > 0:
            rotationCmd = 1.0

        #generate and x,y error signal
        xDiff = abs(pose_.position.x - setPoints[0])
        yDiff = abs(pose_.position.y - setPoints[1])

        
        zDiff = self.depth_set_point - pose_.position.z
        if abs(zDiff) <= 0.05:
            zDiff = 0.
        elif zDiff > 0.1:
            zDiff = 0.75
        else:
            zDiff = -0.75

        print(pose_.position.z, zDiff)

        #if withen tolerance, make the error signal zero
        if xDiff < 0.5:
            xDiff = 0
        if yDiff < 0.5:
            yDiff = 0

        #saturate the error signal
        if xDiff > 1:
            xDiff = 1
        if yDiff > 1:
            yDiff = 1

        #check the sign on x
        if setPoints[0] > pose_.position.x:
            xCmd = xDiff
        else:
            xCmd = -xDiff

        #check the sign on y
        if setPoints[1] > pose_.position.y:
            yCmd = yDiff
        else:
            yCmd = -yDiff

        #apply roation to the x,y command vector
        mtx = self.rotationMtx(np.radians(yaw)) 
        vec = np.array([ [yCmd], [xCmd] ]).reshape(2, 1)
        vec = np.matmul(mtx,vec)
        vec = vec / abs(np.max(vec))


        vec = np.round(vec)

        #saturate
        vec[vec > 0] = 1.0
        vec[vec < 0] = -1.0

        #compile the controller message
        controlCmd = [rotationCmd, zDiff, 0.0, vec[0], vec[1], 0.0, 0.0, 0.0]
        
        #set up a new joy message
        joyMsg = Joy()
        joyMsg.axes = controlCmd
        joyMsg.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        #publish the joy message
        self.command.publish(joyMsg)

        #print xDiff, yDiff, rotationCmd

        #if at waypoint, return true
        if xDiff == 0 and yDiff == 0 and rotationCmd == 0:
            return True

        #else return false
        else:
            return False

    def callback(self,gazebo_pose):

        #if not at the current waypoint, keep going there
        if self.commandRobot(self.waypoints[self.currentWaypoint], gazebo_pose) == True:

            if self.currentWaypoint != len(self.waypoints) - 1:
                self.currentWaypoint += 1

    
if __name__ == "__main__":

    #init the node
    rospy.init_node("waypoint_control", log_level=rospy.INFO)

    control()

    #log info and spin
    rospy.loginfo("Starting Waypoint Control...")
    rospy.spin()
