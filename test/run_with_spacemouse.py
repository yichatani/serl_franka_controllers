import sys
import time
import subprocess
import threading
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event
from collections import defaultdict
from dynamic_reconfigure.client import Client
import numpy as np
import rospy
import geometry_msgs.msg as geom_msg
import franka_msgs.msg as franka_msg
from scipy.spatial.transform import Rotation as R

_state_lock = threading.Lock()
_O_T_EE = None

def state_callback(msg):
    global _O_T_EE
    with _state_lock:
        _O_T_EE = np.array(msg.O_T_EE).reshape(4, 4).T

class Spacemouse(Thread):
    def __init__(self, max_value=500, deadzone=(0,0,0,0,0,0), dtype=np.float32):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        super().__init__()
        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

    def get_motion_state(self): #this method gets the movement of the mouse 
        me = self.motion_event
        state = np.array(me.translation + me.rotation, 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self): #transforms get_motion_state 
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]

        # Set values lesser than 0.3 to 0 for better control
        tf_state[np.abs(tf_state) < 0.3] = 0
        # tf_state = tf_state * SCALE_FACTOR
        tf_state[:3] *= SCALE_FACTOR        
        tf_state[3:] *= SCALE_FACTOR * 1.5

        tf_state = tf_state * 0.25

        return tf_state

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1/200)
        finally:
            spnav_close()

SCALE_FACTOR = 0.5
POS_SCALE = 0.4
ROT_SCALE = 0.1
CONTROL_HZ = 20.0

def main():
    input("\033[33mPress enter to start roscore and controller.\033[0m")
    try:
        roscore = subprocess.Popen('roscore')
        time.sleep(1)
    except:
        pass

    ctrl = subprocess.Popen(
        ['roslaunch', 'serl_franka_controllers', 'impedance.launch',
         'robot_ip:=172.16.0.2', 'load_gripper:=true'],
        stdout=subprocess.PIPE)

    rospy.init_node('spacemouse_control')
    pub = rospy.Publisher(
        '/cartesian_impedance_controller/equilibrium_pose',
        geom_msg.PoseStamped, queue_size=10)
    client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")
    rospy.Subscriber(
        '/franka_state_controller/franka_states',
        franka_msg.FrankaState, state_callback, queue_size=1)

    print("Waiting for franka state...")
    time.sleep(2)
    with _state_lock:
        T = _O_T_EE.copy() if _O_T_EE is not None else None
    if T is None:
        print("ERROR: No state. Exiting.")
        ctrl.terminate(); roscore.terminate(); sys.exit(1)

    # Use current pose as initial target
    pos = T[:3, 3].copy()
    rot = R.from_matrix(T[:3, :3])
    print(f"Initial pose: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    print(f"Initial quat: {rot.as_quat()}")

    # print(f"matrix: {T[:3, :3]}")

    input("\033[33mPress enter to start spacemouse control. Ctrl+C to stop.\033[0m")

    sm = Spacemouse()
    sm.start()
    dt = 1.0 / CONTROL_HZ

    # exit()
    time.sleep(1)
    # Setting the reference limiting values through ros dynamic reconfigure
    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
        client.update_configuration({"translational_clip_" + direction: 0.02})
        client.update_configuration({"rotational_clip_" + direction: 0.08})
    time.sleep(1)
    print("\nNew reference limiting values has been set")

    try:
        while not rospy.is_shutdown():
            motion = sm.get_motion_state_transformed()
            print(f"\rmotion={motion}")

            print(f"original pos:\n{pos}")
            with _state_lock:                                                                                                                         
                if _O_T_EE is not None:                                                                                                               
                    pos = _O_T_EE[:3, 3].copy()                                                                                                       
            pos += motion[:3] * POS_SCALE
            # pos[0] = np.clip(pos[0], 0.25, 0.75)
            # pos[1] = np.clip(pos[1], -0.35, 0.35)
            # pos[2] = np.clip(pos[2], 0.05, 0.55)
            print(f"new pos:\n{pos}")

            delta_rot = R.from_euler('xyz', motion[3:] * ROT_SCALE)
            quat_0 = rot.as_quat()
            rot = delta_rot * rot
            quat = rot.as_quat()
            print(f"original rot quat:\n{quat_0}")
            print(f"new rot quat:\n{quat}")


            msg = geom_msg.PoseStamped()
            msg.header.frame_id = "0"
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = geom_msg.Point(*pos)
            msg.pose.orientation = geom_msg.Quaternion(*quat)
            pub.publish(msg)

            # print(f"\rtarget=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})", end="")

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sm.stop()

    ctrl.terminate()
    roscore.terminate()

if __name__ == "__main__":
    main()
