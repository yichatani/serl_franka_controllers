import os
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(save_dir, exist_ok=True)
import time
import threading
import numpy as np
import rospy
import franka_msgs.msg as franka_msg
import sensor_msgs.msg as sensor_msg

_state_lock = threading.Lock()
_O_T_EE = None
_gripper_width = None

def state_callback(msg):
    global _O_T_EE
    with _state_lock:
        _O_T_EE = np.array(msg.O_T_EE).reshape(4, 4).T

def gripper_callback(msg):
    global _gripper_width
    with _state_lock:
        _gripper_width = sum(msg.position)

def main():
    rospy.init_node('record_data', anonymous=True)
    rospy.Subscriber('/franka_state_controller/franka_states',
        franka_msg.FrankaState, state_callback, queue_size=1)
    rospy.Subscriber('/franka_gripper/joint_states',
        sensor_msg.JointState, gripper_callback, queue_size=1)

    print("Waiting for franka state...")
    time.sleep(2)

    trajectory = []
    input("\033[33mPress enter to start recording. Ctrl+C to stop and save.\033[0m")

    try:
        while not rospy.is_shutdown():
            with _state_lock:
                if _O_T_EE is not None:
                    gripper_open = 1 if (_gripper_width is not None and _gripper_width > 0.04) else 0
                    trajectory.append({
                        'timestamp': time.time(),
                        'O_T_EE': _O_T_EE.copy(),
                        'gripper_open': gripper_open,
                    })
            print(f"\rRecording... frames={len(trajectory)}", end="")
            time.sleep(1.0 / 30)
    except KeyboardInterrupt:
        pass

    filename = os.path.join(save_dir, f"traj_{int(time.time())}.npy")
    np.save(filename, np.array(trajectory))
    print(f"\nSaved {len(trajectory)} frames to {filename}")

if __name__ == "__main__":
    main()
