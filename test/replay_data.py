import os
import sys
import time
import subprocess
import threading
import numpy as np
import rospy
import actionlib
import geometry_msgs.msg as geom_msg
import franka_msgs.msg as franka_msg
import franka_gripper.msg as gripper_msg
from dynamic_reconfigure.client import Client
from scipy.spatial.transform import Rotation as R

_state_lock = threading.Lock()
_O_T_EE = None

def state_callback(msg):
    global _O_T_EE
    with _state_lock:
        _O_T_EE = np.array(msg.O_T_EE).reshape(4, 4).T

def make_pose_msg(T):
    msg = geom_msg.PoseStamped()
    msg.header.frame_id = "0"
    msg.header.stamp = rospy.Time.now()
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    msg.pose.position = geom_msg.Point(*pos)
    msg.pose.orientation = geom_msg.Quaternion(*quat)
    return msg

def main():
    if len(sys.argv) < 2:
        # Find latest trajectory file
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        if not files:
            print(f"No .npy files found in {data_dir}")
            sys.exit(1)
        filepath = os.path.join(data_dir, files[-1])
    else:
        filepath = sys.argv[1]

    traj = np.load(filepath, allow_pickle=True)
    print(f"Loaded {len(traj)} frames from {filepath}")

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

    rospy.init_node('replay_data')
    pub = rospy.Publisher(
        '/cartesian_impedance_controller/equilibrium_pose',
        geom_msg.PoseStamped, queue_size=10)
    client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")
    rospy.Subscriber(
        '/franka_state_controller/franka_states',
        franka_msg.FrankaState, state_callback, queue_size=1)

    move_client = actionlib.SimpleActionClient('/franka_gripper/move', gripper_msg.MoveAction)
    grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', gripper_msg.GraspAction)
    print("Waiting for gripper action servers...")
    move_client.wait_for_server()
    grasp_client.wait_for_server()

    print("Waiting for franka state...")
    time.sleep(2)

    # Set reference limiting
    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
        client.update_configuration({"translational_clip_" + direction: 0.02})
        client.update_configuration({"rotational_clip_" + direction: 0.08})

    # Move to first frame
    first_frame = traj[0] if isinstance(traj[0], dict) else traj[0].item()
    first_T = first_frame['O_T_EE']
    pos = first_T[:3, 3]
    print(f"Moving to first frame: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    input("\033[33mPress enter to move to first frame.\033[0m")
    msg = make_pose_msg(first_T)
    for _ in range(20):
        pub.publish(msg)
        time.sleep(0.1)
    time.sleep(5)

    # Replay
    input("\033[33mPress enter to start replay.\033[0m")
    try:
        for i in range(len(traj)):
            frame = traj[i] if isinstance(traj[i], dict) else traj[i].item()
            T = frame['O_T_EE']
            pub.publish(make_pose_msg(T))

            # Gripper control on state change
            gripper_open = frame.get('gripper_open', 1)
            prev_frame = (traj[i - 1] if isinstance(traj[i - 1], dict) else traj[i - 1].item()) if i > 0 else {}
            if i == 0 or gripper_open != prev_frame.get('gripper_open', 1):
                if gripper_open:
                    move_client.send_goal(gripper_msg.MoveGoal(width=0.08, speed=0.1))
                    time.sleep(0.5)  # wait for gripper to open before moving
                else:
                    grasp_client.send_goal(gripper_msg.GraspGoal(
                        width=0.01, speed=0.1, force=40,
                        epsilon=gripper_msg.GraspEpsilon(inner=0.08, outer=0.08)))
                    time.sleep(0.5)  # wait for gripper to grasp before moving

            # Use original timing
            if i < len(traj) - 1:
                next_frame = traj[i + 1] if isinstance(traj[i + 1], dict) else traj[i + 1].item()
                next_t = next_frame['timestamp']
                cur_t = frame['timestamp']
                dt = next_t - cur_t
                dt = max(0, min(dt, 0.5))  # clamp to avoid long pauses
            else:
                dt = 1.0 / 30

            pos = T[:3, 3]
            print(f"\r[{i+1}/{len(traj)}] x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}", end="")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nReplay interrupted.")

    print("\nReplay done.")
    input("\033[33mPress enter to exit.\033[0m")
    ctrl.terminate()
    roscore.terminate()

if __name__ == "__main__":
    main()
