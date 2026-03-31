import subprocess
import time
import rospy
import actionlib
import franka_gripper.msg as gripper_msg

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
    time.sleep(3)

    rospy.init_node('test_gripper')

    move_client = actionlib.SimpleActionClient('/franka_gripper/move', gripper_msg.MoveAction)
    grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', gripper_msg.GraspAction)

    print("Waiting for gripper action servers...")
    move_client.wait_for_server()
    grasp_client.wait_for_server()
    print("Connected.")

    # Test 1: Open
    input("\033[33mPress enter to open gripper (width=0.08m).\033[0m")
    goal = gripper_msg.MoveGoal(width=0.08, speed=0.1)
    move_client.send_goal(goal)
    move_client.wait_for_result()
    print(f"Result: {move_client.get_result()}")

    # Test 2: Close halfway
    input("\033[33mPress enter to close gripper to 0.04m.\033[0m")
    goal = gripper_msg.MoveGoal(width=0.04, speed=0.1)
    move_client.send_goal(goal)
    move_client.wait_for_result()
    print(f"Result: {move_client.get_result()}")

    # Test 3: Grasp
    input("\033[33mPress enter to grasp (width=0.01m, force=40N).\033[0m")
    goal = gripper_msg.GraspGoal(
        width=0.01, speed=0.1, force=40,
        epsilon=gripper_msg.GraspEpsilon(inner=0.01, outer=0.01))
    grasp_client.send_goal(goal)
    grasp_client.wait_for_result()
    print(f"Result: {grasp_client.get_result()}")

    # Test 4: Open again
    input("\033[33mPress enter to open gripper again.\033[0m")
    goal = gripper_msg.MoveGoal(width=0.08, speed=0.1)
    move_client.send_goal(goal)
    move_client.wait_for_result()
    print(f"Result: {move_client.get_result()}")

    print("Done.")

if __name__ == "__main__":
    main()
