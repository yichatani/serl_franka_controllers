"""
Franka Impedance Controller Precision Test (Simplified)
Test 4: policy-like absolute pose tracking at realistic frequencies.
"""

# python test/test_precision.py --robot_ip=172.16.0.2 --test=policy --repeat_n=10 --settle_time=3

import sys, time, subprocess, threading
import rospy
import numpy as np
import geometry_msgs.msg as geom_msg
import franka_msgs.msg as franka_msg
from dynamic_reconfigure.client import Client
from absl import app, flags
from scipy.spatial.transform import Rotation as R

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", None, "IP address of the robot.", required=True)
flags.DEFINE_string("load_gripper", "false", "Whether or not to load the gripper.")
flags.DEFINE_string("test", "all", "Test to run: steady, repeat, tracking, policy, all")
flags.DEFINE_integer("settle_time", 3, "Seconds to wait after sending command.")
flags.DEFINE_integer("repeat_n", 5, "Repetitions for repeatability test.")
flags.DEFINE_float("policy_hz", 5.0, "Command frequency for policy-like test.")

DEFAULT_EULER = (np.pi, 0, np.pi / 2)


class FrankaStateListener:
    def __init__(self):
        self._lock = threading.Lock()
        self._O_T_EE = None
        self._O_T_EE_d = None
        self._sub = rospy.Subscriber(
            "/franka_state_controller/franka_states",
            franka_msg.FrankaState, self._cb, queue_size=1)

    def _cb(self, msg):
        with self._lock:
            self._O_T_EE = np.array(msg.O_T_EE).reshape(4, 4).T
            self._O_T_EE_d = np.array(msg.O_T_EE_d).reshape(4, 4).T

    def get(self):
        with self._lock:
            if self._O_T_EE is None:
                return None, None
            return self._O_T_EE.copy(), self._O_T_EE_d.copy()

    def wait_for_data(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.get()[0] is not None:
                return True
            time.sleep(0.05)
        return False


# ── Helpers ──

def pos_err_mm(T, target_xyz):
    return np.linalg.norm(T[:3, 3] - np.array(target_xyz)) * 1000

def ori_err_deg(T, euler=DEFAULT_EULER):
    R_cmd = R.from_euler("xyz", list(euler)).as_matrix()
    R_err = T[:3, :3] @ R_cmd.T
    return np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))

def make_pose_msg(x, y, z, euler=DEFAULT_EULER):
    msg = geom_msg.PoseStamped()
    msg.header.frame_id = "0"
    msg.header.stamp = rospy.Time.now()
    msg.pose.position = geom_msg.Point(x, y, z)
    q = R.from_euler("xyz", list(euler)).as_quat()
    msg.pose.orientation = geom_msg.Quaternion(*q)
    return msg

def move(pub, x, y, z, settle):
    msg = make_pose_msg(x, y, z)
    for _ in range(10):
        pub.publish(msg)
        time.sleep(0.1)
    time.sleep(settle)


def run_trajectory(pub, listener, waypoints, dt, label=""):
    """
    Send a sequence of absolute poses at fixed dt, measure instantaneous
    tracking lag at each step. Returns array of errors in mm.
    """
    errs = []
    for i, wp in enumerate(waypoints):
        pub.publish(make_pose_msg(*wp))
        time.sleep(dt)
        T, _ = listener.get()
        if T is not None:
            errs.append(pos_err_mm(T, wp))
    errs = np.array(errs) if errs else np.array([])
    if len(errs):
        print(f"    {label}  steps={len(errs)}  "
              f"mean={errs.mean():.3f}  std={errs.std():.3f}  max={errs.max():.3f} mm")
    return errs


# ── Test 1: Steady-State ──

def test_steady(pub, listener, settle):
    print("\n" + "=" * 55)
    print("TEST 1: Steady-State Positioning Accuracy")
    print("=" * 55)

    targets = [
        (0.45, -0.15, 0.25), (0.45,  0.15, 0.25),
        (0.55, -0.15, 0.25), (0.55,  0.15, 0.25),
        (0.50,  0.00, 0.20), (0.50,  0.00, 0.35),
        (0.50,  0.00, 0.45), (0.45,  0.00, 0.30),
        (0.55,  0.00, 0.30),
    ]

    errs_p, errs_o = [], []
    for i, t in enumerate(targets):
        print(f"  [{i+1}/{len(targets)}] -> ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
        move(pub, *t, settle)
        T, _ = listener.get()
        if T is None:
            print("    No state!"); continue
        pe, oe = pos_err_mm(T, t), ori_err_deg(T)
        errs_p.append(pe); errs_o.append(oe)
        a = T[:3, 3]
        print(f"    actual=({a[0]:.5f},{a[1]:.5f},{a[2]:.5f})  pos_err={pe:.3f}mm  ori_err={oe:.4f}°")

    if errs_p:
        print(f"\n  Pos err (mm):  mean={np.mean(errs_p):.3f}  std={np.std(errs_p):.3f}  max={np.max(errs_p):.3f}")
        print(f"  Ori err (deg): mean={np.mean(errs_o):.4f}  std={np.std(errs_o):.4f}  max={np.max(errs_o):.4f}")


# ── Test 2: Repeatability ──

def test_repeat(pub, listener, settle, n):
    print("\n" + "=" * 55)
    print(f"TEST 2: Repeatability (n={n})")
    print("=" * 55)

    target = (0.50, 0.0, 0.30)
    detour = (0.45, 0.10, 0.40)
    pts = []

    for rep in range(n):
        move(pub, *detour, max(1, settle - 1))
        move(pub, *target, settle)
        T, _ = listener.get()
        if T is None:
            print(f"  [{rep+1}] No state!"); continue
        a = T[:3, 3]
        pts.append(a.copy())
        pe = pos_err_mm(T, target)
        print(f"  [{rep+1}] actual=({a[0]:.5f},{a[1]:.5f},{a[2]:.5f})  err={pe:.3f}mm")

    if len(pts) >= 2:
        pts = np.array(pts)
        c = pts.mean(axis=0)
        dists = np.linalg.norm(pts - c, axis=1) * 1000
        print(f"\n  Centroid: ({c[0]:.5f},{c[1]:.5f},{c[2]:.5f})")
        print(f"  Spread (mm): mean={np.mean(dists):.4f}  std={np.std(dists):.4f}  max={np.max(dists):.4f}")
        print(f"  Per-axis std (mm): x={np.std(pts[:,0])*1000:.4f}  y={np.std(pts[:,1])*1000:.4f}  z={np.std(pts[:,2])*1000:.4f}")


# ── Test 3: Trajectory Tracking (absolute, single axis) ──

def test_tracking(pub, listener):
    print("\n" + "=" * 55)
    print("TEST 3: Trajectory Tracking (absolute target)")
    print("=" * 55)

    x, y = 0.50, 0.0
    z0, z1 = 0.20, 0.45
    n_steps, dt = 50, 0.1

    move(pub, x, y, z0, 3)

    errs = []
    for direction, (za, zb) in [("up", (z0, z1)), ("down", (z1, z0))]:
        print(f"  {direction}: z={za:.2f} -> {zb:.2f}")
        for i in range(n_steps):
            cz = za + (zb - za) * i / (n_steps - 1)
            pub.publish(make_pose_msg(x, y, cz))
            time.sleep(dt)
            T, _ = listener.get()
            if T is not None:
                errs.append(pos_err_mm(T, (x, y, cz)))

    if errs:
        errs = np.array(errs)
        print(f"\n  Tracking err (mm): mean={errs.mean():.3f}  std={errs.std():.3f}  max={errs.max():.3f}")


# ── Test 4: Policy-Like Absolute Pose Tracking ──

def test_policy_like(pub, listener, hz):
    """
    Measures instantaneous tracking lag when sending absolute poses
    at policy frequency. No error accumulation — each step's error
    is just "how far behind is the controller right now".

    Sub-tests:
    A) Straight line       — constant velocity, single axis
    B) Square path         — direction changes
    C) Diagonal + z-wave   — multi-axis simultaneous motion
    D) Random walk         — noisy policy output
    E) Speed comparison    — same path at different velocities
    """
    dt = 1.0 / hz
    print("\n" + "=" * 55)
    print(f"TEST 4: Policy-Like Tracking Lag ({hz:.0f} Hz)")
    print("=" * 55)
    print(f"  All errors are instantaneous lag (target vs actual at each step).")
    print(f"  No accumulation — absolute pose means each step is independent.\n")

    all_errs = []

    # ── 4A: Straight line z ──
    print("  --- 4A: Straight line z ---")
    n = 100
    start = np.array([0.50, 0.0, 0.22])
    move(pub, *start, 0)
    wps = [start + np.array([0, 0, 0.002]) * (i + 1) for i in range(n)]
    e = run_trajectory(pub, listener, wps, dt, "z-line")
    if len(e): all_errs.append(e)

    # ── 4B: Square in xy ──
    print("  --- 4B: Square path (60mm side) ---")
    side = 0.06
    sps = 30  # steps per side
    step = side / sps
    sq_start = np.array([0.47, -0.03, 0.30])
    move(pub, *sq_start, 0)

    wps = []
    pos = sq_start.copy()
    for dx, dy in [(step,0), (0,step), (-step,0), (0,-step)]:
        for _ in range(sps):
            pos = pos + np.array([dx, dy, 0])
            wps.append(pos.copy())
    e = run_trajectory(pub, listener, wps, dt, "square")
    if len(e): all_errs.append(e)

    # ── 4C: Diagonal + sinusoidal z ──
    print("  --- 4C: Diagonal + z sine wave ---")
    n = 100
    diag_start = np.array([0.45, -0.05, 0.30])
    move(pub, *diag_start, 0)
    wps = []
    for i in range(n):
        t = i / (n - 1)
        x = diag_start[0] + 0.10 * t        # +100mm in x
        y = diag_start[1] + 0.10 * t         # +100mm in y
        z = diag_start[2] + 0.03 * np.sin(2 * np.pi * t)  # ±30mm sine in z
        wps.append(np.array([x, y, z]))
    e = run_trajectory(pub, listener, wps, dt, "diagonal+sine")
    if len(e): all_errs.append(e)

    # ── 4D: Random walk ──
    print("  --- 4D: Random walk (std=2mm/step) ---")
    rand_start = np.array([0.50, 0.0, 0.30])
    move(pub, *rand_start, 0)

    rng = np.random.RandomState(42)
    n = 100
    wps = []
    pos = rand_start.copy()
    for i in range(n):
        delta = rng.randn(3) * 0.002
        delta[2] *= 0.5
        pos = pos + delta
        pos[0] = np.clip(pos[0], 0.35, 0.65)
        pos[1] = np.clip(pos[1], -0.20, 0.20)
        pos[2] = np.clip(pos[2], 0.15, 0.50)
        wps.append(pos.copy())
    e = run_trajectory(pub, listener, wps, dt, "random")
    if len(e): all_errs.append(e)

    # ── 4E: Speed comparison (same line, different step sizes) ──
    print("  --- 4E: Speed comparison (same path, different step sizes) ---")
    for speed_label, step_mm in [("slow 1mm", 0.001), ("medium 3mm", 0.003), ("fast 5mm", 0.005)]:
        sp_start = np.array([0.50, 0.0, 0.22])
        move(pub, *sp_start, 0)
        n = int(0.20 / step_mm)  # all travel 100mm total
        n = min(n, 100)
        wps = [sp_start + np.array([0, 0, step_mm]) * (i + 1) for i in range(n)]
        e = run_trajectory(pub, listener, wps, dt, speed_label)
        if len(e): all_errs.append(e)

    # ── Summary ──
    if all_errs:
        combined = np.concatenate(all_errs)
        print(f"\n  ── Test 4 Overall ──")
        print(f"  Tracking lag (mm): mean={combined.mean():.3f}  std={combined.std():.3f}  "
              f"max={combined.max():.3f}  median={np.median(combined):.3f}")
        print(f"  This is the lag floor — your policy's own prediction error adds on top.")


# ── Main ──

def main(_):
    try:
        input("\033[33mPress enter to start roscore + controller.\033[0m")
        try:
            roscore = subprocess.Popen("roscore"); time.sleep(1)
        except: pass

        ctrl = subprocess.Popen(
            ["roslaunch", "serl_franka_controllers", "impedance.launch",
             f"robot_ip:={FLAGS.robot_ip}", f"load_gripper:={FLAGS.load_gripper}"],
            stdout=subprocess.PIPE)

        rospy.init_node("franka_precision_test")
        pub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped, queue_size=10)
        time.sleep(1)

        listener = FrankaStateListener()
        print("Waiting for franka_states ...")
        if not listener.wait_for_data():
            print("ERROR: No state. Exiting."); ctrl.terminate(); roscore.terminate(); sys.exit(1)
        print("  Ready.\n")

        client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")
        for d in ["x", "y", "z", "neg_x", "neg_y", "neg_z"]:
            client.update_configuration({"translational_clip_" + d: 0.01})
            client.update_configuration({"rotational_clip_" + d: 0.08})

        home = (0.50, 0.0, 0.30)
        input("\033[33mPress enter to move to home.\033[0m")
        move(pub, *home, 3)

        t = FLAGS.test.lower()
        if t in ("steady", "all"):
            input("\033[33mPress enter for Test 1: Steady-State.\033[0m")
            test_steady(pub, listener, FLAGS.settle_time)
            move(pub, *home, 2)
        if t in ("repeat", "all"):
            input("\033[33mPress enter for Test 2: Repeatability.\033[0m")
            test_repeat(pub, listener, FLAGS.settle_time, FLAGS.repeat_n)
            move(pub, *home, 2)
        if t in ("tracking", "all"):
            input("\033[33mPress enter for Test 3: Trajectory Tracking.\033[0m")
            test_tracking(pub, listener)
            move(pub, *home, 2)
        if t in ("policy", "all"):
            input("\033[33mPress enter for Test 4: Policy-Like Tracking.\033[0m")
            test_policy_like(pub, listener, FLAGS.policy_hz)
            move(pub, *home, 2)

        input("\033[33mPress enter to stop and exit.\033[0m")
        ctrl.terminate(); roscore.terminate(); sys.exit()

    except Exception as e:
        rospy.logerr(f"Error: {e}")
        try: ctrl.terminate(); roscore.terminate()
        except: pass
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)