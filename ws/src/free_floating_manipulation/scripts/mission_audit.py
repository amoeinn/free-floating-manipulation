#!/usr/bin/env python3
"""An independent verdict on what the mission actually did.

A behavior tree returning SUCCESS is the same class of claim as a simulator
service returning true: it reports that the calls were made, not that they had
the intended effect. Phase 5 met that three times in one pose path. Phase 4
supplied the version that matters here, a tracker holding smooth confident
lock on a pose 180 degrees wrong with nothing in the track reporting it.

So this subscribes to ground truth and to the executive's declared belief, and
never talks to the tree. It cannot be reassured by anything the executive
says, because it is not listening to the executive's opinion of itself, only
to the pose the executive published and the pose the scene knows to be true.

It is mutation checked before it is believed. `acquire_target` carries an
`inject_flip` hook precisely so that a mission can be made to return SUCCESS
end to end while being 180 degrees wrong, and this must report that. An
auditor that cannot catch the one failure mode the phase documents is
decoration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node as RosNode
from std_msgs.msg import String

TOLERANCE_DEG = 10.0


def quaternion_to_matrix(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def geodesic_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


class MissionAudit(RosNode):
    def __init__(self):
        super().__init__("mission_audit")
        self.truth = None
        self.estimate = None
        self.worst = 0.0
        self.samples = 0
        self.create_subscription(PoseStamped, "/truth/target/pose", self.on_truth, 10)
        self.create_subscription(PoseStamped, "/target/pose_estimate",
                                 self.on_estimate, 10)
        self.create_subscription(String, "/mission/report", self.on_report, 10)
        self.pub = self.create_publisher(String, "/mission/audit", 10)
        self.out = Path("/tmp/mission_audit.json")
        self.get_logger().info(
            "audit up: comparing /target/pose_estimate against /truth/target/pose. "
            "It is not subscribed to anything the executive says about itself.")

    def on_truth(self, msg):
        self.truth = quaternion_to_matrix(msg.pose.orientation)

    def on_estimate(self, msg):
        self.estimate = quaternion_to_matrix(msg.pose.orientation)
        if self.truth is None:
            return
        error = geodesic_deg(self.estimate, self.truth)
        self.worst = max(self.worst, error)
        self.samples += 1

    def on_report(self, msg):
        """The tree has finished. Give the independent verdict."""
        try:
            claim = json.loads(msg.data)
        except json.JSONDecodeError:
            claim = {"raw": msg.data}
        if not claim.get("executed", False):
            # A mission that never acted on its belief is not a mission this
            # can grade. Saying MISSION CORRECT there would be grading the
            # pose that happened to be on the wire, not the mission.
            verdict = {"verdict": "MISSION ABORTED, NOTHING TO GRADE",
                       "why": claim.get("abort_reason", "the tree aborted"),
                       "tree_said": claim.get("status", "unknown")}
        elif self.estimate is None or self.truth is None:
            verdict = {"verdict": "NO EVIDENCE",
                       "why": "no pose estimate or no truth was ever seen"}
        else:
            error = geodesic_deg(self.estimate, self.truth)
            flipped = error > 90.0
            ok = error <= TOLERANCE_DEG
            verdict = {
                "verdict": "MISSION CORRECT" if ok else "MISSION WRONG",
                "final_attitude_error_deg": round(error, 3),
                "worst_attitude_error_deg": round(self.worst, 3),
                "samples": self.samples,
                "looks_flipped": flipped,
                "tree_said": claim.get("status", "unknown"),
                "why": ("the executive's final belief is within tolerance of the "
                        "true client attitude" if ok else
                        f"the executive's final belief is {error:.1f} deg from the "
                        "true client attitude" +
                        (", which is the 180 degree flip phase 4 documented"
                         if flipped else "")),
            }
        verdict["tolerance_deg"] = TOLERANCE_DEG
        verdict["claim"] = claim
        text = json.dumps(verdict, indent=2)
        self.out.write_text(text + "\n")
        self.pub.publish(String(data=json.dumps(verdict)))
        # Reset before the next mission. Holding the estimate across missions
        # made the audit render a verdict on the previous mission's belief:
        # the mutated run scored 0.797 deg and MISSION CORRECT because the
        # only estimate it had was the nominal run's. An auditor that reports
        # on stale evidence is worse than none.
        self.estimate = None
        self.worst = 0.0
        self.samples = 0
        banner = verdict["verdict"]
        log = self.get_logger().info if banner == "MISSION CORRECT" else \
            self.get_logger().error
        log(f"{banner}: tree said {verdict.get('tree_said')}, audit says "
            f"{verdict.get('final_attitude_error_deg', 'n/a')} deg from truth")
        print(text, flush=True)


def main():
    rclpy.init()
    node = MissionAudit()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
