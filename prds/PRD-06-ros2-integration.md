# PRD-06: ROS2 Integration

> Module: GS-GHOST | Priority: P1  
> Depends on: PRD-03, PRD-05  
> Status: ⬜ Not started

## Objective
ANIMA can trigger GS-GHOST from ROS2 using recorded or streamed monocular RGB data and publish reconstruction artifacts back into the robotics graph.

## Context (from paper)
The paper’s target deployment spaces include robotics and teleoperation, but the paper itself is not a ROS-native system. This PRD is the minimal adaptation layer.  
**Paper reference**: §5 "Conclusion"

## Acceptance Criteria
- [ ] ROS2 node subscribes to monocular RGB frames or accepts sequence directory paths
- [ ] Reconstruction trigger uses an action or service interface rather than blocking the graph
- [ ] Node publishes artifact paths plus lightweight markers for downstream ANIMA modules
- [ ] Test: `uv run pytest tests/test_ros2_contract.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_gs_ghost/ros2/node.py` | ROS2 batch bridge node | §5 adaptation | ~180 |
| `src/anima_gs_ghost/ros2/messages.py` | topic/service contracts | — | ~100 |
| `launch/gs_ghost.launch.py` | launch entrypoint | — | ~60 |
| `tests/test_ros2_contract.py` | interface validation | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `/camera/rgb/image_raw`
- `/gs_ghost/run_sequence`

### Outputs
- `/gs_ghost/status`
- `/gs_ghost/artifacts`
- `/gs_ghost/markers`

### Algorithm
```python
class GhostBatchNode(Node):
    def handle_request(self, req):
        return dispatch_reconstruction_job(req.sequence_path, req.prompt)
```

## Dependencies
```toml
rclpy = "*"
sensor_msgs = "*"
std_msgs = "*"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| rosbag or live RGB source | variable | external | runtime-provided |

## Test Plan
```bash
uv run pytest tests/test_ros2_contract.py -v
```

## References
- Paper: §5
- Depends on: PRD-03, PRD-05
- Feeds into: PRD-07

