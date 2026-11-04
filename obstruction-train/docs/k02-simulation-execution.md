# K02-a: Simulation trajectory dispatch seam

`dispatch_simulation_trajectory(request, *, expected_binding, simulator)` is a small, standalone CPU-side boundary for handing a validated trajectory to an injected simulation callable. It imports no robot or simulator SDK, provides no default driver, and never opens a connection. The caller must supply `simulator(request)` explicitly.

## Payload contract

Both `request` and `expected_binding` are dictionaries. The binding consists of exactly these fields:

- `robot_model`: nonempty string
- `joint_names`: nonempty ordered list of unique, nonempty strings
- `coordinate_frame`: nonempty string
- `calibration_id`: nonempty string

The request contains exactly the binding fields plus:

- `mode`: must be explicitly `"simulation"`; missing, unknown, and `"real"` modes are rejected
- `command_id`: nonempty string
- `position_unit`: exactly `"rad"`
- `time_unit`: exactly `"s"`
- `points`: nonempty ordered list of points

Each point contains exactly `positions` and `time_from_start`. `positions` is a list of finite real, non-boolean numbers with one value per joint, in the binding's joint order. `time_from_start` is a finite real, non-boolean number in seconds; it must be positive and strictly increase across points. Unsupported extra fields are rejected.

The request's model, joint names (including order), frame, and calibration ID must exactly match a valid `expected_binding`. Validation errors raise `ValueError` and do not call the simulator. The dispatcher deep-copies the request before validating and passes that validated copy to the simulator, so simulator mutation cannot change the caller's request or binding. Simulator exceptions propagate, and its return value is passed through unchanged.

## Limitations and safety boundary

This is only the K02-a trajectory dispatch seam. It is not yet wired into `run_replay` or Isaac, and it is not a physical-safety verifier. It does not check joint limits, velocity/acceleration, IK, planning, collisions, live state, feedback, or stopping behavior. No real driver is implemented or available through this API. A simulator call, its return value, or command acceptance does not establish physical execution or task success.

The broader K02 camera/state/gripper interfaces, pause/stop/timeout propagation, and real-adapter isolation remain pending. Hardware compatibility and physical safety are not complete or claimed here.