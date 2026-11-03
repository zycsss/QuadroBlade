# Level2 Reliable Sprint

入口为 `lsy_drone_racing/control/mpcc_reliable_sprint_controller.py`，已配置为
`config/level2.toml` 的默认控制器。

```sh
pixi run python scripts/sim.py --config level2.toml
```

关闭可视化运行：

```sh
pixi run python scripts/sim.py --config level2.toml --render False
```

控制器使用观测到的门框与障碍物、二阶姿态预测和 Level2 默认 `v_lead=1.0`。
显式配置的控制参数优先。

依赖链为 `mpcc_reliable_sprint_controller` → `mpcc_momentum_bar_controller` →
`mpcc_corridor_controller` → `mpcc_level2_controller` → `mpcc_controller` →
`my_controller`。这些父类、`momentum_bar_solver.py`、`gate_bar_solver.py`、
`corridor_solver.py`、`level2_solver.py`、`path_planning.py` 和 `astar.py`
均为运行依赖，需要保留。原生求解器使用项目现有的 acados 安装流程。
