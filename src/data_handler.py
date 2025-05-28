import h5py
import numpy as np
from .particle_data import ParticleData

# 文件结构示例
"""
/celestial_data
├── timestep_0
│   ├── positions   (N,3)
│   ├── velocities  (N,)
│   ├── masses      (N,)
├── timestep_1
│   └── ...
└── metadata
    ├── time        (T,) 时间戳，对应上述time_step
    ├── real_time   (T,) 模拟时间
    └── num_bodies  (T,) 各时间步天体数量
"""


class DynamicWriter:
    def __init__(self, filename):
        self.file = h5py.File(filename, 'w')
        self.meta = self.file.create_group("metadata")
        self.meta.create_dataset("time", (0,), maxshape=(None,), dtype=np.float64)
        self.meta.create_dataset("num_bodies", (0,), maxshape=(None,), dtype=np.int32)
        self.meta.create_dataset("real_time", (0,), maxshape=(None,), dtype=np.float64)
        self.current_step = 0

    def write_step(self, data: ParticleData, time: float):
        """ 写入单个时间步数据 """
        active_idx = data.active_indices
        if active_idx.size == 0:
            print("No active particles to plot.")
            return

        positions = data.position[active_idx]
        speeds = np.linalg.norm(data.velocity[active_idx], axis=1)
        masses = data.mass[active_idx]
        # 数据验证
        assert len({len(positions), len(speeds), len(masses)}) == 1
        N = len(positions)

        # 创建时间步分组
        step_group = self.file.create_group(f"timestep_{self.current_step}")
        step_group.create_dataset("positions", data=positions.astype(np.float32))
        step_group.create_dataset("velocities", data=speeds.astype(np.float32))
        step_group.create_dataset("masses", data=masses.astype(np.float32))

        # 更新元数据
        self.meta["time"].resize((self.current_step + 1,))
        self.meta["num_bodies"].resize((self.current_step + 1,))
        self.meta["real_time"].resize((self.current_step + 1,))
        self.meta["time"][self.current_step] = self.current_step  # 示例时间值
        self.meta["num_bodies"][self.current_step] = N
        self.meta["real_time"][self.current_step] = time

        self.current_step += 1

    def close(self):
        self.file.close()


class DynamicLoader:
    def __init__(self, filename):
        self.file = h5py.File(filename, 'r')
        self.max_step = self.file["metadata/num_bodies"].shape[0]
        self.current_step = 0

    def no_more(self):
        return self.current_step >= self.max_step

    def __call__(self, reverse=False):
        #if self.current_step >= self.max_step:
        #    self.current_step = 0  # 循环读取
        assert self.current_step < self.max_step

        # 获取当前步数据
        step_group = self.file[f"timestep_{self.current_step}"]
        N = self.file["metadata/num_bodies"][self.current_step]
        if reverse:
            self.current_step -= 1
        else:
            self.current_step += 1

        return (
            step_group["positions"][:],  # 全部位置
            step_group["velocities"][:],  # 全部速度
            step_group["masses"][:],  # 全部质量
        )

    def close(self):
        self.file.close()

    def goto(self, step):
        assert 0 <= step < self.max_step
        self.current_step = step

def get_all_speeds(filename):
    with h5py.File(filename, 'r') as f:
        num_steps = f["metadata/num_bodies"].shape[0]
        all_speeds = []
        for step in range(num_steps):
            speeds = f[f"timestep_{step}/velocities"][:]
            all_speeds.append(speeds)
        return np.concatenate(all_speeds)
