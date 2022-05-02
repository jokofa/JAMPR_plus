#
import os
import logging
from typing import Optional, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ImageMagickWriter
import torch

# configure matplotlib logger
logging.getLogger("matplotlib").setLevel(logging.WARNING)

if "PYCHARM_HOSTED" in os.environ:
    matplotlib.use("TKAgg")     # for use with GUI/IDE


class Viewer:
    """Renders routing environment by plotting changes of routing edges for each step."""
    def __init__(self,
                 locs: np.ndarray,
                 save_dir: Optional[str] = None,
                 as_gif: bool = True,
                 gif_naming: Optional[str] = None,
                 **kwargs):
        self.locs = locs
        self.save_dir = os.path.join(save_dir, "gifs") if save_dir is not None else None
        self.as_gif = as_gif
        if self.as_gif:
            matplotlib.use("Agg")   # for saving stream to file

        self.edges = None
        self.writer = None
        self.cmap = plt.get_cmap("tab20")

        plt.ion()
        # scale arrow sizes by plot scale, indicated by max distance from center
        max_dist_from_zero = np.max(np.abs(locs))
        self.hw = max_dist_from_zero * 0.02
        self.hl = self.hw * 1.25

        # create figure objects
        self.fig, self.ax = plt.subplots()

        self.plot_locs(self.locs, **kwargs)

        if save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        if not self.as_gif:
            plt.show(block=False)
        else:
            assert save_dir is not None, f"Must specify save_dir to create gif."
            metadata = dict(title='routing_env_render', artist='Matplotlib', comment='matplotlib2gif')
            self.writer = ImageMagickWriter(fps=2, metadata=metadata)
            if gif_naming is None:
                gif_naming = f"render.gif"
            if gif_naming[-4:] != ".gif":
                gif_naming += ".gif"
            outfile = os.path.join(self.save_dir, gif_naming)
            self.writer.setup(fig=self.fig, outfile=outfile)

    def plot_locs(self, locs: np.ndarray, add_idx: bool = True, **kwargs):
        # scatter plot of locations
        self.ax.scatter(locs[:, 0], locs[:, 1], c='k')
        self.ax.scatter(locs[0, 0], locs[0, 1], c='r', s=7 ** 2, marker='s')  # depot/start node
        if add_idx:
            # add node indices
            for i in range(1, locs.shape[0]):
                self.ax.annotate(i, (locs[i, 0], locs[i, 1]),
                                 xytext=(locs[i, 0]+0.012, locs[i, 1]+0.012),
                                 fontsize='medium', fontweight='roman')

    def update(self,
               buffer: Dict,
               cost: float,
               n_iters: Optional[int] = None,
               pause_sec: float = 0.5,
               new_locs: Optional[np.ndarray] = None,
               **kwargs):
        """Update current dynamic figure.

        Args:
            buffer: dictionary of data to plot
            cost: cost of current solution
            n_iters: current iteration
            pause_sec: float specifying seconds to wait before updating figure
            new_locs: optional new locations

        """
        if new_locs is not None:
            self.plot_locs(new_locs, **kwargs)

        [p.remove() for p in self.ax.patches]    # remove all previous patches
        num_tours = 0
        if 'edges' in buffer.keys():
            edges = buffer['edges']
            if isinstance(edges, np.ndarray):   # TSP
                edges = [edges]
            num_tours = len(edges)
            if num_tours > self.cmap.N:
                self.cmap = plt.get_cmap('jet', len(edges))
            for i, r in enumerate(edges):
                assert len(r.shape) == 2 and r.shape[0] == 2
                self._draw_edges(edges=r, color=self.cmap(i))
        elif 'tours' in buffer.keys():
            tours = buffer['tours']
            raise NotImplementedError
        else:
            ValueError("No 'edges' or 'tours' found in buffer.")

        iter_str = f"Iter: {n_iters}, " if n_iters is not None else ''
        self.ax.set_title(f"{iter_str}cost: {cost:.4f}, k: {num_tours}")
        self.ax.set_aspect('equal', adjustable='box')
        self._flush(pause_sec, **kwargs)

    def _flush(self, pause_sec: float = 0.1, **kwargs):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.as_gif and self.writer is not None:
            self.writer.grab_frame(**kwargs)
        else:
            plt.pause(pause_sec)

    def _draw_edges(self, edges: np.ndarray, color: str = "b", **kwargs):
        coords = self.locs[edges]
        X = coords[0, :, 0]
        Y = coords[0, :, 1]
        dX = coords[1, :, 0] - X
        dY = coords[1, :, 1] - Y
        for x, y, dx, dy in zip(X, Y, dX, dY):
            self.ax.arrow(x, y, dx, dy,
                          color=color,
                          linestyle='-',
                          head_width=self.hw,
                          head_length=self.hl,
                          length_includes_head=True,
                          **kwargs)

    def render_rgb(self) -> np.ndarray:
        """Returns the current figure as RGB value numpy array."""
        return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)\
            .reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    def save(self, path: Optional[str] = None):
        """Save the current figure on specified path. If path is None, uses default save_dir."""
        if self.as_gif:
            self.writer.finish()
            outfile = path if path is not None else os.path.join(self.save_dir, "final.gif")
            self.writer.saving(fig=self.fig, outfile=outfile, dpi=120)
            return True
        return False

    def close(self):
        """Finish and clean up figure and writer processing."""
        plt.clf()
        plt.close('all')
        plt.ioff()


#
# ============= #
# ### TEST #### #
# ============= #
def _test():
    from torch.utils.data import DataLoader
    from lib.routing import RPDataset, RPEnv, GROUPS, TYPES, TW_FRACS

    #SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    SAMPLE_CFG = {"groups": ['r'], "types": [1], "tw_fracs": [1.0]}
    LPATH = "./solomon_stats.pkl"
    SMP = 32
    N = 100
    BS = 16
    BS_ = BS
    MAX_CON = 5
    CUDA = False
    SEED = 123
    POMO = False #True
    N_POMO = 8
    if POMO:
        BS_ = BS // N_POMO
    GIF = False

    device = torch.device("cuda" if CUDA else "cpu")

    ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
    ds.seed(SEED)
    data = ds.sample(sample_size=SMP, graph_size=N)

    dl = DataLoader(
        data,
        batch_size=BS_,
        collate_fn=lambda x: x,  # identity -> returning simple list of instances
        shuffle=False
    )

    env = RPEnv(debug=True,
                device=device,
                max_concurrent_vehicles=MAX_CON,
                k_nbh_frac=0.4,
                pomo=POMO,
                num_samples=N_POMO,
                enable_render=True,
                plot_save_dir="./PLOTS" if GIF else None,
                )
    env.seed(SEED + 1)

    for batch in dl:
        env.load_data(batch)
        obs = env.reset()
        done = False
        i = 0
        start_tws = env._stack_to_tensor(batch, "tw")[:, :, 1]
        # print(env.coords[:, 0])

        while not done:
            # print(i)
            # select tour randomly and then select available node with earliest TW
            tr = torch.randint(MAX_CON, size=(BS,), device=device)
            t_nbh = obs.nbh[torch.arange(BS), tr]
            t_msk = obs.nbh_mask[torch.arange(BS), tr]
            nd = torch.zeros(BS, dtype=torch.long, device=device)
            for j, (nbh, msk, start_tw) in enumerate(zip(t_nbh, t_msk, start_tws)):
                available_idx = nbh[~msk]  # mask is True where infeasible
                idx = available_idx[start_tw[available_idx].argsort(-1, descending=False)]
                nd[j] = idx[0]
            # step
            obs, rew, done, info = env.step(torch.stack((tr, nd), dim=-1))
            env.render(as_gif=GIF, pause_sec=0.3)

            i += 1

        # print(info)
