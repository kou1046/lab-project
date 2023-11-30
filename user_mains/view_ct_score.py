from typing import Iterable
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from django.db.models import Prefetch
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torch.functional import F

import db_setup
from api import models
from user_mains.utils.common import checkpoint
from user_mains.utils.myplot import scatter_hist


FPS = 25
INTERVAL_SEC = 10
INTERVAL_FRAME = FPS * INTERVAL_SEC


class Encoder(nn.Module):
    def __init__(self, zdim: int):
        super().__init__()
        self.rnn = nn.LSTM(3, 128, batch_first=True, num_layers=3)
        self.mean_layer = nn.Linear(128, zdim)
        self.log_var_layer = nn.Linear(128, zdim)

    def forward(self, x):
        z, _ = self.rnn(x)

        mean = self.mean_layer(z[:, -1])
        log_var = self.log_var_layer(z[:, -1])

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, zdim: int):
        super().__init__()
        self.rnn = nn.LSTM(zdim, 128, batch_first=True, num_layers=3)
        self.fc = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())

    def forward(self, z):
        z = z.unsqueeze(1).repeat([1, (INTERVAL_FRAME * 2) // FPS + 1, 1])
        y, _ = self.rnn(z)
        y = self.fc(y)
        return y


class LSTMVAE(nn.Module):
    def __init__(self, zdim: int):
        super().__init__()
        self.encoder = Encoder(zdim)
        self.decoder = Decoder(zdim)

    def _sample_z(self, mean, log_var):
        ep = torch.randn_like(mean)
        z = mean + ep * torch.exp(log_var / 2)
        return z

    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = self._sample_z(mean, log_var)
        return z, mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mean, log_var = self.encode(x)
        y = self.decode(z)
        return y, mean, log_var


class VAELoss(nn.Module):
    def forward(self, pred_x, x, mean, log_var):
        bce_loss = F.binary_cross_entropy(pred_x, x, reduction="sum")
        kl_loss = 0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
        loss = bce_loss - kl_loss
        return loss


class ObserverSeenID:
    def __init__(self, ids):
        self.id_isseen = {id_: False for id_ in ids}

    def __call__(self, id_: int):
        if not id_ is not self.id_isseen:
            raise ValueError()

        self.id_isseen[id_] = True

    def unseen_id(self):
        return {id_ for id_ in self.id_isseen if not self.id_isseen[id_]}

    def reset(self):
        for id_ in self.id_isseen:
            self.id_isseen[id_] = False


def get_positive_ct_score_frames(group: models.Group) -> np.ndarray:
    frames = group.frames.select_related("ct_score")
    frame_numbers: list[int] = []
    scores: list[tuple[int, int, int, int, int]] = []

    for frame in frames:
        ct_score: models.CTScore = frame.ct_score
        frame_numbers.append(frame.number)
        scores.append(ct_score.score())

    positive_slope_indices = np.where(
        np.diff(scores, axis=0) > 0,
    )

    return np.array(frame_numbers)[positive_slope_indices[0]]


def get_group_member_actions_df(group_dir_path: Path) -> dict[int, pd.DataFrame]:
    action_csv_paths = group_dir_path.glob("*.csv")
    ret_dict: dict[int, pd.DataFrame] = {}
    for action_csv_path in action_csv_paths:
        person_id = action_csv_path.stem
        action_mat = pd.read_csv(action_csv_path)
        ret_dict[person_id] = action_mat

    return ret_dict


def column_sums_every_n_rows(arr, n):
    # 元の配列を n 行ごとに分割
    split_arrays = np.split(arr, range(n, arr.shape[0], n))

    # 各分割された配列の列ごとの合計を計算
    column_sums = [np.sum(split_array, axis=0) for split_array in split_arrays]

    return np.vstack(column_sums)


def create_data(group: models.Group):
    member_ids = [member.id for member in group.members.all()]
    observer = ObserverSeenID(member_ids)

    positive_ct_score_frames = get_positive_ct_score_frames(group)
    positive_ct_score_frames_member_actions: dict[str, dict[int, list[tuple[int, int, int, int, int]]]] = {}

    for min_frame_number, max_frame_number in tqdm(
        zip(positive_ct_score_frames - INTERVAL_FRAME, positive_ct_score_frames + INTERVAL_FRAME),
        total=positive_ct_score_frames.size,
    ):
        member_actions = {id_: [] for id_ in member_ids}

        # CTscoreが上昇したのフレーム数fx 付近のフレーム (fx - INTERVAL_FRAME) ~ (fx + INTERVAL_FRAME)
        positive_ct_score_around_frames = group.frames.filter(
            number__lte=max_frame_number, number__gte=min_frame_number
        ).prefetch_related(Prefetch("people", models.Person.objects.select_related("action", "box")))

        for frame in positive_ct_score_around_frames:
            frame: models.Frame

            if not frame.people.first():
                for id_ in member_actions:
                    member_actions[id_].append((0, 0, 0))
                continue

            for person in frame.people.all():
                person: models.Person

                action: models.Action = person.action
                member_actions[person.box.id].append(
                    (int(action.programming), int(action.watching_display), int(action.using_computer))
                )

                observer(person.box.id)

            for unseen_id in observer.unseen_id():
                member_actions[unseen_id].append((0, 0, 0))

            observer.reset()

        positive_ct_score_frames_member_actions[f"{min_frame_number}~{max_frame_number}"] = member_actions

    return positive_ct_score_frames_member_actions


class SeqDataset(data.Dataset):
    def __init__(self, groups: Iterable[models.Group]):
        super().__init__()

        dataset: list[list[tuple[int, int, int]]] = []  # batch_size * seq_size * vec_size(3)

        for group in groups:
            cache_function = checkpoint(f"{group.name}.pickle")(create_data)

            for frame_range, id_scores in cache_function(group).items():
                frame_min, frame_max = [int(frame) for frame in frame_range.split("~")]
                for person_id, scores in id_scores.items():
                    # scores: seq_size * vec_size
                    every_sec_sums = column_sums_every_n_rows(np.array(scores), FPS)
                    normalize_every_sec_sums = every_sec_sums / FPS
                    dataset.append(normalize_every_sec_sums)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return torch.Tensor(self.dataset[index])


def test(net: LSTMVAE, data: SeqDataset):
    def on_click(e):
        ax2.cla()
        z1 = e.xdata
        z2 = e.ydata

        pred_x = net.decode(torch.Tensor([[z1, z2]]).to(device))

        for score, label in zip(
            pred_x.to("cpu").squeeze(0).detach().numpy().T,
            ("programming", "watching_display", "micro_computer"),
        ):
            ax2.plot(range(0, len(score)), score, linestyle="--", label=label)

        fig2.legend()
        fig2.canvas.draw()

    fig2, ax2 = plt.subplots()
    net.eval()
    net.load_state_dict(torch.load("sample.pth", map_location=device))
    zs, _, _ = net.encode(torch.stack([t for t in data]).to(device))
    zs = zs.to("cpu").detach().numpy()
    zs_mean = np.mean(zs, axis=0)
    fig, axes = scatter_hist([zs[:, 0]], [zs[:, 1]], "k")
    axes[0].scatter(*zs_mean, marker="*", color="r")
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()


def train_vae(net: LSTMVAE, seqdataset: SeqDataset):
    batch_size = 64
    loader = data.DataLoader(seqdataset, batch_size=batch_size)
    criterion = VAELoss()
    opt = optim.Adam(net.parameters())
    max_epoch = 1000

    for epoch in range(1, max_epoch + 1):
        print(f"epoch: {epoch}/{max_epoch}")
        losses = []
        epoch_loss = 0
        for x in tqdm(loader):
            batch_size, seq_size, vec_size = x.shape
            x = x.to(device)
            pred_x, mean, log_var = net(x)

            loss = criterion(pred_x, x, mean, log_var)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_loss = loss.to("cpu").detach().numpy() / x.numel()
            epoch_loss += batch_loss

        epoch_loss /= len(loader)
        losses.append(epoch_loss)

        print(f"loss: {epoch_loss}")

    torch.save(net.state_dict(), "sample.pth")


device = "cuda" if torch.cuda.is_available() else "cpu"
groups = models.Group.objects.filter(name__in=["20230816_ube_G5", "20230816_ube_G2", "20230924_hagi_G1"])
seqdataset = SeqDataset(groups)
net = LSTMVAE(2).to(device)

test(net, seqdataset)
