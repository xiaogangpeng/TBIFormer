import torch
import numpy as np

def APE(V_pred, V_trgt, frame_idx):
    V_pred = V_pred - V_pred[:, :, :, 0:1, :]
    V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
    return err * scale


def JPE(V_pred, V_trgt, frame_idx):
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2), dim=1).cpu().data.numpy().mean()
    return err * scale


# def ADE(V_pred, V_trgt, frame_idx):
#     scale = 1000
#     err = np.arange(len(frame_idx), dtype=np.float_)
#     for idx in range(len(frame_idx)):
#         err[idx] = torch.linalg.norm(V_trgt[:, :, :frame_idx[idx], :1, :] - V_pred[:, :, :frame_idx[idx], :1, :], dim=-1).mean(1).mean()
#     return err * scale


def FDE(V_pred,V_trgt, frame_idx):
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.linalg.norm(V_trgt[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :] - V_pred[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :], dim=-1).mean(1).mean()
    return err * scale