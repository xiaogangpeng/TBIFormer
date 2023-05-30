import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import torch_dct as dct
from models.net import TBIFormer
from tqdm import tqdm
from utils.opt import Options
from utils.dataloader import Data
from utils.metrics import FDE, JPE, APE
from utils.TRPE import bulding_TRPE_matrix




if __name__ == '__main__':
    opt = Options().parse()
    device = opt.device
    test_dataset = Data(dataset='mocap_umpm', mode=1, transform=False, device=device, opt=opt)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)


    model = TBIFormer(input_dim=opt.d_model, d_model=opt.d_model,
                        d_inner=opt.d_inner, n_layers=opt.num_stage,
                        n_head=opt.n_head , d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout, device=device,kernel_size=opt.kernel_size, opt=opt).to(device)



    checkpoint = torch.load('./checkpoints/best_epoch.model', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model loaded.')
    print(model)
    print(f"best_epoch: {checkpoint['epoch']}")


    body_edges = np.array(
    [[0,1], [1,2],[2,3],[0,4],
    [4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
    )


    with torch.no_grad():
        model.eval()
        loss_list=[]

        frame_idx = [5, 10, 15, 20, 25]
        n = 0
        ape_err_total = np.arange(len(frame_idx), dtype = np.float_)
        jpe_err_total = np.arange(len(frame_idx), dtype = np.float_)
        fde_err_total = np.arange(len(frame_idx), dtype = np.float_)

        for batch_i, batch_data in tqdm(enumerate(test_dataloader, 0)):
            n+=1
            input_seq, output_seq = batch_data
            B, N, _, D = input_seq.shape
            input_ = input_seq.view(-1, 50, input_seq.shape[-1])
            output_ = output_seq.view(output_seq.shape[0] * output_seq.shape[1], -1, input_seq.shape[-1])


            trj_dist = bulding_TRPE_matrix(input_seq.reshape(B,N,-1,15,3), opt)  #  trajectory similarity distance
            offset = input_[:, 1:50, :] - input_[:, :49, :]     #   dispacement sequence
            src = dct.dct(offset)

            rec_ = model.forward(src, N, trj_dist)
            rec = dct.idct(rec_)
            results = output_[:, :1, :]
            for i in range(1, 26):
                results = torch.cat(
                    [results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)],
                    dim=1)
            results = results[:, 1:, :]  # 3 15 45

            prediction = results.view(B, N, -1, 15, 3)
            gt = output_.view(B, N, -1, 15, 3)

            ape_err = APE(gt, prediction, frame_idx)
            jpe_err = JPE(gt, prediction, frame_idx)
            fde_err = FDE(gt, prediction, frame_idx)

            ape_err_total += ape_err
            jpe_err_total += jpe_err
            fde_err_total += fde_err




    print()

    print("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d}".format("Lengths", 200, 400, 600, 800, 1000))
    print("=== MPJPE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", jpe_err_total[0]/n,
                                                                                 jpe_err_total[1] / n,
                                                                                 jpe_err_total[2]/n,
                                                                                 jpe_err_total[3]/n,
                                                                                 jpe_err_total[4]/n ))
    print("=== Aligned MPJPE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", ape_err_total[0] / n,
                                                                                 ape_err_total[1] / n,
                                                                                 ape_err_total[2] / n,
                                                                                 ape_err_total[3] / n,
                                                                                 ape_err_total[4] / n))
    print("=== FDE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", fde_err_total[0] / n,
                                                                                 fde_err_total[1] / n,
                                                                                 fde_err_total[2] / n,
                                                                                 fde_err_total[3] / n,
                                                                                 fde_err_total[4] / n))

