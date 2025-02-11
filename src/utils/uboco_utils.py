import numpy as np
import cupy as cp

class TSMParser():
    def __init__(
        self,
        sim_size,
        topk_percent,
        rtp_thr_size,
        rtp_thr2,
        boundary_gap_margin = None,
        mode='train'
    ):
        self.sim_size = sim_size
        self.kernel_size = 2 * sim_size + 1
        assert rtp_thr_size > self.kernel_size, 'Too small rtp size threshold!'
        self.kernel = self.uboco_kernel()

        self.topk_percent = topk_percent
        self.rtp_thr_size = rtp_thr_size
        self.boundary_gap_margin = boundary_gap_margin
        self.rtp_thr2 = rtp_thr2
        self.rtp_thr_mean_max = None

        self.mode = mode
        self.boundary_pred = []
        self.diagonal_scores_container = None

    def uboco_kernel(self):
        kernel = cp.zeros((self.kernel_size, self.kernel_size))
        
        sim = cp.ones((self.sim_size, self.sim_size))
        neg_sim = cp.ones((self.sim_size, self.sim_size)) * (-1)

        kernel[0:self.sim_size, 0:self.sim_size] = sim
        kernel[0:self.sim_size, self.sim_size+1:self.kernel_size] = neg_sim
        kernel[self.sim_size+1:self.kernel_size, 0:self.sim_size] = neg_sim
        kernel[self.sim_size+1:self.kernel_size, self.sim_size+1:self.kernel_size] = sim

        return kernel

    '''
    diagonal convoultion from 'start' index to 'end' index
    Note that 'end' index is not included
    '''
    def diagonal_convolution(self, padded, start, end):
        diag_entry = cp.zeros(end-start)

        for i in range(start, end):
            diag_entry[i - start] = cp.sum(padded[i:i+self.kernel_size, i:i+self.kernel_size] * self.kernel)
        
        return diag_entry

    def dynamic_diag_convoltion(self, start_idx, tsm):
        tsm_size = tsm.shape[0]
        boundary_score = cp.zeros(tsm_size)
        boundary_score[self.sim_size:-self.sim_size] = self.diagonal_scores_container[(start_idx + self.sim_size):(start_idx + tsm_size - self.sim_size)] 
        
        padded = cp.pad(tsm, self.sim_size, 'constant', constant_values=0)
        boundary_score[:self.sim_size] = self.diagonal_convolution(padded, 0, self.sim_size)
        boundary_score[-self.sim_size:] = self.diagonal_convolution(padded, tsm_size-self.sim_size, tsm_size)

        boundary_score = cp.clip(boundary_score, a_min=1e-5, a_max=None)
        return boundary_score


    def tsm_parsing(self, start_idx, tsm):
        # print("Starting from ", start_idx)
        tsm_size = tsm.shape[0]
        if tsm_size < self.rtp_thr_size:
            # print("Small tsm stop")
            return

        boundary_score = self.dynamic_diag_convoltion(start_idx, tsm)

        if self.boundary_gap_margin > 1e-7 and self.mode=='test':
            boundary_score[:self.boundary_gap_margin] = 0
            boundary_score[-self.boundary_gap_margin:] = 0

        if cp.max(boundary_score) - cp.mean(boundary_score) < self.rtp_thr_mean_max:
            # print(f"Mean Max Stop {np.max(boundary_score)}, {np.mean(boundary_score)} {np.max(boundary_score) - np.mean(boundary_score)}")
            return
        
        topk = int(tsm_size * self.topk_percent)
        argsort = cp.argsort(boundary_score)[::-1]
        if self.mode == 'test':
            sample = int(argsort[0])
        elif self.mode == 'train':
            new_boundary_score = cp.zeros(tsm_size)
            new_boundary_score[argsort[:topk]] = boundary_score[argsort[:topk]]

            normalize = new_boundary_score / cp.sum(new_boundary_score)
            cp.random.seed(1997)
            sample = cp.random.multinomial(1, normalize, size=None)
            sample = int(cp.where(sample == 1)[0][0])
        else:
            print("Wrong TSM parser mode")
            pass
        
        tsm1 = tsm[0:sample, 0:sample]
        tsm2 = tsm[sample+1: , sample+1:]
        self.boundary_pred.append(start_idx + sample)
        # print("Sample!!!", start_idx + sample)
        # print(self.boundary_pred)

        self.tsm_parsing(start_idx, tsm1)
        self.tsm_parsing(start_idx + sample + 1, tsm2)
        return
    
    def clear_parser(self):
        self.boundary_pred.clear()

    def first_all_diag_conv(self, tsm):
        padded = cp.pad(tsm, self.sim_size, 'constant', constant_values=0)
        diagonal_scores = self.diagonal_convolution(padded, start=0, end=tsm.shape[0])
        self.diagonal_scores_container = cp.clip(diagonal_scores, a_min=1e-5, a_max=None)

        self.rtp_thr_mean_max = (cp.max(self.diagonal_scores_container) - cp.mean(self.diagonal_scores_container)) * self.rtp_thr2
        # print("thredhold 2:", self.rtp_thr_mean_max)

    def run_rtp(self, tsm):
        self.clear_parser()
        self.first_all_diag_conv(tsm)
        self.tsm_parsing(0, tsm)
        n_frames = tsm.shape[0]
        boundary_01 = cp.zeros(n_frames,)
        # print("n_frames:", n_frames)
        # print("self.boundary_pred:", self.boundary_pred)
        boundary_01[self.boundary_pred] = 1
        
        boundary_idx = sorted(self.boundary_pred)

        return boundary_01, boundary_idx