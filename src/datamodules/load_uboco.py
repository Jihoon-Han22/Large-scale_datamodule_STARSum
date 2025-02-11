import torch
import cupy as cp
from utils.uboco_utils import TSMParser
from src.models.components.uboco.uboco_sum import UBoCoEncoder

def boundary_detection(
    feature,
    model,
    parser_config
    ):

    feature = feature.cuda()
    featured_frames = feature.shape[0]
    feat_list, tsm_list, tsm_total = model.forward(torch.unsqueeze(feature, 0))


    sim_size_ratio = parser_config['sim_size_ratio']
    topk_percent = parser_config['topk_percent']
    rtp_thr1 = parser_config['rtp_thr1']
    rtp_thr2 = parser_config['rtp_thr2']
    bd_gap_margin = parser_config['bd_gap_margin']
    
    tsm_parser = TSMParser(
        sim_size = int(featured_frames * sim_size_ratio),
        topk_percent = topk_percent,
        rtp_thr_size = int(featured_frames * rtp_thr1),
        rtp_thr2 = rtp_thr2,
        boundary_gap_margin = bd_gap_margin,
        mode = 'test'
        )
    
    tsm_total = cp.asarray(tsm_total.detach().squeeze())
    pred_boundary, pred_boundary_idx = tsm_parser.run_rtp(tsm_total)
    
    return pred_boundary, pred_boundary_idx