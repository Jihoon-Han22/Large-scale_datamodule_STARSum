# -*- coding: utf-8 -*-
import numpy as np
#from knapsack import knapsack_ortools
from src.utils.knapsack_implementation import knapSack
import math
import torch
import pdb
# from knapsack_implementation import knapSack

def generate_summary(mode, ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    # print("positions")
    # print(positions)
    # print("n_frames")
    # print(n_frames)
    ypred_copy = ypred.clone().detach().numpy()
    # print("ypred_copy: ")
    # print(ypred_copy)
    # print("ypred_copy_size: ")
    # print(ypred_copy.size)
    n_segs = len(cps)
    # pdb.set_trace()
    #print("n_frames: ", n_frames)
    # n_frames = n_frames
    # pdb.set_trace()
    
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred_copy): 
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred_copy[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx][0]), int(cps[seg_idx][1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    # print("limits", limits)
    # print("nfps", nfps)
    # print("seg_score", seg_score)
    # print("len(nfps)", len(nfps))
    picks = knapSack(limits, nfps, seg_score, len(nfps))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    return summary