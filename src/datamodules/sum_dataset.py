# -*- coding: utf-8 -*-
import h5py
import numpy as np
import json
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.datamodules.load_uboco import boundary_detection
from src.models.components.uboco.uboco_sum import UBoCoEncoder
import pdb

def scene_shot_union(scene_boundary, shot_boundary, margin=3, trust='scene'):
    n_scene = scene_boundary.shape[0]
    n_shot = shot_boundary.shape[0]
    # print("hello", n_scene, n_shot)

    union_shot_boundary = []
    adjust_scene_boundary = scene_boundary

    i, j, last = 0, 0, -10000
    while i < n_scene and j < n_shot:
        # print("log", i, j, scene_boundary[i], shot_boundary[j], last)
        if scene_boundary[i] < shot_boundary[j]:
            if trust == 'shot':
                if scene_boundary[i] - last > margin and shot_boundary[j] - scene_boundary[i] > margin:
                    union_shot_boundary.append(scene_boundary[i])
                    last = scene_boundary[i]
                elif scene_boundary[i] - last <= margin:
                    adjust_scene_boundary[i] = last
                elif shot_boundary[j] - scene_boundary[i] <= margin:
                    adjust_scene_boundary[i] = shot_boundary[j]
            else:
                union_shot_boundary.append(scene_boundary[i])
                last = scene_boundary[i]
            i += 1
        elif scene_boundary[i] > shot_boundary[j]:
            if trust == 'shot':
                union_shot_boundary.append(shot_boundary[j])
                last = shot_boundary[j]
            else:
                if shot_boundary[j] - last > margin and scene_boundary[i] - shot_boundary[j] > margin:
                    union_shot_boundary.append(shot_boundary[j])
                    last = shot_boundary[j]
            j += 1
        else:
            union_shot_boundary.append(scene_boundary[i])
            last = scene_boundary[i]
            i += 1
            j += 1
        
        if i == n_scene:
            while j < n_shot:
                union_shot_boundary.append(shot_boundary[j])
                j += 1
        
        if j == n_shot:
            while i < n_scene:
                union_shot_boundary.append(scene_boundary[i])
                i += 1

    return torch.tensor(union_shot_boundary), adjust_scene_boundary
            
class SumDataset(Dataset):
    def __init__(self, 
        exp_type, 
        h5_file_path, 
        train_test_json, 
        mode, 
        split_id,
        bd_model_path,
        uboco_config,
        tsm_parser_config
    ):
        self.mode = mode
        self.exp_type = exp_type
        print(f'Experiment : {self.exp_type}')
 
        self.split_id = split_id
        print(f'Loading {self.mode} dataset h5 file... : {h5_file_path}')
        
        # The file should be closed after opening or with block should be used
        with h5py.File(h5_file_path, 'r') as hf:
            self.video_data = h5py.File(h5_file_path, 'r')
        with open(train_test_json, 'r') as fd:
            json_load = json.load(fd)
            if self.exp_type == 'original':
                self.data = json_load
            # elif self.exp_type == 'imp' or self.exp_type == 'div':
            #     self.data = json_load
            else:
                NotImplementedError

        # self.list_frame_features, self.list_gtscores, self.list_user_summary, self.list_change_points, \
        # self.list_n_frames, self.list_picks, self.list_n_frame_per_seg = [],[],[],[],[],[],[]
        
        # if self.exp_type == 'imp' or self.exp_type == 'div':
        #     self.list_sum_ratios = []

        self.video_names = self.data[self.mode + '_keys']
        
        if os.path.exists(os.path.join(os.path.dirname(bd_model_path), f'scene_boundary_{mode}.pth')) and \
            os.path.exists(os.path.join(os.path.dirname(bd_model_path), f'shot_boundary_{mode}.pth')):
            self.list_scene_boundary = torch.load(os.path.join(os.path.dirname(bd_model_path), f'scene_boundary_{mode}.pth')) 
            self.list_shot_boundary = torch.load(os.path.join(os.path.dirname(bd_model_path), f'shot_boundary_{mode}.pth')) 
            print(f"Boundary file for {mode} loaded!")

        else:
            self.list_scene_boundary, self.list_shot_boundary = [],[]
            model = UBoCoEncoder(**uboco_config)
            model.load_state_dict(torch.load(bd_model_path))
            model.cuda()
            model.eval()
            print(f"Loading boundary detection model for {mode}... : {bd_model_path}")
            for video_name in self.video_names:
                features = torch.tensor(np.array(self.video_data[video_name + '/features']))
                # gtscore = torch.tensor(hdf[video_name + '/gtscore'])
                # user_summary = torch.tensor(hdf[video_name + '/user_summary'])
                # n_frames = np.array(hdf[video_name + '/n_frames'])
                # picks = torch.tensor(hdf[video_name + '/picks'], dtype=torch.int64)
                # n_frame_per_seg = torch.tensor(hdf[video_name + '/n_frame_per_seg'], dtype=torch.int64)
                change_points = torch.tensor(np.array(self.video_data[video_name + '/change_points']), dtype=torch.int64)


                # self.list_frame_features.append(features)
                # self.list_gtscores.append(gtscore)

                # self.list_user_summary.append(user_summary)
                # self.list_n_frames.append(n_frames)
                # self.list_picks.append(picks)
                # self.list_n_frame_per_seg.append(n_frame_per_seg)
                # self.list_change_points.append(change_points)

                _, pred_boundary_idx = boundary_detection(features, model, tsm_parser_config)
                pred_boundary_idx = torch.tensor(pred_boundary_idx)
                shot_boundary = torch.unique(change_points[:-1,1])
                if shot_boundary[0] < 1e-4:
                    shot_boundary = shot_boundary[1:]
                
                shot_boundary, scene_boundary = scene_shot_union(pred_boundary_idx, shot_boundary, margin=1, trust='shot')

                self.list_scene_boundary.append(scene_boundary)
                self.list_shot_boundary.append(shot_boundary)

            torch.save(self.list_scene_boundary ,os.path.join(os.path.dirname(bd_model_path), f'scene_boundary_{mode}.pth'))
            torch.save(self.list_shot_boundary ,os.path.join(os.path.dirname(bd_model_path), f'shot_boundary_{mode}.pth'))
            
            print(f"Boundary data for {mode} stored!")

                # if self.exp_type == 'imp' or self.exp_type == 'div':
                #     sum_ratio = torch.tensor(hdf[video_name + '/sum_ratio']) #! if div, imp set
                #     self.list_sum_ratios.append(sum_ratio)
        
        
        
    def __len__(self):
        """ Function to be called for the `len` operator of `SumDataset` Dataset. """
        self.len = len(self.data[self.mode+'_keys'])
        return self.len
    
    def __getitem__(self, index):
        """ Function to be called for the index operator of `SumDataset` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))
        n_frames = d['features'].shape[0]
        cps = np.array(self.video_data[video_name + '/change_points'])
        d['n_frames'] = np.array(n_frames)
        d['picks'] = np.array([i for i in range(n_frames)])
        d['change_points'] = cps
        d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
        d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)
        d['scene_boundary'] = self.list_scene_boundary[index]
        d['shot_boundary'] = self.list_shot_boundary[index]

        # if self.exp_type == 'imp' or self.exp_type == 'div':
        #     data['sum_ratio'] = self.list_sum_ratios[index]
        #     # d['video_boundary'] = np.array(self.video_data[video_name + '/video_boundary']) #! if div, imp set
    
        return d

class BatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore= [],[],[]
        cps, nseg, n_frames, picks, gt_summary, scene_boundary, shot_boundary = [], [], [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
                scene_boundary.append(data['scene_boundary'])
                shot_boundary.append(data['shot_boundary'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        max_len = max(list(map(lambda x: x.shape[0], features)))
        #n_scenes_max_len = max(list(map(lambda x: x.shape[0]+1, scene_boundary)))
        #n_shots_max_len = max(list(map(lambda x: x.shape[0]+1, shot_boundary)))
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        #n_scenes = torch.LongTensor(list(map(lambda x: x.shape[0]+1, scene_boundary)))
        #n_shots = torch.LongTensor(list(map(lambda x: x.shape[0]+1, shot_boundary)))
        #scene_mask = torch.arange(n_scenes_max_len)[None, :] < n_scenes[:, None]
        #shot_mask = torch.arange(n_shots_max_len)[None, :] < n_shots[:, None]
        
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        # batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask}
        batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask, \
                      'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps, \
                        'gt_summary': gt_summary, 'scene_boundary': scene_boundary, 'shot_boundary': shot_boundary} # 'scene_mask': scene_mask, 'shot_mask': shot_mask 
        return batch_data