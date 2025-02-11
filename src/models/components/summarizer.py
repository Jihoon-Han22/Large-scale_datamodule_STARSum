import torch
import torch.nn as nn

from src.models.components.stage_encoder import StageEnocoder
from src.models.components.score_predictor import ScorePredictor
from torch.nn.utils.rnn import pad_sequence


import pdb

class HiSumNet(nn.Module):
    def __init__(
        self, 
        stage: list, 
        depth: list, 
        input_dim: int, 
        num_heads: int, 
        mlp_ratio: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.stage = stage

        if stage[0]:
            self.scene_encoder = StageEnocoder(depth=depth[0], input_dim=input_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            self.scene_score_predictor = ScorePredictor(input_dim=input_dim)
        
        if stage[1]:
            self.shot_encoder = StageEnocoder(depth=depth[1], input_dim=input_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            self.shot_score_predictor = ScorePredictor(input_dim=input_dim)
        
        if stage[2]:
            self.frame_encoder = StageEnocoder(depth=depth[2], input_dim=input_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            self.frame_score_predictor = ScorePredictor(input_dim=input_dim)
        
    def forward(self, feature, scene_boundary, shot_boundary, mask=None): # scene_mask=None, shot_mask=None
        original = feature
        # print("feature")
        # print(feature)
        # print(feature.shape)
        # print("scene_boundary")
        # print(scene_boundary)
        # print("shot_boundary")
        # print(shot_boundary)
        # print("stage")
        # print(self.stage)
        if self.stage[0]:
            # print("inside stage 0")
            scene_mean_feature, for_repeat, new_mask = self.aggregate_in_boundary(feature, scene_boundary)
            # print("scene_mean_feature") # scene으로 자르고 mean 한 feature
            # print(scene_mean_feature.shape)
            # print("scene_mean_feature")
            # print(scene_mean_feature)
            # print("for_repeat")  # scene 사이 개수
            # print(for_repeat.shape)
            # print("mask")
            # print(new_mask.shape)

            scene_feature = self.scene_encoder(scene_mean_feature, mask = new_mask, stage=0)
            scene_score = self.scene_score_predictor(scene_feature, stage=0)
            # print("scene_feature")
            # print(scene_feature)
            # print(scene_feature.shape)

            # print("scene_score")
            # print(scene_score) 
            scene_score = self.repeat_interleave_to_frame_level(scene_score, for_repeat)
            scene_feat_stretched = self.repeat_interleave_to_frame_level(scene_feature, for_repeat)
            # print("original")
            # print(original)
            # print(original.shape)

            # print("scene_feat_stretched")
            # print(scene_feat_stretched)
            # print(scene_feat_stretched.shape)
            feature = original + scene_feat_stretched
            # print("scene features")
            # print(feature)
            # print(feature.shape)
        if self.stage[1]:
            # print("inside stage 1")
            shot_mean_feature, for_repeat, new_mask = self.aggregate_in_boundary(feature, shot_boundary)
            shot_feature = self.shot_encoder(shot_mean_feature, mask = new_mask, stage=1)
            shot_score = self.shot_score_predictor(shot_feature,stage=1)

            shot_score = self.repeat_interleave_to_frame_level(shot_score, for_repeat)
            shot_feat_stretched = self.repeat_interleave_to_frame_level(shot_feature, for_repeat)

            # print("original")
            # print(original)
            # print("shot_feat_stretched")
            # print(shot_feat_stretched)
            feature = original + shot_feat_stretched
            # print("shot features")
            # print(feature)
            # print(feature.shape)
        if self.stage[2]:
            # print("inside stage 2")
            frame_feature = self.frame_encoder(feature, stage=2, mask=mask)
            frame_score = self.frame_score_predictor(frame_feature, stage=2)

        return scene_score, shot_score
    
    '''
    HELPER FUNCTIONS
    '''

    def aggregate_in_boundary(self, src, boundary):
        B = len(boundary)

        segment_avg_list = []
        for_repeat_list = []
        for b in range(B):
            segments = torch.tensor_split(src[b], boundary[b].squeeze().cpu(), dim=0)
            # print("segments")
            # print(segments)
            segment_avg = []
            for_repeat = []
            for seg in segments:
                assert seg.shape[1] != 0, f"zero length segment!, {boundary}"
                for_repeat.append(seg.shape[0])
                segment_avg.append(torch.mean(seg, dim=0))
                # print("segment_avg")
                # print(segment_avg)
                # print(segment_avg[0].shape)

            # torch.stack(segment_avg, dim=1)
            for_repeat = torch.tensor(for_repeat).cuda()

            segment_avg_list.append(torch.stack(segment_avg, dim=0))
            for_repeat_list.append(for_repeat)
        
        n_scenes_max_len = max(list(map(lambda x: x.shape[0], for_repeat_list)))
        n_scenes = torch.LongTensor(list(map(lambda x: x.shape[0], for_repeat_list)))
        mask = (torch.arange(n_scenes_max_len)[None, :] < n_scenes[:, None]).cuda()

        segment_avg_list = pad_sequence(segment_avg_list, batch_first=True)
        for_repeat_list = pad_sequence(for_repeat_list, batch_first=True)

        return segment_avg_list, for_repeat_list, mask
        
    
    def repeat_interleave_to_frame_level(self, src, repeat):
        B = len(src)
        device = 'cuda:0'
        stretch_list = []
        for b in range(B):
            # print("src[b]")
            # print(src[b])
            # print(src[b].shape)
            # print("torch.tensor(repeat[b])")
            # print(torch.tensor(repeat[b]))
            # print(torch.tensor(repeat[b]).shape)
            stretch = torch.repeat_interleave(src[b], repeat[b].clone().detach(), dim=0)
            stretch = torch.squeeze(stretch, 1)
            stretch_list.append(stretch)
        return torch.squeeze(torch.stack(stretch_list), 0)
        