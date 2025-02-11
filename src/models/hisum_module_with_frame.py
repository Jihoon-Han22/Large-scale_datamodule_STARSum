import os
import torch
import numpy as np
import torch.nn as nn

from torchmetrics import MaxMetric
from pytorch_lightning import LightningModule

from src.models.components.stage_encoder import StageEnocoder
from src.utils.evaluation_metrics import evaluate_summary
from src.utils.generate_summary import generate_summary
# import pudb
import torch.multiprocessing
import torch.nn.init as init

class HiSumModule(LightningModule):
    def __init__(
        self,
        exp_type: str,
        dataset_type: str,
        dataset_split: int,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        histogram: bool,
        see_vis: bool,
        vis_dir: str,
    ):  
        torch.multiprocessing.set_sharing_strategy('file_system')

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = True

        self.net = net
        self.optimizer = None
        self.train_f1_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.test_f1_best = MaxMetric()
        self.criterion = nn.MSELoss()

        self.dataset_type = dataset_type
        self.exp_type = exp_type

        self.see_vis = see_vis
        self.vis_dir = vis_dir

    def forward(self, features, scene_boundary, shot_boundary, mask): # scene_mask, shot_mask
        return self.net(features, scene_boundary, shot_boundary, mask) # scene_mask, shot_mask

    def on_train_start(self):
        self.train_f1_best.reset()
        self.val_f1_best.reset()
        self.test_f1_best.reset()
        if self.see_vis:
            os.makedirs(self.vis_dir, exist_ok=True)

    def step(self, batch):
        invalid = ['user_summary', 'gtscore']
        # x = {k:v for k,v in batch.items() if k not in invalid}
        x = batch
        y = batch['gtscore']
        scene_score, shot_score, frame_score = self.forward(x['features'], x['scene_boundary'], x['shot_boundary'], x['mask']) # x['scene_mask'], x['shot_mask']
        scene_loss = self.criterion(scene_score.squeeze(), y.squeeze())
        shot_loss = self.criterion(shot_score.squeeze(), y.squeeze())
        frame_loss = self.criterion(frame_score.squeeze(), y.squeeze())
        return scene_score, shot_score, frame_score, scene_loss, shot_loss, frame_loss, y
    
    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        _, _, frame_score, scene_loss, shot_loss, frame_loss, targets = self.step(batch)
        
        loss = scene_loss + shot_loss + frame_loss
        # pdb.set_trace()
        #self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar("loss/train", loss, self.current_epoch) 
        # pudb.set_trace()
        mode = 'train'
        train_f1 = self.gen_sum_and_eval(mode, batch, frame_score)
        
        
        return {"loss": loss, "logits" : frame_score, "f1score": train_f1, "targets": targets, "scene_loss": scene_loss, "shot_loss": shot_loss, "frame_loss": frame_loss}

    # 원래는 없었음
    def training_epoch_end(self, outputs):
        train_f1_epoch = np.array([output['f1score'] for output in outputs])
        mean_f1 = train_f1_epoch.mean()

        train_loss = np.array([output['loss'].cpu() for output in outputs])
        mean_train_loss = train_loss.mean()
        
        scene_loss = np.array([output['scene_loss'].clone().detach().cpu() for output in outputs])
        mean_scene_loss = scene_loss.mean()
        
        shot_loss = np.array([output['shot_loss'].clone().detach().cpu() for output in outputs])
        mean_shot_loss = shot_loss.mean()
        
        frame_loss = np.array([output['frame_loss'].clone().detach().cpu() for output in outputs])
        mean_frame_loss = frame_loss.mean()
        
        self.train_f1_best.update(mean_f1)
        # self.log("train/loss", mean_train_loss, on_epoch=True, prog_bar=True, rank_zero_only=True, logger=True)

        self.logger.experiment.add_scalar("loss/train", mean_train_loss, self.current_epoch) 
        self.logger.experiment.add_scalar("scene_loss/train", mean_scene_loss, self.current_epoch) 
        self.logger.experiment.add_scalar("shot_loss/train", mean_shot_loss, self.current_epoch) 
        self.logger.experiment.add_scalar("frame_loss/train", mean_frame_loss, self.current_epoch)

        self.log("train/f1_epoch", mean_f1, on_epoch=True, prog_bar=True, rank_zero_only=True, logger=False)
        
        self.logger.experiment.add_scalar("train/f1_epoch", mean_f1, self.current_epoch)
        self.logger.experiment.add_scalar("train/f1_best", self.train_f1_best.compute(), self.current_epoch)

        print(f'\n@{self.current_epoch} Train loss: {mean_train_loss}, Train Score: {mean_f1}, Best Score: {self.train_f1_best.compute()}')
        print(f'\n@{self.current_epoch} scene_loss: {mean_scene_loss}, shot_loss: {mean_shot_loss}, frame_loss: {mean_frame_loss}')
        

    def validation_step(self, batch, batch_idx):
        _, _, frame_score, sc_loss, sh_loss, fr_loss, targets = self.step(batch)
        mode = 'val'
        if self.exp_type == 'original':
            val_f1 = self.gen_sum_and_eval(mode, batch, frame_score)
        
        # val_f1 = sum(val_f1) / len(val_f1)

        loss = sc_loss + sh_loss + fr_loss

        return {"loss": loss, "logits" : frame_score, "f1score": val_f1, "targets": targets, "sc_loss": sc_loss, "sh_loss": sh_loss, "fr_loss": fr_loss}
    
    def validation_epoch_end(self, outputs):
        val_f1_epoch = np.array([output['f1score'] for output in outputs])
        val_loss = np.array([output['loss'].cpu() for output in outputs])
        val_sc_loss = np.array([output['sc_loss'].clone().detach().cpu() for output in outputs])
        val_sh_loss = np.array([output['sh_loss'].clone().detach().cpu() for output in outputs])
        val_fr_loss = np.array([output['fr_loss'].clone().detach().cpu() for output in outputs])
        mean_f1 = val_f1_epoch.mean()
        mean_val_loss = val_loss.mean()
        mean_val_sc__loss = val_sc_loss.mean()
        mean_val_sh__loss = val_sh_loss.mean()
        mean_val_fr__loss = val_fr_loss.mean()

        self.val_f1_best.update(mean_f1)

        # self.log("val/loss", mean_val_loss.item(), on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True, logger=False)
        self.logger.experiment.add_scalar("val/loss", mean_val_loss.item(), self.current_epoch) 

        self.log("val/f1_epoch", mean_f1, on_epoch=True, prog_bar=True, rank_zero_only=True, logger=False)
        # self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True, rank_zero_only=True, logger=True)
        print(f'\n@{self.current_epoch} Val loss: {mean_val_loss}, Val Score: {mean_f1}, Best Score: {self.val_f1_best.compute()}')
        
        self.logger.experiment.add_scalar("val/f1_epoch", mean_f1, self.current_epoch)
        self.logger.experiment.add_scalar("val/f1_best", self.val_f1_best.compute(), self.current_epoch)

    def test_step(self, batch, batch_idx):
        video_name = batch['video_name']
        scene_boundary = batch['scene_boundary']
        shot_boundary = batch['shot_boundary']
        scene_score, shot_score, frame_score, _, _, _, _ = self.step(batch)
        
        self.save_vis(self.vis_dir, scene_score, shot_score, frame_score, scene_boundary, shot_boundary, name=video_name[0])

        # I added this line
        _, _, frame_score, sc_loss, sh_loss, fr_loss, targets = self.step(batch)
        mode = 'test'
        if self.exp_type == 'original':
            test_f1 = self.gen_sum_and_eval(mode, batch, frame_score)
        # test_f1 = sum(test_f1) / len(test_f1)

        loss = sc_loss + sh_loss + fr_loss

        #self.log("Test/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
        self.logger.experiment.add_scalar("Test/loss", loss.item(), self.current_epoch) 

        return {"loss": loss, "logits" : frame_score, "f1score": test_f1, "targets": targets}

    def test_epoch_end(self, outputs):
        test_f1_epoch = np.array([output['f1score'] for output in outputs])
        test_loss = np.array([output['loss'].cpu() for output in outputs])
        test_mean_f1 = test_f1_epoch.mean()
        mean_test_loss = test_loss.mean()
        
        self.test_f1_best.update(test_mean_f1)
        # self.log("test/f1_epoch", test_mean_f1, on_epoch=True, prog_bar=True, rank_zero_only=True)
        # self.log("test/f1_best", self.test_f1_best.compute(), on_epoch=True, prog_bar=True, rank_zero_only=True)
        print(f'\n@{self.current_epoch} Test loss: {mean_test_loss}, Test Score: {test_mean_f1}, Best Test Score: {self.test_f1_best.compute()}')
        
        self.logger.experiment.add_scalar("test/f1_epoch", test_mean_f1, self.current_epoch)
        self.logger.experiment.add_scalar("test/f1_best", self.test_f1_best.compute(), self.current_epoch)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        self.optimizer = optimizer
        return {
            "optimizer": optimizer
        }
    
    def xavier_init(model):
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            else:
                bound = torch.sqrt(6) / torch.sqrt(param.shape[0] + param.shape[1])
                param.data.uniform_(-bound, bound)

    '''
    HELPER FUNCTIONS
    '''
    def gen_sum_and_eval(self, mode, data, logit):
        # print("gen_sum_and_eval run")
        eval_metric = 'avg'
        # if self.dataset_type == 'summe':
        #     eval_metric = 'max'
        # elif self.dataset_type == 'tvsum':
        #     eval_metric = 'avg'
        # else: NotImplementedError
        # print("logit:")
        # print(logit)
        # print("logit.size:")
        # print(logit.size())
        num_of_data = len(data['video_name'])
        logit = logit.squeeze().cpu()
        # print("logit after squeeze:")
        # print(logit)
        # print("logit after squeeze size:")
        # print(logit.size())
        if (mode == 'train'):
            f_score_list = []
            for i in range(num_of_data):
                logit_i = logit[i]
                cps = data['change_points'][i]
                num_frames = data['n_frames'][i]
                nfps = data['n_frame_per_seg'][i].tolist()
                positions = data['picks'][i]
                gt_summary = data['gt_summary'][i]
                # print("in loop logit:")
                # print(logit_i)
                # print("in loop logit.size:")
                # print(logit_i.size())
                machine_summary = generate_summary(mode, logit_i, cps, num_frames, nfps, positions)
                f_score, _, _ = evaluate_summary(machine_summary, gt_summary, eval_metric)
                f_score_list.append(f_score)
            return np.average(f_score_list)

        if (mode == 'val') or (mode == 'test'):
            cps = data['change_points'][0]
            num_frames = data['n_frames'][0]
            nfps = data['n_frame_per_seg'][0].tolist()
            positions = data['picks'][0]
            gt_summary = data['gt_summary'][0]
            # gt_summary = data['gt_summary'][0].detach().numpy().copy()
            machine_summary = generate_summary(mode, logit, cps, num_frames, nfps, positions)
            f_score, _, _ = evaluate_summary(machine_summary, gt_summary, eval_metric)
            return f_score
    
    def save_vis(self, dir, scene_score, shot_score, frame_score, scene_boundary, shot_boundary, name=None):
        if name is None:
            name=self.global_step
                
        scene_boundary_file = os.path.join(dir, f"uboco_bd_{name}.pt")
        torch.save(scene_boundary, scene_boundary_file)
        
        shot_boundary_file = os.path.join(dir, f"shot_bd_{name}.pt")
        torch.save(shot_boundary, shot_boundary_file)
        
        sc_score_file = os.path.join(dir, f"sc_score_{name}.pt")
        torch.save(scene_score, sc_score_file)

        sh_score_file = os.path.join(dir, f"sh_score_{name}.pt")
        torch.save(shot_score, sh_score_file)

        fr_score_file = os.path.join(dir, f"fr_score_{name}.pt")
        torch.save(frame_score, fr_score_file)

    