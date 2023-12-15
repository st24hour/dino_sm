import torch
import numpy as np
from matplotlib import cm
import copy
from torchvision.utils import make_grid

from opt import get_opts

# dataset
from dataset import Patches_Dataset, DataAugmentationDINO, ValTransform
from torch.utils.data import DataLoader

# model
from models import vits_dict, MultiCropWrapper, DINOHead
from losses import DINOLoss

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# js
import os
import pandas as pd
from vision_transformer3 import vision_transformer3 
from collections import OrderedDict

# import lightning as pl
# import smdistributed.dataparallel.torch.torch_smddp 
# from pytorch_lightning.fabric.plugins.environments.lightning import LightningEnvironment

try:
    from sagemaker_training import environment
    #import smdistributed.dataparallel.torch.torch_smddp
    from pytorch_lightning.plugins.environments import LightningEnvironment    
    SMDDP = True
except:
    print('Using pytorch ddp.')
    SMDDP = False

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def keep_only_visual(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'visual' in k:
            name = k.split('visual.')[-1]
            new_state_dict[name] = v
    return new_state_dict

def att2img(att, cmap=cm.get_cmap('plasma')):
    """
    att: (H, W)
    """
    x = att.cpu().numpy()
    mi, ma = np.min(x), np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x_ = cmap(x)[..., :3] # (H, W, 3)
    x_ = torch.from_numpy(x_).permute(2, 0, 1) # (3, H, W)
    return x_


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=1e-6):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * \
                             (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class DINOSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        # model = vits_dict[hparams.arch]
        # student_backbone = model(patch_size=hparams.patch_size,
        #                          drop_path_rate=hparams.drop_path_rate)
        # self.teacher_backbone = model(patch_size=hparams.patch_size)
        default_kwargs = {
                                 # image
                                'image_size': 224,
                                'patch_size': hparams.patch_size, # if '32' in hparams.arch else 16,
                                'mode': 'base',
                                'embed_dim': 512
                            }
        print(default_kwargs)
        student_backbone=vision_transformer3(**default_kwargs)
        self.teacher_backbone = vision_transformer3(**default_kwargs)
        if hparams.pretrained_path: # fine-tune from pretrained dino
            print(f'loading pretrained model from {hparams.pretrained_path} ...')
            ckpt = torch.load(hparams.pretrained_path, map_location='cpu')
            # self.student.load_state_dict(ckpt['teacher'])
            state_dict = remove_module_from_state_dict(ckpt['state_dict'])
            state_dict = keep_only_visual(state_dict)
            msg=student_backbone.load_state_dict(state_dict, strict=True)
            print(msg)

        student_head = DINOHead(student_backbone.embed_dim, hparams.out_dim,
                                hparams.norm_last_layer)
        teacher_head = DINOHead(self.teacher_backbone.embed_dim, hparams.out_dim)

        self.student = MultiCropWrapper(student_backbone, student_head)

        self.teacher = MultiCropWrapper(self.teacher_backbone, teacher_head)
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())

        # teacher is not trained
        for p in self.teacher.parameters(): p.requires_grad = False

        self.loss = DINOLoss(hparams.out_dim,
                             hparams.local_crops_number+2,
                             hparams.warmup_teacher_temp,
                             hparams.final_teacher_temp,
                             hparams.warmup_teacher_temp_epochs,
                             hparams.num_epochs)

    def setup(self, stage=None):
        print('loading image paths ...')

        #########################################################
        csv_path = args.process_list_csv
        split_path = args.split_csv

        bags_dataset = pd.read_csv(csv_path)
        # split_set = pd.read_csv(split_path)
        # val_slides = ((split_set['val']).dropna()).to_list()
        # test_slides = ((split_set['test']).dropna()).to_list()
        # total = len(bags_dataset)
        total = 2

        h5_file_paths, slide_file_paths = [], []
        for bag_candidate_idx in range(total):
            if bags_dataset.loc[bag_candidate_idx, 'status'] == 'failed_seg':
                continue
            #if bags_dataset['slide_id'][bag_candidate_idx] in val_slides:
            #    continue
            #if bags_dataset['slide_id'][bag_candidate_idx] in test_slides:
            #    continue

            patient_id, wsi_id = bags_dataset['slide_id'][bag_candidate_idx].split('/')
            slide_id = bags_dataset['slide_id'][bag_candidate_idx].split(args.slide_ext)[0]
            bag_name = slide_id+'.h5'
            h5_file_paths.append(os.path.join(args.data_path, 'patches', bag_name))
            slide_file_paths.append(os.path.join(args.data_slide_dir, slide_id + args.slide_ext))
            #print('\ndataset generation progress: {}/{}'.format(bag_candidate_idx, total))
            #print(slide_id)


        transform = DataAugmentationDINO(args.global_crops_scale,
                                         args.local_crops_scale,
                                         args.local_crops_number)
        eval_transform = ValTransform()
        # self.train_dataset = Patches_Dataset(hparams.root_dir, 'train')
        self.train_dataset = Patches_Dataset(file_paths=h5_file_paths, wsis=slide_file_paths, custom_transforms=transform, pretrained=True, split='train')
        self.val_dataset = Patches_Dataset(file_paths=h5_file_paths, wsis=slide_file_paths, custom_transforms=eval_transform, pretrained=True, split='val')

        print(f'{len(self.train_dataset)} image paths loaded!')

        # self.val_dataset = copy.deepcopy(self.train_dataset)
        # self.val_dataset.split = 'val'

        ################################################################


    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for n, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if n.endswith(".bias") or len(p.shape) == 1:
                not_regularized.append(p)
            else:
                regularized.append(p)
        param_groups = [{'params': regularized},
                        {'params': not_regularized, 'weight_decay': 0.}]

        self.lr = args.lr * (args.batch_size*args.num_gpus/256)
        opt = torch.optim.AdamW(param_groups, self.lr)

        return opt

    def train_dataloader(self):
        self.loader = DataLoader(self.train_dataset,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 pin_memory=True,
                                 drop_last=True)

        # define schedulers based on number of iterations
        niter_per_ep = len(self.loader)
        self.lr_sch = cosine_scheduler(self.lr, 1e-6, args.num_epochs, niter_per_ep//args.num_gpus,
                                       args.warmup_epochs)
        # weight decay scheduler
        self.wd_sch = cosine_scheduler(args.weight_decay_init, args.weight_decay_end,
                                       args.num_epochs, niter_per_ep//args.num_gpus)
        # momentum scheduler
        self.mm_sch = cosine_scheduler(args.momentum_teacher, 1.0,
                                       args.num_epochs, niter_per_ep//args.num_gpus)

        return self.loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=args.num_workers,
                          batch_size=1, # validate one image
                          pin_memory=True)

    def training_step(self, batch, batch_idx):
        """
        batch: a list of "2+local_crops_number" tensors
               each tensor is of shape (B, 3, h, w)
        """
        opt = self.optimizers()
        # update learning rate, weight decay
        for i, param_group in enumerate(opt.param_groups):
            param_group['lr'] = self.lr_sch[self.global_step]
            if i == 0: # only the first group is regularized
                param_group['weight_decay'] = self.wd_sch[self.global_step]

        teacher_output = self.teacher(batch[:2])
        student_output = self.student(batch)
        loss = self.loss(student_output, teacher_output, self.current_epoch)

        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradient
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), args.clip_grad)
        # cancel gradient for the first epochs
        if self.current_epoch < args.ep_freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None
        opt.step()

        # EMA update for the teacher
        m = self.mm_sch[self.global_step]
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_((1-m)*ps.data)

        self.log('rates/lr', opt.param_groups[0]['lr'])
        self.log('rates/weight_decay', opt.param_groups[0]['weight_decay'])
        self.log('rates/momentum', m)
        self.log('train/loss', loss, True)

    # def validation_step(self, batch, batch_idx):
    #     img_orig, img_norm = batch

    #     w_featmap = img_norm.shape[-1] // hparams.patch_size
    #     h_featmap = img_norm.shape[-2] // hparams.patch_size

    #     atts = self.teacher_backbone.get_last_selfattention(img_norm)
    #     atts = atts[:, :, 0, 1:].reshape(1, -1, h_featmap, w_featmap)
    #     atts = torch.nn.functional.interpolate(atts,
    #                 scale_factor=hparams.patch_size, mode="nearest")[0] # (6, h, w)

    #     return {'attentions': atts, 'img': img_orig}

    # def validation_epoch_end(self, outputs):
    #     atts = outputs[0]['attentions']

    #     tb = self.logger.experiment
    #     tb.add_image('image', outputs[0]['img'][0], self.global_step)
    #     atts_vis = [att2img(att) for att in atts]
    #     tb.add_image('attentions', make_grid(atts_vis, nrow=3), self.global_step)


if __name__ == '__main__':
    args = get_opts(SMDDP)
    fix_random_seeds(args.seed)
    if SMDDP:
        #SageMaker
        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        args.local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
        args.train_dir = os.environ.get('SM_CHANNEL_TRAINING')    
        args.val_dir = args.train_dir
        args.backup_dir = '/opt/ml/checkpoints/backups/'
        sm_ckpt_dir = '/opt/ml/checkpoints/backups'
        os.environ['SM_CKPT_DIR'] = sm_ckpt_dir
        os.makedirs(sm_ckpt_dir, exist_ok=True)             
        default_root_dir = '/opt/ml/model'     
        os.environ['S3_LOG_PATH']= "/opt/ml/checkpoints/tf_logs"
        args.log_dir = os.environ['S3_LOG_PATH']     
        sm_env = environment.Environment() 
        env = LightningEnvironment()
        env.world_size = lambda: int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        env.global_rank = lambda: int(os.environ['OMPI_COMM_WORLD_RANK'])   

        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        args.num_nodes = args.world_size // args.gpus
        os.environ['WORLD_SIZE'] = str(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        if os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') is None:
            os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = '0'  
          
        os.environ['WORLD_SIZE'] = str(args.world_size)    
        os.environ['NODE_RANK'] = str(args.rank//args.gpus)
        os.environ['RANK']=str(args.rank)
        os.environ['LOCAL_RANK']=str(args.local_rank)  

    system = DINOSystem(args)

    ckpt_cb = ModelCheckpoint(dirpath=f'{args.output_dir}/ckpts/{args.exp_name}',
                              filename='{epoch:d}',
                              save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir=args.output_dir,
                               name=args.exp_name,
                               default_hp_metric=False)
    
    ddp = DDPStrategy(
        cluster_environment=env, 
        process_group_backend="smddp", 
        accelerator="gpu"
    )

    trainer = Trainer(max_epochs=args.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      precision=16 if args.fp16 else 32,
                      accelerator='auto',
                      devices=args.num_gpus,
                    #   strategy=DDPStrategy(find_unused_parameters=False)
                    #            if hparams.num_gpus>1 else None,
                      strategy=ddp,
                      num_sanity_val_steps=1)

    trainer.fit(system, ckpt_path=args.ckpt_path)