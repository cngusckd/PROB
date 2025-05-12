from ml_collections import config_dict


args = config_dict.ConfigDict()

# Parameters from dino_4scale.py
args.num_classes = 91
args.lr = 0.0001
args.param_dict_type = 'default'
args.lr_backbone = 1e-05
args.lr_backbone_names = ['backbone.0']
args.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
args.lr_linear_proj_mult = 0.1
args.ddetr_lr_param = False
args.batch_size = 2
args.weight_decay = 0.0001
args.epochs = 36
args.lr_drop = 30
args.save_checkpoint_interval = 1
args.clip_max_norm = 0.1
args.onecyclelr = False
args.multi_step_lr = False
args.lr_drop_list = [33, 45]

args.modelname = 'dino'
args.frozen_weights = None
args.backbone = 'resnet50'
args.use_checkpoint = False

args.dilation = False
args.position_embedding = 'sine'
args.pe_temperatureH = 20
args.pe_temperatureW = 20
args.return_interm_indices = [1, 2, 3]
args.backbone_freeze_keywords = None
args.enc_layers = 6
args.dec_layers = 6
args.unic_layers = 0
args.pre_norm = False
args.dim_feedforward = 2048
args.hidden_dim = 256
args.dropout = 0.0
args.nheads = 8
args.num_queries = 900
args.query_dim = 4
args.num_patterns = 0
args.pdetr3_bbox_embed_diff_each_layer = False
args.pdetr3_refHW = -1
args.random_refpoints_xy = False
args.fix_refpoints_hw = -1
args.dabdetr_yolo_like_anchor_update = False
args.dabdetr_deformable_encoder = False
args.dabdetr_deformable_decoder = False
args.use_deformable_box_attn = False
args.box_attn_type = 'roi_align'
args.dec_layer_number = None
args.num_feature_levels = 4
args.enc_n_points = 4
args.dec_n_points = 4
args.decoder_layer_noise = False
args.dln_xy_noise = 0.2
args.dln_hw_noise = 0.2
args.add_channel_attention = False
args.add_pos_value = False
args.two_stage_type = 'standard'
args.two_stage_pat_embed = 0
args.two_stage_add_query_num = 0
args.two_stage_bbox_embed_share = False
args.two_stage_class_embed_share = False
args.two_stage_learn_wh = False
args.two_stage_default_hw = 0.05
args.two_stage_keep_all_tokens = False
args.num_select = 300
args.transformer_activation = 'relu'
args.batch_norm_type = 'FrozenBatchNorm2d'
args.masks = False
args.aux_loss = True
args.set_cost_class = 2.0
args.set_cost_bbox = 5.0
args.set_cost_giou = 2.0
args.cls_loss_coef = 1.0
args.mask_loss_coef = 1.0
args.dice_loss_coef = 1.0
args.bbox_loss_coef = 5.0
args.giou_loss_coef = 2.0
args.enc_loss_coef = 1.0
args.interm_loss_coef = 1.0
args.no_interm_box_loss = False
args.focal_alpha = 0.25

args.decoder_sa_type = 'sa'
args.matcher_type = 'HungarianMatcher'
args.decoder_module_seq = ['sa', 'ca', 'ffn']
args.nms_iou_threshold = -1

args.dec_pred_bbox_embed_share = True
args.dec_pred_class_embed_share = True

args.use_dn = True
args.dn_number = 100
args.dn_box_noise_scale = 1.0
args.dn_label_noise_ratio = 0.5
args.embed_init_tgt = True
args.dn_labelbook_size = 91

args.match_unstable_error = True

args.use_ema = False
args.ema_decay = 0.9997
args.ema_epoch = 0

args.use_detached_boxes_dec_out = False

args.enc_scale = 3

args.drop_lr_now = False