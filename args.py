from dotmap import DotMap

ARGS = DotMap(
    # main code
    frozen_weights=None,
    seed=0,
    device='cuda',
    data_root='data/OWOD',
    batch_size=2,
    num_workers=8,
    lr_linear_proj_names=['reference_points', 'sampling_offsets'],
    lr=2e-4,
    lr_backbone_names=["backbone.0"],
    lr_linear_proj_mult=2e-5,
    sgd=False,
    weight_decay=1e-4,    
    pretrain=None,
    freeze_prob_model=False,
    start_epoch=0,
    nc_epoch=0,
    clip_max_norm=0.1,
    
    # build_model
    num_classes=81,
    num_queries=100,
    num_feature_levels=4,
    aux_loss=True,
    with_box_refine=False,
    two_stage=False,
    masks=False,
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    # obj_loss_coef=1, # duplicated
    # mask_loss_coef=,
    # dice_loss_coef=,
    dec_layers=6,
    hidden_dim=256,
    focal_alpha=0.25,
    # obj_temp=1, # duplicated
    # dataset_file=,
    
    # build_backbone
    lr_backbone=2e-5,
    backbone='dino_resnet50',
    dilation=False,
    position_embedding='sine',
    
    # build_deforamble_transformer
    nheads=8,
    enc_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    dec_n_points=4,
    enc_n_points=4,
    
    # build_matcher
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    
    # bash script
    output_dir='output',
    dataset='TOWOD',
    PREV_INTRODUCED_CLS=0,
    CUR_INTRODUCED_CLS=5,
    train_set='owod_t1_5classes_train', 
    test_set='owod_all_task_test',
    epochs=5,
    model_type='prob',
    obj_loss_coef=8e-4,
    obj_temp=1.3,
    exemplar_replay_selection=True,
    exemplar_replay_max_length=850,
    exemplar_replay_dir='',
    exemplar_replay_cur_file='',
)