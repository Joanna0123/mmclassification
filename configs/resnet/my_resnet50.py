_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/coffee.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    head=dict(
        type='LinearClsHead',
        num_classes=99,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# optimizer = dict(type='Adam', lr=0.003, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='fixed')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50, 100, 150])


runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)


load_from='models/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
