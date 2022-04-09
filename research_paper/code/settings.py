base_architecture = 'vgg19'
### Modified by Caroline Cocca for gender classification dataset
img_size = 64
prototype_shape = (20, 128, 1, 1)
num_classes = 2
### Modified by Caroline Cocca for gender classification dataset
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '003'

### Modified by Caroline Cocca for gender classification dataset
data_path = './gender_classification_dataset/'
train_dir = data_path + 'train/'
test_dir = data_path + 'validation/'
train_push_dir = data_path + 'train/'
train_batch_size = 200
test_batch_size = 100
train_push_batch_size = 200
### Modified by Caroline Cocca for gender classification dataset

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

### Modified by Caroline Cocca for shorter train/test time
num_train_epochs = 5
num_warm_epochs = 2

push_start = 0
push_epochs = [4]
### Modified by Caroline Cocca for shorter train/test time
