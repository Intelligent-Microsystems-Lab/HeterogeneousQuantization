# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Download data for unit tests and pretrained networks.

mkdir ../pretrained_efficientnet

wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz -P ../pretrained_efficientnet
tar zxf ../pretrained_efficientnet/efficientnet-lite0.tar.gz -C ../pretrained_efficientnet
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz -P ../pretrained_efficientnet
tar zxf ../pretrained_efficientnet/efficientnet-lite1.tar.gz -C ../pretrained_efficientnet
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite2.tar.gz -P ../pretrained_efficientnet
tar zxf ../pretrained_efficientnet/efficientnet-lite2.tar.gz -C ../pretrained_efficientnet
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite3.tar.gz -P ../pretrained_efficientnet
tar zxf ../pretrained_efficientnet/efficientnet-lite3.tar.gz -C ../pretrained_efficientnet
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz -P ../pretrained_efficientnet
tar zxf ../pretrained_efficientnet/efficientnet-lite4.tar.gz -C ../pretrained_efficientnet

# teacher network for distillation
wget https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz -P ../pretrained_efficientnet

mkdir ../unit_tests

wget https://notredame.box.com/shared/static/8whfsaisnjipyns49i2tv7jlqhnkb4vo.zip -O ../unit_tests/efficientnet.zip 
unzip ../unit_tests/efficientnet.zip -d ../unit_tests


wget https://notredame.box.com/shared/static/4idh6fkysgbj0nzd43ymxkirwttmc5pq.zip -O ../unit_tests/pc_modular.zip 
unzip ../unit_tests/pc_modular.zip -d ../unit_tests


wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O ../unit_tests/efficientnet/panda.jpg

mkdir ../pretrained_resnet
wget https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip -O ../pretrained_resnet/resnet18.zip
unzip ../pretrained_resnet/resnet18.zip -d ../pretrained_resnet

wget https://notredame.box.com/shared/static/75qdzq728n4a7qovtkujax0dxicwzrtv.h5 -O ../pretrained_resnet/resnet20_cifar10.h5

wget https://notredame.box.com/shared/static/z1pxy1b5poz8cdarg4wgm2jp4esg0mji.zip -O  ../pretrained_resnet/resnet20_cifar10.zip
unzip ../pretrained_resnet/resnet20_cifar10.zip -d ../pretrained_resnet


wget https://notredame.box.com/shared/static/rys28uy4o85hu1sntzmkcfo1mffd79ub.zip -O  ../pretrained_efficientnet/enet-lite0_best.zip
unzip ../pretrained_efficientnet/enet-lite0_best.zip -d ../pretrained_efficientnet


mkdir ../pretrained_mobilenetv2
wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth -O ../pretrained_mobilenetv2/mobilenet_v2-b0353104.pth
wget https://notredame.box.com/shared/static/xtwcx89qez08k9uxdpa20csboyl7leby.zip -O ../pretrained_mobilenetv2/mobilenetv2_fp32.zip
unzip ../pretrained_mobilenetv2/mobilenetv2_fp32.zip -d ../pretrained_mobilenetv2

wget https://notredame.box.com/shared/static/xswvymwy9sd3ircbwbn5k8k4uxsge07u.zip -O ../unit_tests/mobilnetv2_unit_test.zip
unzip ../unit_tests/mobilnetv2_unit_test.zip -d ../unit_tests


wget https://notredame.box.com/shared/static/ztcr84dfjg9j0qre0pr7bvtjvqgqyvoo.zip -O ../unit_tests/squeezenext_unit_test.zip
unzip ../unit_tests/squeezenext_unit_test.zip -d ../unit_tests

