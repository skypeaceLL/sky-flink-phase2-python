# Apache Flink极客挑战赛——垃圾图片分类 -- 复赛Python code

说明：
1. 采用Tensorflow slim (开源)
2. model.py为主入口文件。
3. 总共训练出三个模型文件，分别基于 resnet_v1_101（resnet_v1_50也可）, inception_v4, inception_v3的ImageNet 预训练checkpoint。
4. 训练各个模型的具体参数分别见model.py, model1_config.py, model2_config.py, model3_config.py
5. 训练过程中按指定时间间隔生成多个Checkpoint文件，训练完后，逐个检查Checkpoint的val_acc，选择最大值的checkpoint来Export saved model。
6. 预训练文件名分别是pre_train/resnent_v1_101.ckpt, pre_train/inception_v4.ckpt, pre_train/inception_v3.ckpt。
这些文件太大，无法上传到 git server上。需要从以下地址下载：
http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

