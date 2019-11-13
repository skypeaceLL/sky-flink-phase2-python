# Apache Flink极客挑战赛——垃圾图片分类—复赛Python code

# 1. 代码说明
1. 采用Tensorflow slim (开源)
2. model.py为主入口文件。
3. 总共训练出三个模型文件，分别基于 resnet_v1_101（resnet_v1_50也可以）, inception_v4, inception_v3的ImageNet数据集的预训练checkpoint。
4. 训练各个模型的具体参数分别见model.py, model1_config.py, model2_config.py, model3_config.py
5. 训练过程中以指定时间间隔生成多个Checkpoint文件，训练完后，逐个检查Checkpoint的val_acc，选择最大值的checkpoint用来Export saved models(3个TF Saved Model）。
6. pre_train目录下预训练文件有3个，分别是：
```
pre_train/resnent_v1_101.ckpt
pre_train/inception_v4.ckpt
pre_train/inception_v3.ckpt
```
这些文件太大，无法上传到 git server上，请从以下地址下载并解压到pre_train目录下：
http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

# 2. 许可声明
你可以使用此代码用于学习和研究，但务必不要将此代码用于任何商业用途和比赛项目（slim那部分代码除外）。
