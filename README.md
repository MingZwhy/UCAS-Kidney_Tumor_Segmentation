# 肾脏肿瘤分割 by UCAS_机器学习 

#### 介绍
基于昇思MindSpore AI框架的肾脏肿瘤分割
参赛成员：
       陈远腾 赵昱杰 谷朝阳 彭睿思 游昆霖

#### 软件架构
基础框架：
tensorflow_gpu 2.10.0
所需python库：
	numpy           1.21.6 
	glob               
	os
	matplotlib
	SimpleITK      2.2.0
	nibabel          4.0.2

#### 使用说明

step1  git clone https://github.com/neheller/kits19   <br />
step2  安装github仓库README文件中所要求的的python库   <br />
step3  运行start_code中的get_imaging.py或get_imaging_v2.py文件 <br />
		  获取训练所需数据集。<br />

step4  git clone 本仓库 <br />
step5  运行main.py（不要改变刚刚下载的数据集所在文件夹名字）<br />
		  需要输入数据集所在文件夹的根目录<br />
		  例如：我的数据集所在路径为：<br />
		  D:/lumor_segementation/kits19-master/data/<br />
		  因此数据集所在文件夹的根目录为：<br />
		  D:/lumor_segementation/kits19-master/ <br />

运行命令：
python main.py --nii_data_dir_path D:/lumor_segementation/kits19-master/ <br />
						 --if_save_weights True <br />
						 --learn_rate 0.0001 <br />
						 --train_epochs 20 <br />

运行结束后将会在D:/lumor_segementation/kits19-master/目录下生成：<br />
p_image                                处理后的图像文件夹 <br />
p_segemen                           处理后的标签文件夹 <br />
evaluate_image                    处理后的无标签图像文件夹 <br />
model_weights                     训练后的模型参数文件夹 <br />
predict_result                       无标签图像预测结果文件夹 <br />

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

