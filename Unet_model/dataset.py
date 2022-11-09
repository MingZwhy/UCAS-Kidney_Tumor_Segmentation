import numpy as np
import glob                         #用于遍历文件夹
import nibabel as nib               #用于处理医学nii图像
from imageio import imwrite         #用于保存图片
import os                           #用于创建文件夹

def process_haslabel_pic(load_nii_dir_path, save_image_dir_path, save_segemen_dir_path, num_of_nii = 210):
    """
    process_haslabel_pic 用于处理带标签的nii图像
    step1:从load_nii_dir_path读取nii图像并进行处理，得到可视化原图与其语义分割标签
    将可视化原图与其语义分割标签分别存于save_image_dir_path 和 save_segemen_dir_path目录下
    :param load_nii_dir_path: 读取nii图像的根目录
    :param save_image_dir_path: 保存image的根目录
    :param save_segemen_dir_path: 保存segementation的根目录
    :param num_of_nii: 默认为210，有0-209共210个带标签的nii资源
    :return:
    """

    """
    load_nii_dir_path
                    \              / imaging.nii.gz
                    | case_00000 --
                    | case_00001   \ segementation.gz
                    | case_00002
                    | ......
                    | case_00209
                    
    save_image_dir_path
    will be structed like:
    
    dir_path
            \               / image00000.png
            | case_00000   /  image00001.png
            | case_00001 --   ......
            | case_00002   \  image00600.png
            | ......        \ image00601.png
            | case_00209
            
            
    save_segementation_dir_path
    will be structed like:
    
    dir_path
            \               / segementation00000.png
            | case_00000   /  segementation00001.png
            | case_00001 --   ......
            | case_00002   \  segementation00600.png
            | ......        \ segementation00601.png
            | case_00209
    """

    #规范化路径
    if(save_image_dir_path[-1] != '/'):
        save_image_dir_path = save_image_dir_path + "/"
    if(save_segemen_dir_path[-1] != '/'):
        save_segemen_dir_path = save_segemen_dir_path + "/"

    #处理图片并保存
    for index in range(num_of_nii):
        case_path = "case_{:05d}".format(index)
        vol = nib.load(load_nii_dir_path + case_path + "/imaging.nii.gz")
        seg = nib.load(load_nii_dir_path + case_path + "/segmentation.nii.gz")
        print("processing case({:01d})".format(index))
        if (vol.shape[0] != seg.shape[0]):
            print("wrong, num of per vol != seg")
            break
        else:
            print("case ", index, " including ", vol.shape[0], "image and segmentation")

        # get_data
        vol = vol.get_data()
        seg = seg.get_data()
        vol = vol.astype(np.uint8)  # 保存时有一定损失
        seg = seg.astype(np.uint8)

        # save
        # 创建文件夹
        if (os.path.exists(save_image_dir_path + case_path) == False):
            os.makedirs(save_image_dir_path + case_path)
        if (os.path.exists(save_segemen_dir_path + case_path) == False):
            os.makedirs(save_segemen_dir_path + case_path)

        for pic_num in range(seg.shape[0]):
            save_image_path = save_image_dir_path + case_path + "/image{:05d}.png".format(pic_num)
            imwrite(save_image_path, vol[pic_num])
            save_segemen_path = save_segemen_dir_path + case_path + "/segmentation{:05d}.png".format(pic_num)
            imwrite(save_segemen_path, seg[pic_num])


def process_nolabel_pic(load_nii_dir_path, save_image_dir_path, begin_index=210, num_of_nii=18):
    """
    process_nolabel_pic 用于处理不带标签的nii图像
    step1:从load_nii_dir_path读取nii图像并进行处理，得到可视化原图 （无标签）
    将可视化原图存于save_image_dir_path目录下
    :param load_nii_dir_path: 读取nii图像的根目录
    :param save_image_dir_path: 保存image的根目录
    :param begin_index: 开始的下标
    :param num_of_nii: 默认为18，有210-227共18个带标签的nii资源
    :return:
    """

    """
    load_nii_dir_path
                    \              / imaging.nii.gz
                    | case_00000 --
                    | case_00001   \ segementation.gz
                    | case_00002
                    | ......
                    | case_00209

    save_image_dir_path
    will be structed like:

    dir_path
            \               / image00000.png
            | case_00000   /  image00001.png
            | case_00001 --   ......
            | case_00002   \  image00600.png
            | ......        \ image00601.png
            | case_00209
    """

    # 规范化路径
    if (save_image_dir_path[-1] != '/'):
        save_image_dir_path = save_image_dir_path + "/"

    index = begin_index
    end_index = begin_index + num_of_nii
    # 处理图片并保存
    while index < end_index:
        case_path = "case_{:05d}".format(index)
        vol = nib.load(load_nii_dir_path + case_path + "/imaging.nii.gz")
        print("processing case({:01d})".format(index))
        print("case ", index, " including ", vol.shape[0], "(only)image")

        # get_data
        vol = vol.get_data()
        vol = vol.astype(np.uint8)

        # save
        # 创建文件夹
        if (os.path.exists(save_image_dir_path + case_path) == False):
            os.makedirs(save_image_dir_path + case_path)

        for pic_num in range(vol.shape[0]):
            save_image_path = save_image_dir_path + case_path + "/image{:05d}.png".format(pic_num)
            imwrite(save_image_path, vol[pic_num])

        index = index + 1