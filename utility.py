import numpy as np
# import os
from PIL import Image

def compute_precision_recall(score_A_np):
    array_5 = np.where(score_A_np[:, 1] == 5.0)
    array_7 = np.where(score_A_np[:, 1] == 7.0)
    print("len(array_5), ", len(array_5))
    print("len(array_7), ", len(array_7))

    mean_5 = np.mean((score_A_np[array_5])[:, 0])
    mean_7 = np.mean((score_A_np[array_7])[:, 0])
    medium = (mean_5 + mean_7) / 2.0
    print("mean_5, ", mean_5)
    print("mean_7, ", mean_7)
    print("medium, ", medium)

    array_upper = score_A_np[:, 0] >= medium
    array_lower = score_A_np[:, 0] < medium
    print("np.sum(array_upper.astype(np.float32)), ", np.sum(array_upper.astype(np.float32)))
    print("np.sum(array_lower.astype(np.float32)), ", np.sum(array_lower.astype(np.float32)))
    array_5_tf = score_A_np[:, 1] == 5.0
    array_7_tf = score_A_np[:, 1] == 7.0
    print("np.sum(array_5_tf.astype(np.float32)), ", np.sum(array_5_tf.astype(np.float32)))
    print("np.sum(array_7_tf.astype(np.float32)), ", np.sum(array_7_tf.astype(np.float32)))

    tn = np.sum(np.equal(array_lower, array_5_tf).astype(np.int32))
    tp = np.sum(np.equal(array_upper, array_7_tf).astype(np.int32))
    fp = np.sum(np.equal(array_upper, array_5_tf).astype(np.int32))
    fn = np.sum(np.equal(array_lower, array_7_tf).astype(np.int32))

    precision = tp / (tp + fp + 0.00001)
    recall = tp / (tp + fn + 0.00001)

    return tp, fp, tn, fn, precision, recall


def unnorm_img(img_np):
    img_np_255 = (img_np + 1.0) * 127.5
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    return img_np_uint8


def convert_np2pil(images_255):
    list_images_PIL = []
    for num, images_255_1 in enumerate(images_255):
        image_1_PIL = Image.fromarray(images_255_1)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL
    
def make_output_img(img_batch_5, img_batch_7, x_z_x_5, x_z_x_7, epoch, log_file_name, out_img_dir):
    (data_num, img1_h, img1_w, _) = img_batch_5.shape

    img_batch_5_unn = unnorm_img(img_batch_5).reshape(img_batch_5.shape[0], img_batch_5.shape[1], img_batch_5.shape[2])
    img_batch_7_unn = unnorm_img(img_batch_7).reshape(img_batch_7.shape[0], img_batch_7.shape[1], img_batch_7.shape[2])
    x_z_x_5_unn = unnorm_img(x_z_x_5).reshape(x_z_x_5.shape[0], x_z_x_5.shape[1], x_z_x_5.shape[2])
    x_z_x_7_unn = unnorm_img(x_z_x_7).reshape(x_z_x_7.shape[0], x_z_x_7.shape[1], x_z_x_7.shape[2])

    img_batch_5_PIL = convert_np2pil(img_batch_5_unn)
    img_batch_7_PIL = convert_np2pil(img_batch_7_unn)
    x_z_x_5_PIL = convert_np2pil(x_z_x_5_unn)
    x_z_x_7_PIL = convert_np2pil(x_z_x_7_unn)

    wide_image_np = np.ones(((img1_h + 1) * data_num - 1, (img1_w + 1) * 4 - 1), dtype=np.uint8) * 255
    wide_image_PIL = Image.fromarray(wide_image_np)
    for num, (ori_5, ori_7, xzx5, xzx7) in enumerate(zip(img_batch_5_PIL, img_batch_7_PIL, x_z_x_5_PIL, x_z_x_7_PIL)):
        wide_image_PIL.paste(ori_5, (0, num * (img1_h + 1)))
        wide_image_PIL.paste(xzx5, (img1_w + 1, num * (img1_h + 1)))
        wide_image_PIL.paste(ori_7, ((img1_w + 1) * 2, num * (img1_h + 1)))
        wide_image_PIL.paste(xzx7, ((img1_w + 1) * 3, num * (img1_h + 1)))

    wide_image_PIL.save(out_img_dir + "/resultImage_"+ log_file_name + '_' + str(epoch) + ".png")






