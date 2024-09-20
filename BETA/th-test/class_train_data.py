import os
from shutil import copy2
import tensorflow as tf  

tf.config.set_visible_devices([], 'GPU')
thr = [100,200,300,500,700,1000,1500,2000,2500,3000]

def classify_and_copy_files(thnum, src_dir, dest_dir1, dest_dir2, num_files=5000):
    con=0
    class_model = tf.keras.models.load_model('./MODELS-th'+str(thnum)+'/class_model300')
    for i in range(num_files):
        file_path = os.path.join(src_dir, f"NewFile-HighDensity-4G{i}")
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = [float(line.split()[1]) for line in lines[:300]]

        data = tf.expand_dims(data, axis=0)


        prediction = class_model.predict(data)

        class_label = tf.argmax(prediction, axis=1).numpy()[0]
        if class_label == 1:
            copy2(file_path, os.path.join(dest_dir1, f"NewFile-HighDensity-4G{i}"))
        else:
            copy2(file_path, os.path.join(dest_dir2, f"NewFile-HighDensity-4G{i}"))
            con+=1
    print(con)



for thnum in thr:

    source_directory = '../BETA-class/L15'
    destination_directory_L1 = './train_data/'+str(thnum)+'/L1'
    destination_directory_L2 = './train_data/'+str(thnum)+'/L2'
    # 创建目标目录如果它们不存在
    os.makedirs(destination_directory_L1, exist_ok=True)
    os.makedirs(destination_directory_L2, exist_ok=True)
    # 调用函数
    classify_and_copy_files(thnum, source_directory, destination_directory_L1, destination_directory_L2)