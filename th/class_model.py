import tensorflow as tf
import load_trace
import numpy as np
import os
# 准备数据
tf.config.set_visible_devices([], 'GPU')
thr = [100,200,300,500,700,1000,1500,2000,2500,3000]
for th in thr:
    data_len = 300
    _, train_data1, _ = load_trace.load_trace('./threshold/th-'+str(th)+'/L1/')
    _, train_data2, _ = load_trace.load_trace('./threshold/th-'+str(th)+'/L2/')

    train_data1 = np.array(train_data1)[:, :data_len]
    train_data2 = np.array(train_data2)[:, :data_len]

    y_train1 = np.ones(len(train_data1))
    y_train2 = np.zeros(len(train_data2))

    x_train = np.concatenate((train_data1,train_data2), axis=0)
    y_train = np.concatenate((y_train1, y_train2), axis=0)

    data_with_labels = list(zip(x_train, y_train))
    np.random.shuffle(data_with_labels)
    x_train_shuffled, y_train_shuffled = zip(*data_with_labels)
    x_train_shuffled = np.array(x_train_shuffled)
    y_train_shuffled = np.array(y_train_shuffled)

    data_with_labels = list(zip(x_train_shuffled, y_train_shuffled))
    total_samples = len(data_with_labels)
    split_point = int(0.7 * total_samples)
    train_data = data_with_labels[:split_point]
    test_data = data_with_labels[split_point:]
    x_train_final, y_train_final = zip(*train_data)
    x_test_final, y_test_final = zip(*test_data)

    x_train_final = np.array(x_train_final)
    y_train_final = np.array(y_train_final)
    x_test_final = np.array(x_test_final)
    y_test_final = np.array(y_test_final)

    # 定义卷积和 LSTM 层的输入
    input_shape = (data_len, 1)
    input_data = tf.keras.layers.Input(shape=input_shape)

    # 卷积层
    conv_output = tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu')(input_data)
    conv_output = tf.keras.layers.AveragePooling1D(pool_size=2)(conv_output)
    conv_output = tf.keras.layers.Flatten()(conv_output)

    dense_output = tf.keras.layers.Dense(units=256, activation='relu')(conv_output)

    output = tf.keras.layers.Dense(units=2, activation='softmax')(dense_output)

    # 构建模型
    model = tf.keras.models.Model(inputs=input_data, outputs=output)

    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train_final, y_train_final, epochs=20)

    # 评估模型
    print(th)
    model.evaluate(x_test_final, y_test_final)
    if not os.path.exists('./MODELS-th'+str(th)+'/'):
        os.makedirs('./MODELS-th'+str(th)+'/')
    model_save_path = './MODELS-th'+str(th)+'/class_model'+str(data_len)
    model.save(model_save_path)