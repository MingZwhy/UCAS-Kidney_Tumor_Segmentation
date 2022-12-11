import tensorflow as tf
import train

def evaluate_model(model, evaluate_ds, model_kind):
    print("begin to evaluate the model:")
    if(model_kind == "FCN_model"):
        loss,accuracy = model.evaluate(evaluate_ds)
        print("using FCN_model:")
        print("loss: ", loss)
        print("accuracy: ", accuracy)
        return loss, accuracy, 0

    #对Unet和Linknet模型，无法使用model.fit方法，需要自定义评估

    #定义损失函数
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    #定义评估参数
    eva_loss = tf.keras.metrics.Mean(name='eva_loss')
    eva_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='eva_acc')
    eva_iou = train.MeanIOU(3, name = 'eva_iou')

    #初始化状态
    eva_loss.reset_states()
    eva_acc.reset_states()
    eva_iou.reset_states()

    for images, labels in evaluate_ds:
        predictions = model(images)
        loss = loss_fn(labels, predictions)

        eva_loss(loss)
        eva_acc(labels, predictions)
        eva_iou(labels, predictions)

    if(model_kind == "Unet"):
        print("Using Unet_model:")
    else:
        print("Using LinkNet_model:")

    template1 = "evaluate --> Loss: {:.2f}, Accuracy: {:.2f}, IOU: {:.2f}"
    print(template1.format(eva_loss.result(), eva_acc.result() * 100, eva_iou.result()))
    return eva_loss.result(), eva_acc.result(), eva_iou.result()