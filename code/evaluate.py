import tensorflow as tf

def evaluate_model(model, evaluate_ds):
    loss,accuracy = model.evaluate(evaluate_ds)
    print("loss: ", loss)
    print("accuracy: ", accuracy)
    return loss, accuracy