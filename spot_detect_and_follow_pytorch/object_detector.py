
def predict(model, image):
    preds = model.predict(image)
    #print(preds)
    # boxes, scores, classes
    pred = preds[0].boxes
    box = pred.xyxy.numpy()
    return box, pred.conf.numpy(), pred.cls.numpy()
