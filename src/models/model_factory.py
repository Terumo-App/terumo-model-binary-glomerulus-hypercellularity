from EfficientNet import Net

def get_model(model_name):
    if model_name == "efficienteNet":
        return Net(net_version="b0", num_classes=10).to(config.DEVICE)