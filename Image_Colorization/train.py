from os import truncate
from model import *
from torch_helper import *
from utils import custom_plot

import matplotlib.pyplot as plt
import numpy as np


def train(args, x_train, y_train, x_test, y_test, colours, model_mode='base', model=None):
    # Set the maximum number of threads to prevent crash in Teaching Labs
    #####################################################################################
    # TODO: Implement this function to train model and consider the below items         #
    # 0. read the utils file and use 'process' and 'get_rgb_cat' to get x and y for     #
    #    test and train dataset                                                         #
    # 1. Create train and test data loaders with respect to some hyper-parameters       #
    # 2. Get an instance of your 'model_mode' based on 'model_mode==base' or            #
    #    'model_mode==U-Net'.                                                           #
    # 3. Define an appropriate loss function (cross entropy loss)                       #
    # 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
    # 5. Implement the main loop function with n_epochs iterations which the learning   #
    #    and evaluation process occurred there.                                         #
    # 6. Save the model weights                                                         #
    # Hint: Modify the predicted output from the model, to use loss function in step 3  #
    #####################################################################################
    torch.set_num_threads(5)
    np.random.seed(args.seed)
    save_dir = "outputs/" + args.experiment_name

    device = torch.device("cuda") if (torch.cuda.is_available() and args.gpu) else torch.device("cpu")

    print("Transforming data...")
    # Get X(grayscale images) and Y(the nearest Color to each pixel based on given color dictionary)
    train_rgb, train_grey = process(x_train, y_train, downsize_input=args.downsize_input, category_id=args.category_id)
    train_rgb_cat = rgb2label(train_rgb, colours, args.batch_size)
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input, category_id=args.category_id)
    test_rgb_cat = rgb2label(test_rgb, colours, args.batch_size)

    # LOAD THE MODEL
    if model_mode == 'base':
      model = BaseModel(kernel=args.kernel_size, num_filters=args.num_filters, num_colors=24).to(device)
    elif model_mode == 'custom_unet':
      model = CustomUNET(kernel=args.kernel_size, num_filters=args.num_filters, num_colors=24).to(device)
    else:
      model = CustomUnetWithResiduals(kernel=args.kernel_size, num_filters=args.num_filters, num_colors=24).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create the outputs' folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses = []
    valid_losses = []
    valid_accs = []
    
    # Training loop
    for epoch in range(args.epochs):

        epoch_train_loss = 0
        epoch_valid_loss = 0
        count = 0
        for i, (xs, ys) in enumerate(get_batch(train_grey, train_rgb_cat, args.batch_size)):
            count += 1
            model.train()
            images, labels = get_torch_vars(xs, ys, args.gpu)
            labels = torch.squeeze(labels)
            optimizer.zero_grad()
            prediction = model(images)
            batch_loss = loss_function(prediction, labels)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()
        train_losses.append(epoch_train_loss/count)

        count = 0 
        total = 0
        correct = 0
        for i, (xs, ys) in enumerate(get_batch(test_grey, test_rgb_cat, args.batch_size)):
            count += 1
            model.eval()
            images, labels = get_torch_vars(xs, ys, args.gpu)
            labels = torch.squeeze(labels)
            prediction = model(images)
            batch_loss = loss_function(prediction, labels)
            epoch_valid_loss += batch_loss.item()

            _, prediction = torch.max(prediction, dim=1)
            correct += (prediction == labels).sum()
            total += labels.numel()
        valid_accs.append((100 * (correct / total)).cpu())
        valid_losses.append(epoch_valid_loss/count)
        print(f"epoch: {epoch},  train loss: {train_losses[-1]}, validation loss: {valid_losses[-1]}, validation accuracy:{valid_accs[-1]}")
    custom_plot(images.cpu(), labels.cpu(), prediction.cpu(),colours,path=save_dir + "/sample_output.png",visualize=True)
    # Plot training-validation curve
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve.png")

    plt.subplot(1,2,2)
    plt.plot(valid_accs)
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve_acc.png")

    if args.checkpoint:
        print('Saving model...')
        torch.save(model.state_dict(), f"{model_mode}.pth")

    return model
