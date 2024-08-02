import math
import os
import random

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle
import numpy as np
import torch
from utils.nn import utils as utils

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(model, criterion, optimizer, writer, trainloader, val_loader, embed_fn, saveloc, res, normalization_nums, epochs=1, totem_name=''):
    if not os.path.exists(os.path.join(saveloc,'models')):
        os.mkdir(os.path.join(saveloc,'models'))
    in_mins, in_maxs, out_mins, out_maxs = normalization_nums
    epoch_loss = 0.0
    avg_epoch_loss = 10000000
    epoch_losses = []
    val_losses = []
    batch_losses = []
    print(f"{len(trainloader)} batches to iterate through per epoch")
    for epoch in range(epochs):                          # loop over the dataset_helpers multiple times
        print(f'Epoch {epoch}:')
        val_inputs, val_labels = next(iter(val_loader))
        for i, data in enumerate(trainloader, 0):
            inputs, true_outputs = data                 # get the inputs; data is a list of [inputs, labels]
            # Adding noise to the input coords
            # print("adding noise")
            # inputs = inputs + torch.randn_like(inputs) * 0.01
            inputs = embed_fn(inputs)

            outputs = model(inputs)                     # forward + backward + optimize
            # COMPARE to predicting the average
            # for i in range(len(outputs)):
            #     outputs[i] = torch.Tensor([-0.0017, 0.3117])
            loss = criterion(outputs, true_outputs)
            optimizer.zero_grad()                       # zero the parameter gradients
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            # print(f"SAVING MODEL FOR BATCH {i}...")
            # torch.save(model.state_dict(), saveloc + f'/models/model_e0_batch{i}.pt')

            if (epoch == 0 or epoch == 1 or epoch == 49 or epoch == 99) and i == 2:
                inputs_uvc_unn = utils.unnormalize(data[0], in_mins, in_maxs).detach().numpy()
                preds_uvt_unn = utils.unnormalize(outputs, out_mins, out_maxs).detach().numpy()
                labels_uvt_unn = utils.unnormalize(data[1], out_mins, out_maxs).detach().numpy()
                # Generate ~50 unique random indices
                size = 50 if 50 <= outputs.shape[0] else outputs.shape[0]
                indices = np.random.choice(inputs_uvc_unn.shape[0], size, replace=False)
                # Select the corresponding rows from the arrays
                pruned_inputs_unn = inputs_uvc_unn[indices]
                pruned_preds_unn = preds_uvt_unn[indices]
                pruned_labels_unn = labels_uvt_unn[indices]

                pruned_preds = outputs.detach().numpy()[indices]
                pruned_labels = data[1].detach().numpy()[indices]
                plt.figure(figsize=(6, 6))

                plt.scatter(pruned_inputs_unn[:, 0], res - pruned_inputs_unn[:, 1], color='green', marker='x', alpha=0.5,
                            label='inputs')
                plt.scatter(pruned_preds_unn[:, 0], res - pruned_preds_unn[:, 1], color='blue', marker='x', alpha=0.5,
                            label='preds')
                plt.scatter(pruned_labels_unn[:, 0], res - pruned_labels_unn[:, 1], color='orange', marker='x', alpha=0.5,
                            label='ground truth')
                plt.xlim(0, res)
                plt.ylim(0, res)
                plt.legend()

                # Draw a circle around the actual totem location
                totem_center = [128, 47] #TODO un-hardcode! This is for r256
                radius_approx = 45
                add_circle(totem_center, radius_approx)
                title = f'Predictions on a subset of a random batch during epoch {epoch}, MSE = {loss:.2e}'
                plt.title(title)
                savename = saveloc+ f'/random_sample_visual_e{epoch}.png'
                plt.savefig(savename, bbox_inches='tight')
                plt.close()

                plt.scatter(pruned_preds[:, 0], pruned_preds[:, 1], color='blue', marker='x', alpha=0.5,
                            label='preds')
                plt.scatter(pruned_labels[:, 0], pruned_labels[:, 1], color='orange', marker='x', alpha=0.5,
                            label='ground truth')
                plt.legend()
                title = f'(Normalized) Predictions on a subset of a random batch of during after epoch {epoch}, MSE = {loss:.2e}'
                plt.title(title)
                plt.gca().set_aspect('equal', adjustable='box')

                savename = saveloc + f'/random_sample_visual_norm_e{epoch}.png'
                plt.savefig(savename, bbox_inches='tight')
                plt.close()

            # if i % 4 == 3:    # every 4 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                              batch_loss,
                              epoch * len(trainloader) + i)
            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            # writer.add_figure('predictions vs. actuals',
            #                 plot_classes_preds(net, inputs, labels),
            #                 global_step=epoch * len(trainloader) + i)
            print(f'average epoch loss across batches so far in epoch {epoch}: {epoch_loss / (i + 1)}  /  batch loss: {batch_loss}')
            avg_epoch_loss = epoch_loss / (i + 1)
        epoch_losses.append(avg_epoch_loss)
        sub_epochs = [epoch + j / float(i + 1) for epoch in range(epochs) for j in range(i + 1)]

        epoch_loss = 0
        val_inputs = embed_fn(val_inputs)
        val_outputs = model(val_inputs)  # forward + backward + optimize
        loss = criterion(val_outputs, val_labels)
        val_loss = loss.item()
        val_losses.append(val_loss)
        print(f"Validation loss for epoch {epoch - 1}: {val_loss}")
        writer.add_scalars('loss/', {
        'val loss': val_loss,
        'epoch loss': avg_epoch_loss,
        },
                          epoch)

        print(f"SAVING MODEL FOR EPOCH {epoch}...")
        torch.save(model.state_dict(), saveloc+f'/models/model_{epoch}.pt')

    print('Finished Training')
    writer.flush()
    writer.close()
    # Get final training loss:
    total_train_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, true_outputs = data
        inputs = embed_fn(inputs)

        outputs = model(inputs)
        loss = criterion(outputs, true_outputs)

        batch_loss = loss.item()

        total_train_loss += batch_loss
        avg_train_loss = total_train_loss / (i + 1)

    print(f'Training error: {avg_train_loss}')

    print("Len sub epochs: ", len(sub_epochs))
    plt.plot(sub_epochs, batch_losses, label='batch_loss', linewidth=0.5)
    plt.plot(range(epochs), epoch_losses, label='epoch_loss')
    plt.plot(range(epochs), val_losses, label='val_loss')
    # Create a ScalarFormatter object
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    # Apply the formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.ylim(0, .01)
    plt.suptitle(f"[{totem_name}] Training curve for res={res}; epochs={epochs}, batch size={trainloader.batch_size}", fontsize=10)
    plt.title(f"Final training err: {avg_train_loss:.2e}; Final val err: {val_loss:.2e}", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    savename = saveloc+'/training_curve.png'
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    savename = saveloc+'/batch_losses'
    np.save(savename, batch_losses)
    savename = saveloc+'/sub_epochs'
    np.save(savename, sub_epochs)
    savename = saveloc+'/epoch_losses'
    np.save(savename, epoch_losses)
    savename = saveloc+'/val_losses'
    np.save(savename, val_losses)

    return avg_train_loss #avg_epoch_loss

def test(model, criterion, testloader, embed_fn, writer, in_mins, in_maxs, out_mins, out_maxs, new_in_mins, new_in_maxs, new_out_mins, new_out_maxs, do_manual_test=False, res=None):
    total_samples = 0
    total_error = 0

    # Set the model to evaluation mode
    model.eval()

    # if do_manual_test:
    #     tv_inputs = torch.load(f'{res}_testset_in.pt')
    #     tv_outputs = torch.load(f'{res}_testset_out.pt')
    #
    #     testset = TotemUnwarpDataset(tv_inputs, tv_outputs)
    #     _, test_dataset = random_split(testset, [len(testset) // 2, math.ceil(
    #         len(testset) / 2)])
    #     testloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset),
    #                                                   worker_init_fn=seed_worker,
    #                                                   generator=g, shuffle=False)
    # Disable gradient computation
    with torch.no_grad():        
        for inputs, labels in testloader:
            inputs = embed_fn(inputs)

            # Forward pass
            outputs = model(inputs)

            #Unnormalize then renormalize (incase using manual test set iwth diff norm coeffs)
            outputs_un = utils.unnormalize(outputs, out_mins, out_maxs)
            outputs, _, _ = utils.normalize(outputs_un, new_out_mins, new_out_maxs)
            # Calculate loss
            loss = criterion(outputs, labels)

            # Update the total error and sample count
            total_error += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    # Calculate the average test error
    test_error = total_error / total_samples

    return testloader, test_error

def inference(model, inputs, embed_fn, in_mins, in_maxs, out_mins, out_maxs):
    total_samples = 0
    total_error = 0
    inputs, _, _ = utils.normalize(inputs, in_mins, in_maxs)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        inputs = embed_fn(inputs)

        # Forward pass
        outputs = model(inputs)
    outputs = outputs[:, 0:2]
    outputs = utils.unnormalize(outputs, out_mins, out_maxs)
    return outputs

def visualize_predictions(model, in_mins, in_maxs, out_mins, out_maxs, new_in_mins, new_in_maxs, new_out_mins, new_out_maxs, embed_fn, size,
                          train_err=None, test_err=None, trainloader=None, testloader=None,
                          is_inference=False, test=True, path_to_exp='.', ext='.', inference_data=None, inference_saveloc=None, is_embedded=False,
                          lr=None, batch_size=None, epochs=None, skip=2, testset=None, is_man=False):
    # Create empty lists to store points for scatter plot
    if not test:
        raise NotImplementedError
    if not is_inference:
        if train_err is None or test_err is None or not trainloader or not testloader:
            raise ValueError("train, test err, train, test loader must all have valid values (not None)")
        uv_cam_predictions_train = []
        uv_cam_labels_train = []
        uv_tot_inputs_train = []

        uv_cam_predictions_test = []
        uv_cam_labels_test = []
        uv_tot_inputs_test = []

        dataloader = trainloader
        sk = 0
        for inputs, labels in dataloader:
            # if sk % skip == 0:
            #     sk += 1
            #     continue
            embedded_inputs = embed_fn(inputs)
            preds = model(embedded_inputs)
            if is_embedded:
                # take the first n,0:2 elements of the nx42 dimensional embedding, which went through the identity
                preds = preds[:,0:2] # shape e.g. Tensor: (50, 2)
            uv_cams = utils.unnormalize(preds, out_mins, out_maxs).detach().numpy() # shape e.g. ndarray (50, 2)
            uv_cams_true = utils.unnormalize(labels, out_mins, out_maxs).detach().numpy()
            uv_tots = utils.unnormalize(inputs, in_mins, in_maxs).detach().numpy()

            # Append scatter points to the respective lists
            uv_cam_predictions_train.append(uv_cams)
            uv_cam_labels_train.append(uv_cams_true)
            uv_tot_inputs_train.append(uv_tots)

        # Concatenate all scatter points
        uv_cam_predictions_train = np.concatenate(uv_cam_predictions_train)
        uv_cam_labels_train = np.concatenate(uv_cam_labels_train)
        uv_tot_inputs_train = np.concatenate(uv_tot_inputs_train)

        dataloader = testloader
        for inputs, labels in dataloader:
            embedded_inputs = embed_fn(inputs)
            preds = model(embedded_inputs)
            if is_embedded:
                # take the first n,0:2 elements of the nx42 dimensional embedding, which went through the identity
                preds = preds[:, 0:2]

            uv_cams_test = utils.unnormalize(preds, out_mins, out_maxs).detach().numpy()
            uv_cams_true_test = utils.unnormalize(labels, new_out_mins, new_out_maxs).detach().numpy()
            uv_tots_test = utils.unnormalize(inputs, new_in_mins, new_in_maxs).detach().numpy()

            # Append scatter points to the respective lists
            uv_cam_predictions_test.append(uv_cams_test)
            uv_cam_labels_test.append(uv_cams_true_test)
            uv_tot_inputs_test.append(uv_tots_test)

        # Concatenate all scatter points

        uv_cam_predictions_test = np.concatenate(uv_cam_predictions_test)
        uv_cam_labels_test = np.concatenate(uv_cam_labels_test)
        uv_tot_inputs_test = np.concatenate(uv_tot_inputs_test)

        # Plot scatter points with unique labels
        # plt.scatter(uv_cam_predictions_train[:, 0], size - uv_cam_predictions_train[:, 1], color='blue', marker='x', alpha=0.5,label='predictions')
        # plt.scatter(uv_cam_labels_train[:, 0], size - uv_cam_labels_train[:, 1], color='green', alpha=0.5,label='ground truth')
        # plt.scatter(uv_tot_inputs_train[:, 0], size - uv_tot_inputs_train[:, 1], color='red',alpha=0.5, label='inputs (totem pixels)')

        plt.scatter(uv_cam_labels_test[:, 0], size - uv_cam_labels_test[:, 1], color='goldenrod', alpha=0.5,label='ground truth (test)')
        plt.scatter(uv_cam_predictions_test[:, 0], size - uv_cam_predictions_test[:, 1], color='gold', marker='x', alpha=0.5, label='predictions (test)')
        plt.scatter(uv_tot_inputs_test[:, 0], size - uv_tot_inputs_test[:, 1], color='maroon', alpha=0.5,label='inputs (totem pixels) (test)')

        # plt.xlim(220, 520)
        # plt.ylim(-450+size, -80+size)
        plt.suptitle("(Full dataset_helpers) Predicted vs True Camera Pixels (80-20 split)")
        plt.title(f"Train err on final epoch: {train_err:.2e}. Test err (is full range: {is_man}): {test_err:.2e}. LR: {lr if lr else 'N/A'}, Batchsize: {batch_size if batch_size else 'N/A'}, Epochs: {epochs if epochs else 'N/A'}", fontsize=8.0)
        plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
        saveloc = os.path.join(path_to_exp, 'nn', ext, f'pred_vs_truth_full_{is_man}.png')
        plt.savefig(saveloc, bbox_inches='tight')
        # plt.show()
        plt.close()
    else:
        # if not inference_data:
        #     raise ValueError("inference data required but it is None.")

        # Use inference data as a single batch
        uv_cam_preds = []
        uv_tot_inputs = []
        preds = inference(model, inference_data, embed_fn, in_mins, in_maxs, out_mins, out_maxs)

        uv_cams_inf = preds
        uv_tots_inf = inference_data

        # Append scatter points to the respective lists
        uv_cam_preds = uv_cams_inf.detach().numpy()
        uv_tot_inputs = uv_tots_inf.detach().numpy()

        # uv_cam_preds = np.concatenate(uv_cam_preds[0])
        # uv_tot_inputs = np.concatenate(uv_tot_inputs)

        plt.scatter(uv_cam_preds[:, 0], size - uv_cam_preds[:, 1], color='goldenrod', alpha=0.5,
                    label='preds')
        plt.scatter(uv_tot_inputs[:, 0], size - uv_tot_inputs[:, 1], color='gold', marker='x',
                    alpha=0.5, label='inputs')

        # plt.xlim(220, 520)
        # plt.ylim(-450+size, -80+size)
        plt.suptitle("Inference results")
        # plt.title(
        #     f"Trained on model: {train_err:.2e}. Test err: {test_err:.2e}. LR: {lr if lr else 'N/A'}, Batchsize: {batch_size if batch_size else 'N/A'}, Epochs: {epochs if epochs else 'N/A'}",
        #     fontsize=8.0)
        plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
        saveloc = os.path.join(path_to_exp, 'nn', ext, 'inference.png')
        plt.savefig(saveloc, bbox_inches='tight')
        # plt.show()
        plt.close()
def visualize_predictions_iteratively(model, in_mins, in_maxs, out_mins, out_maxs, embed_fn, size,
                          train_err=None, test_err=None, trainloader=None, testloader=None,
                          inference=False, test=True, path_to_exp='.', ext='.', inference_data=None, inference_saveloc=None, is_embedded=False, is_man=False):
    if not os.path.exists('iterative_plots'):
        os.mkdir('iterative_plots')
    # Create empty lists to store points for scatter plot
    if not test:
        raise NotImplementedError
    if not inference:
        if train_err is None or test_err is None or not trainloader or not testloader:
            raise ValueError("train, test err, train, test loader must all have valid values (not None)")
        uv_cam_predictions_train = []
        uv_cam_labels_train = []
        uv_tot_inputs_train = []

        uv_cam_predictions_test = []
        uv_cam_labels_test = []
        uv_tot_inputs_test = []

        dataloader = trainloader
        for inputs, labels in dataloader:
            embedded_inputs = embed_fn(inputs)
            preds = model(embedded_inputs)
            if is_embedded:
                # take the first n,0:2 elements of the nx42 dimensional embedding, which went through the identity
                preds = preds[:,0:2]
            uv_cams = utils.unnormalize(preds, out_mins, out_maxs).detach().numpy()
            uv_cams_true = utils.unnormalize(labels, out_mins, out_maxs).detach().numpy()
            uv_tots = utils.unnormalize(inputs, in_mins, in_maxs).detach().numpy()

            # Append scatter points to the respective lists
            uv_cam_predictions_train.append(uv_cams)
            uv_cam_labels_train.append(uv_cams_true)
            uv_tot_inputs_train.append(uv_tots)

        # Concatenate all scatter points
        uv_cam_predictions_train = np.concatenate(uv_cam_predictions_train)
        uv_cam_labels_train = np.concatenate(uv_cam_labels_train)
        uv_tot_inputs_train = np.concatenate(uv_tot_inputs_train)

        dataloader = testloader
        for inputs, labels in dataloader:
            embedded_inputs = embed_fn(inputs)
            preds = model(embedded_inputs)
            if is_embedded:
                # take the first n,0:2 elements of the nx42 dimensional embedding, which went through the identity
                preds = preds[:, 0:2]

            uv_cams_test = utils.unnormalize(preds, out_mins, out_maxs).detach().numpy()
            uv_cams_true_test = utils.unnormalize(labels, out_mins, out_maxs).detach().numpy()
            uv_tots_test = utils.unnormalize(inputs, in_mins, in_maxs).detach().numpy()

            # Append scatter points to the respective lists
            uv_cam_predictions_test.append(uv_cams_test)
            uv_cam_labels_test.append(uv_cams_true_test)
            uv_tot_inputs_test.append(uv_tots_test)

        # Concatenate all scatter points
        uv_cam_predictions_test = np.concatenate(uv_cam_predictions_test)
        uv_cam_labels_test = np.concatenate(uv_cam_labels_test)
        uv_tot_inputs_test = np.concatenate(uv_tot_inputs_test)

        # Iteratively plot scatter points with unique labels
        for i in range(1,len(uv_cam_predictions_train)+1):
            plt.scatter(uv_cam_predictions_train[0:i, 0], size - uv_cam_predictions_train[0:i, 1], color='blue', marker='x', alpha=0.5,label='predictions')
            plt.scatter(uv_cam_labels_train[0:i, 0], size - uv_cam_labels_train[0:i, 1], color='green', alpha=0.5,label='ground truth')

            plt.scatter(uv_tot_inputs_train[0:i, 0], size - uv_tot_inputs_train[0:i, 1], color='red',alpha=0.5, label='inputs (totem pixels)')

            # plt.scatter(uv_cam_labels_test[0:i, 0], 256 - uv_cam_labels_test[0:i, 1], color='goldenrod', alpha=0.5,label='ground truth (test)')
            # plt.scatter(uv_cam_predictions_test[0:i, 0], 256 - uv_cam_predictions_test[0:i, 1], color='gold', marker='x', alpha=0.5, label='predictions (test)')
            # plt.scatter(uv_tot_inputs_test[0:i, 0], 256 - uv_tot_inputs_test[0:i, 1], color='maroon', alpha=0.5,label='inputs (totem pixels) (test)')

            plt.xlim(80,180)
            plt.ylim(0,140)

            plt.suptitle("(Full dataset_helpers) Predicted vs True Camera Pixels (80-20 split)")
            plt.title(f"{i} POINTS. Train error on final epoch: {train_err:.2e}. Test error (is_full_range: {is_man}): {test_err:.2e}", fontsize=9.5)
            # plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
            saveloc = os.path.join(path_to_exp, 'nn', ext, 'iterative_plots', f'pred_vs_truth_full_{is_man}_{i}.png') # TODO: Add higher folder name later
            plt.savefig(saveloc, bbox_inches='tight')
            plt.close()
    else:
        raise NotImplementedError

def add_circle(center, radius):

    # Create a circle patch
    circle = Circle(center, radius, edgecolor='r', facecolor='none')

    # Get the current axes and add the circle patch to it
    ax = plt.gca()
    ax.add_patch(circle)