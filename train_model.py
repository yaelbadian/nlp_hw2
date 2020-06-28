from torch.nn import LogSoftmax, NLLLoss, KLDivLoss, functional
import torch.optim as optim
import torch
import numpy as np
from functools import partial
from chu_liu_edmonds import decode_mst
import matplotlib.pyplot as plt
import time


def nll_loss_func(scores, target, nllloss):
    # scores.shape: [batch_size, seq_length, seq_length]
    # target.shape: [batch_size, seq_length]
    m = LogSoftmax(dim=1)
    output = nllloss(m(scores), target)
    return output


def kl_loss_func(scores, target, klloss):
    # scores.shape: [batch_size, seq_length, seq_length]
    # target.shape: [batch_size, seq_length]
    m = LogSoftmax(dim=1)
    output = klloss(torch.transpose(m(scores[0, :, 1:]) + 0.025, 0, 1),
                    functional.one_hot(target[0, 1:], num_classes=scores.shape[-1]) + 0.025)
    return output


def train(net, train_loader, test_loader, path='models/', epochs=10, plot_train=False):
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    accumulate_grad_steps = 50
    nllloss = NLLLoss(ignore_index=-1)
    loss_func = partial(nll_loss_func, nllloss=nllloss)
    # klloss = KLDivLoss(reduction='batchmean')
    # loss_func = partial(kl_loss_func, klloss=klloss)
    net.train()
    device = net.device
    if net.use_coda:
        net.cuda()
    print("Training Started")
    test_loss_lst, test_acc_lst, train_loss_lst, train_acc_lst, time_lst = [], [], [], [], []
    best_acc = 0
    for epoch in range(epochs):
        t0 = time.time()
        weights = 0
        for i, sentence in enumerate(train_loader):
            headers = sentence[2].to(device)
            sentence_len = sentence[3][0]
            scores = net(sentence)
            loss = loss_func(scores, headers)
            loss = loss * sentence_len
            weights += sentence_len
            if i % accumulate_grad_steps == 0:
                loss = loss / weights
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                net.zero_grad()
                weights = 0
            else:
                loss.backward()
        test_acc, test_loss = predict(net, device, test_loader, loss_func)
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)
        if plot_train:
            train_acc, train_loss = predict(net, device, train_loader, loss_func)
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_acc)
        if best_acc < test_acc and epoch > 5 and test_acc > 0.88:
            tmp_path = path + '_epoch_' + str(epoch) + '_acc_' + str(np.round(test_acc, 4)).replace('.', '') + '.pt'
            net.save(tmp_path)
            best_acc = test_acc
        ctime = (time.time() - t0) / 60
        time_lst.append(ctime)
        print(f"Epoch [{epoch + 1}/{epochs}] Completed \t Test Loss: {test_loss:.3f}"
              f" \t Test Accuracy: {test_acc:.3f} \t Time: {ctime:.2f}")
    plot(test_acc_lst, test_loss_lst, time_lst, path + '_test_plot.png')
    if plot_train:
        plot(train_acc_lst, train_loss_lst, time_lst, path + '_train_plot.png')


def predict_dep(scores):
    predictions = []
    for sentence_scores in scores:
        score_matrix = sentence_scores.cpu().detach().numpy()
        score_matrix[:, 0] = float("-inf")
        mst, _ = decode_mst(score_matrix, len(score_matrix), has_labels=False)
        predictions.append(mst)
    return np.array(predictions)


def predict(net, device, loader, loss_func):
    net.eval()
    acc, num_of_edges, loss = 0, 0, 0
    for i, sentence in enumerate(loader):
        headers = sentence[2].to(device)
        scores = net(sentence)
        loss += loss_func(scores, headers).item()
        predictions = predict_dep(scores)[:, 1:]
        headers = headers.to("cpu").numpy()[:, 1:]
        acc += np.sum(headers == predictions)
        num_of_edges += predictions.size
    net.train()
    return acc / num_of_edges, loss


def plot(accuracies, losses, times, path):
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(list(range(1, 1 + len(losses))), losses, color='turquoise', lw=1.5)
    ax1.tick_params(axis='y', labelcolor='turquoise')
    ax2 = ax1.twinx()
    ax2.set_ylabel('UAS')
    ax2.plot(list(range(len(accuracies))), accuracies, color='orchid', lw=1.5)
    ax2.tick_params(axis='y', labelcolor='orchid')
    fig.tight_layout()
    fig.suptitle(f'Total time: {sum(times):.2f}   Avg epoch time: {sum(times)/len(times):.2f}')
    fig.savefig(path)
    plt.close()
