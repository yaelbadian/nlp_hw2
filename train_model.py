from torch.nn import LogSoftmax, NLLLoss
import torch.optim as optim
import numpy as np
import torch
from chu_liu_edmonds import decode_mst
import matplotlib.pyplot as plt
import time


def loss_func(scores, target, nllloss):
    # scores.shape: [batch_size, seq_length, seq_length]
    # target.shape: [batch_size, seq_length]
    m = LogSoftmax(dim=1)
    output = nllloss(m(scores), target)
    return output


def train(net, train_loader, test_loader, path='/models/', epochs=10):
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    acumulate_grad_steps = 50
    nllloss = NLLLoss(ignore_index=-1)
    net.train()
    device = net.device
    if net.use_coda:
        net.cuda()
    # Training start
    print("Training Started")
    test_loss_lst, test_acc_lst = [], []
    acc_best = 0
    for epoch in range(epochs):
        t0 = time.time()
        for i, sentence in enumerate(train_loader):
            headers = sentence[2].to(device)
            scores = net(sentence)
            loss = loss_func(scores, headers, nllloss)
            loss = loss / acumulate_grad_steps
            loss.backward()
            if i % acumulate_grad_steps == 0:
                optimizer.step()
                net.zero_grad()
        test_acc, test_loss = predict(net, device, test_loader, nllloss)
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)
        if acc_best < test_acc and epoch > 5:
            tmp_path = path + 'epoch_' + str(epoch) + '_acc_' + str(np.round(test_acc, 4)).replace('.', '') + '.pt'
            net.save(tmp_path)
        print(f"Epoch [{epoch + 1}/{epochs}] Completed \t Test Loss: {test_loss:.3f}"
              f" \t Test Accuracy: {test_acc:.3f} \t Time: {(time.time() - t0) / 60:.2f}")
    plot(test_acc_lst, test_loss_lst, path)


def predict_dep(scores):
    predictions = []
    for sentence_scores in scores:
        score_matrix = sentence_scores.cpu().detach().numpy()
        score_matrix[:, 0] = float("-inf")
        mst, _ = decode_mst(score_matrix, len(score_matrix), has_labels=False)
        predictions.append(mst)
    return np.array(predictions)


def predict(net, device, loader, nllloss):
    net.eval()
    acc, num_of_edges, loss = 0, 0, 0
    for i, sentence in enumerate(loader):
        headers = sentence[2].to(device)
        scores = net(sentence)
        loss += loss_func(scores, headers, nllloss).item()
        predictions = predict_dep(scores)[:, 1:]
        headers = headers.to("cpu").numpy()[:, 1:]
        acc += np.sum(headers == predictions)
        num_of_edges += predictions.size
    net.train()
    return acc / num_of_edges, loss


def plot(accuracies, losses, path):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(list(range(len(losses))), losses, color='turquoise', lw=1.5)
    ax1.tick_params(axis='y', labelcolor='turquoise')
    ax2 = ax1.twinx()
    ax2.set_ylabel('UAS')
    ax2.plot(list(range(len(accuracies))), accuracies, color='orchid', lw=1.5)
    ax2.tick_params(axis='y', labelcolor='orchid')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(path)
    plt.close()
