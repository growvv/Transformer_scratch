import torch
import sys
import ipdb
import config
import matplotlib.pyplot as plt
# import seaborn


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def my_predict(model, device, max_lenght):
    indexes = [3, 4, 5, 6, 7]
    sentence_tensor = torch.LongTensor(indexes).unsqueeze(0).to(device)
    outputs = [8]
    for i in range(max_lenght):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)
       
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)
        # print("best_guess: ", best_guess)

        if best_guess == 0:
            break

    return outputs


def draw_atten(attention):
    plt.figure(1)
    atten = attention[0][0]
    print(atten)
    # plt.plot(atten)
    # plt.show()


if __name__ == "__main__":
    x = torch.randint(0,10, (1,8,5,5))
    draw_atten(x)
