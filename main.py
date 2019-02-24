from parameters import *
from model import *
from train import *

if __name__ == '__main__':
    net = CaptchaNet()
    # print(net.named_parameters())
    train(net)
