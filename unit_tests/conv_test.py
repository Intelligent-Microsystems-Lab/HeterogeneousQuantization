# copied from https://github.com/BerenMillidge/PredictiveCodingBackprop
# and modified by Clemens JS Schaefer
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime
from datasets import *
from utils import *
from layers import *


save_dir_fc = "unit_test_conv/"
subprocess.call(["mkdir", "-p", str(save_dir_fc)])


class PCNet(object):
  def __init__(
      self,
      layers,
      n_inference_steps_train,
      inference_learning_rate,
      loss_fn,
      loss_fn_deriv,
      device="cpu",
      numerical_check=False,
  ):
    self.layers = layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.device = device
    self.loss_fn = loss_fn
    self.loss_fn_deriv = loss_fn_deriv
    self.L = len(self.layers)
    self.outs = [[] for i in range(self.L + 1)]
    self.prediction_errors = [[] for i in range(self.L + 1)]
    self.predictions = [[] for i in range(self.L + 1)]
    self.mus = [[] for i in range(self.L + 1)]
    self.numerical_check = numerical_check
    if self.numerical_check:
      print("Numerical Check Activated!")
      for l in self.layers:
        l.set_weight_parameters()

  def update_weights(self, print_weight_grads=False, get_errors=False):
    weight_diffs = []
    for (i, l) in enumerate(self.layers):
      if True:  # i !=1:
        if self.numerical_check:
          true_weight_grad = l.get_true_weight_grad().clone()
        dW = l.update_weights(
            self.prediction_errors[i + 1], update_weights=True
        )
        true_dW = l.update_weights(
            self.predictions[i + 1], update_weights=True
        )
        np.save(save_dir_fc + "dw" + str(i) + "_train.npy", true_dW)
        diff = torch.sum((dW - true_dW) ** 2).item()
        weight_diffs.append(diff)
        if print_weight_grads:
          print("weight grads : ", i)
          print("dW: ", dW * 2)
          print("true diffs: ", true_dW * 2)
          if self.numerical_check:
            print("true weights ", true_weight_grad)
    return weight_diffs

  def forward(self, x):
    for i, l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self, x):
    with torch.no_grad():
      for i, l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def infer(self, inp, label, n_inference_steps=None):
    self.n_inference_steps_train = (
        n_inference_steps
        if n_inference_steps is not None
        else self.n_inference_steps_train
    )
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      for i, l in enumerate(self.layers):
        # initialize mus with forward predictions
        self.mus[i + 1] = l.forward(self.mus[i])
        self.outs[i + 1] = self.mus[i + 1].clone()
      np.save(save_dir_fc + "out_train.npy", self.mus[-1])
      self.mus[-1] = label.clone()  # setup final label
      self.prediction_errors[-1] = -self.loss_fn_deriv(
          self.outs[-1], self.mus[-1]
      )  # self.mus[-1] - self.outs[-1] #setup final prediction errors
      self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
        # reversed inference
        for j in reversed(range(len(self.layers))):
          if j != 0:
            self.prediction_errors[j] = self.mus[j] - self.outs[j]
            self.predictions[j] = self.layers[j].backward(
                self.prediction_errors[j + 1]
            )
            dx_l = self.prediction_errors[j] - self.predictions[j]
            self.mus[j] -= self.inference_learning_rate * (
                2 * dx_l
            )
      np.save(save_dir_fc + "pred0_train.npy", self.predictions[1])
      np.save(save_dir_fc + "pred1_train.npy", self.predictions[2])
      np.save(save_dir_fc + "mus0_train.npy", self.mus[0])
      np.save(save_dir_fc + "mus1_train.npy", self.mus[1])
      np.save(save_dir_fc + "mus2_train.npy", self.mus[2])
      # update weights
      weight_diffs = self.update_weights()
      # get loss:
      L = self.loss_fn(
          self.outs[-1], self.mus[-1]
      ).item()  # torch.sum(self.prediction_errors[-1]**2).item()
      # get accuracy
      acc = 0.0  # accuracy(self.no_grad_forward(inp), label)
      return L, acc, weight_diffs

  def test_accuracy(self, testset):
    accs = []
    for i, (inp, label) in enumerate(testset):
      pred_y = self.no_grad_forward(inp.to(DEVICE))
      acc = 0.0  # accuracy(pred_y, onehot(label).to(DEVICE))
      accs.append(acc)
    return np.mean(np.array(accs)), accs

  def train(
      self,
      dataset,
      testset,
      n_epochs,
      n_inference_steps,
      logdir,
      savedir,
      old_savedir,
      save_every=1,
      print_every=10,
  ):
    if old_savedir != "None":
      self.load_model(old_savedir)
    losses = []
    accs = []
    weight_diffs_list = []
    test_accs = []
    for epoch in range(n_epochs):
      losslist = []
      print("Epoch: ", epoch)
      for i, (inp, label) in enumerate(dataset):
        # if self.loss_fn != cross_entropy_loss:
        #  label = onehot(label).to(DEVICE)
        # else:
        label = label.long().to(DEVICE)
        L, acc, weight_diffs = self.infer(inp.to(DEVICE), label)
        losslist.append(L)
      mean_acc, acclist = self.test_accuracy(dataset)
      accs.append(mean_acc)
      mean_loss = np.mean(np.array(losslist))
      losses.append(mean_loss)
      mean_test_acc, _ = self.test_accuracy(testset)
      test_accs.append(mean_test_acc)
      weight_diffs_list.append(weight_diffs)
      print("TEST ACCURACY: ", mean_test_acc)
      print("SAVING MODEL")
      # self.save_model(
      #    logdir, savedir, losses, accs, weight_diffs_list, test_accs
      # )

  def save_model(
      self, savedir, logdir, losses, accs, weight_diffs_list, test_accs
  ):
    for i, l in enumerate(self.layers):
      l.save_layer(logdir, i)
    np.save(save_dir_fc + "losses.npy", np.array(losses))
    np.save(save_dir_fc + "accs.npy", np.array(accs))
    np.save(save_dir_fc + "weight_diffs.npy", np.array(weight_diffs_list))
    np.save(save_dir_fc + "test_accs.npy", np.array(test_accs))
    # subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print(
        "Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir)
    )
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(["echo", "saved at time: " + str(current_time)])

  def load_model(self, old_savedir):
    for (i, l) in enumerate(self.layers):
      l.load_layer(old_savedir, i)


if __name__ == "__main__":
  global DEVICE
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  parser = argparse.ArgumentParser()
  print("Initialized")
  # parsing arguments
  parser.add_argument("--logdir", type=str, default="logs")
  parser.add_argument("--savedir", type=str, default="savedir")
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--learning_rate", type=float, default=0.0005)
  parser.add_argument("--N_epochs", type=int, default=1)
  parser.add_argument("--save_every", type=int, default=1)
  parser.add_argument("--print_every", type=int, default=10)
  parser.add_argument("--old_savedir", type=str, default="None")
  parser.add_argument("--n_inference_steps", type=int, default=100)
  parser.add_argument("--inference_learning_rate", type=float, default=0.1)
  parser.add_argument("--network_type", type=str, default="pc")
  parser.add_argument("--dataset", type=str, default="cifar")
  parser.add_argument("--loss_fn", type=str, default="mse")

  args = parser.parse_args()
  print("Args parsed")
  # create folders
  # if args.savedir != "":
  #   subprocess.call(["mkdir", "-p", str(args.savedir)])
  # if args.logdir != "":
  #   subprocess.call(["mkdir", "-p", str(args.logdir)])
  print("folders created")

  dataset_x = torch.randn((256, 3, 32, 32))
  dataset_y = torch.randn((256, 12, 24, 24))
  dataset = [[dataset_x[:128, :, :, :], dataset_y[:128, :, :, :]]]
  testset = [[dataset_x[128:, :, :, :], dataset_y[128:, :, :, :]]]

  np.save(save_dir_fc + "/dataset_x.npy", np.array(dataset_x))
  np.save(save_dir_fc + "/dataset_y.npy", np.array(dataset_y))
  loss_fn, loss_fn_deriv = parse_loss_function(args.loss_fn)

  if args.dataset in ["cifar", "mnist", "svhn"]:
    output_size = 10
  if args.dataset == "cifar100":
    output_size = 100

  def onehot(x):
    z = torch.zeros([len(x), output_size])
    for i in range(len(x)):
      z[i, x[i]] = 1
    return z.float().to(DEVICE)

  l5 = ConvLayer(
      32, 3, 6, 128, 5, args.learning_rate, relu, relu_deriv, device=DEVICE
  )
  if args.loss_fn == "crossentropy":
    l6 = ConvLayer(
        28,
        6,
        12,
        128,
        5,
        args.learning_rate,
        relu,
        relu_deriv,
        device=DEVICE,
    )
  else:
    l6 = ConvLayer(
        28,
        6,
        12,
        128,
        5,
        args.learning_rate,
        relu,
        relu_deriv,
        device=DEVICE,
    )
  layers = [l5, l6]

  if args.network_type == "pc":
    net = PCNet(
        layers,
        args.n_inference_steps,
        args.inference_learning_rate,
        loss_fn=loss_fn,
        loss_fn_deriv=loss_fn_deriv,
        device=DEVICE,
    )
  elif args.network_type == "backprop":
    net = Backprop_CNN(
        layers, loss_fn=loss_fn, loss_fn_deriv=loss_fn_deriv
    )
  else:
    raise Exception(
        "Network type not recognised: must be one of 'backprop', 'pc'"
    )
  net.save_model("logs", save_dir_fc, [], [], [], [])
  net.train(
      dataset,
      testset,
      args.N_epochs,
      args.n_inference_steps,
      save_dir_fc,
      save_dir_fc,
      args.old_savedir,
      args.save_every,
      args.print_every,
  )
  # net.save_model("logs", 'unit_test_fc/', [], [], [], [])
