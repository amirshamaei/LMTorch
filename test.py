import unittest
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from LMtorch import LMtorch


class MyTestCase(unittest.TestCase):
    def test_lm_withoutBounds(self):
        def model(x, y=None):
            return ((x - 4) ** 2)

        x = torch.randn(1)
        x_pre = (LMtorch(device='cpu').solve(f=model, x0=x,
                                             max_iter=200, delta=1))

        print(x_pre.cpu().numpy())
        assert torch.abs(torch.sqrt(torch.FloatTensor([4]))) - torch.abs(x_pre.cpu()) < 0.1

    def test_lm(self):
        def model(x, y=None):
            return ((x - 4) ** 2)

        x = torch.randn(1)
        x_pre = (LMtorch(device='cpu').solve(f=model, x0=x, bounds=[torch.FloatTensor([-10]), torch.FloatTensor([6])], max_iter=200, delta=1))

        print(x_pre.cpu().numpy())
        assert torch.abs(torch.sqrt(torch.FloatTensor([4]))) - torch.abs(x_pre.cpu()) < 0.1

    def testmodelless(self):
        def fun(coeff):
            t = torch.linspace(0, 100, 1000)
            return torch.exp(- 10 * coeff * t)

        def model(x, y=None):
            return torch.mean((x - y) ** 2)

        coeff = 0.01
        y = fun(coeff)
        x = torch.randn((1, 1000))
        st = time.time()
        x_new = LMtorch(device='cpu').solve(f=model, x0=x, y=y, bounds=[torch.FloatTensor([-10]), torch.FloatTensor([0.5])], max_iter=1000, delta=1)
        print("predicted value is: {}".format(x_new.cpu().numpy()))

        plt.plot(y)
        plt.plot(x_new.numpy().T)
        plt.show()
        assert torch.norm(x_new - y) < 10

    def testmodelless_variedBounds(self):
        def fun(coeff):
            t = torch.linspace(0, 100, 1000)
            return torch.exp(- 10 * coeff * t)

        def model(x, y=None):
            return torch.mean((x - y) ** 2)

        coeff = 0.01
        y = fun(coeff)
        x = torch.randn((1, 1000))
        st = time.time()
        min_ = torch.arange(start=0,step=-1/1000,end=-1)
        max_ = torch.arange(start=0,step=1/1000,end=1)
        x_new = LMtorch(device='cpu').solve(f=model, x0=x, y=y, bounds=[min_, max_], max_iter=1000, delta=1)
        print("spend time is (cpu): {}".format(time.time()-st))

        plt.plot(y)
        plt.plot(x_new.numpy().T)
        plt.show()
        assert torch.norm(x_new - y) < 10
    def testmodelless_batch(self):
        def fun(coeff):
            t = torch.linspace(0, 100, 1000)
            return torch.exp(- 10 * coeff * t)

        def model(x, y=None):
            return torch.mean((x - y) ** 2)

        coeff = torch.FloatTensor([0.01,0.2]).unsqueeze(1)
        y = fun(coeff)
        x = torch.randn((2,1000))
        st = time.time()
        x_new = LMtorch(device='cpu').solve(f=model, x0=x, y=y, bounds=[torch.FloatTensor([-5]), torch.FloatTensor([0.5])], max_iter=100, delta=1)
        print("spend time is (cpu + batched): {}".format(time.time()-st))

        plt.plot(y[0])
        plt.plot(x_new[0].numpy().T)
        plt.show()
        plt.plot(y[1])
        plt.plot(x_new[1].numpy().T)
        plt.show()
        assert torch.norm(x_new - y) < 10

    def test_lm_cuda(self):
        def model(x, y=None):
            return ((x - 4) ** 2)

        x = torch.randn(1)
        x_pre = (LMtorch(device='cuda').solve(f=model, x0=x, bounds=[torch.FloatTensor([-10]), torch.FloatTensor([5])], max_iter=100, delta=1))
        print("predicted value is: {}".format(x_pre.cpu().numpy()))
        assert torch.abs(torch.sqrt(torch.FloatTensor([4]))) - torch.abs(x_pre.cpu()) < 0.1

    def testmodelless_cuda(self):
        def fun(coeff):
            t = torch.linspace(0, 100, 1000)
            return torch.exp(- 10 * coeff * t)

        def model(x, y=None):
            return torch.mean((x - y) ** 2)

        coeff = 0.01
        y = fun(coeff)
        x = torch.randn((1, 1000))
        st = time.time()
        x_new = LMtorch(device='cuda').solve(f=model, x0=x, y=y, bounds=[torch.FloatTensor([-0.5]), torch.FloatTensor([0.5])], max_iter=100, delta=1)
        print("spend time is (cuda): {}".format(time.time() - st))
        plt.plot(y)
        plt.plot(x_new.cpu().numpy().T)
        plt.show()
        assert torch.norm(x_new - y) < 10

    def testmodelless_batch_cuda(self):
        def fun(coeff):
            t = torch.linspace(0, 100, 1000)
            return torch.exp(- 10 * coeff * t)

        def model(x, y=None):
            return torch.mean((x - y) ** 2)

        coeff = 0.01 * torch.randint(10, (10, 1))
        y = fun(coeff)
        x = torch.randn((10, 1000))
        st = time.time()
        x_new = LMtorch(device='cuda').solve(f=model, x0=x, y=y, bounds=[torch.FloatTensor([-0.5]), torch.FloatTensor([0.5])], max_iter=400, delta=1e-1)
        print("spend time is (cuda + batched): {}".format(time.time() - st))
        plt.plot(y[0])
        plt.plot(x_new[0].numpy().T)
        plt.show()
        plt.plot(y[1])
        plt.plot(x_new[1].numpy().T)
        plt.show()

if __name__ == '__main__':
    unittest.main()
