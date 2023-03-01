import unittest
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from main import LMtorch


class MyTestCase(unittest.TestCase):
    def test_lm(self):
        def model(x,y=None):
            return (x ** 2 - 4)

        x = torch.randn(1)
        x_pre = (LMtorch(device='cpu').solve(f = model, x0=x, bounds=[-1, 1], max_iter=10, delta=0.1))
        assert torch.abs(torch.sqrt(torch.FloatTensor([4])))-torch.abs(x_pre) < 0.1

    def testmodelless(self):
        def fun(coeff):
          t = torch.linspace(0,100,1000)
          return torch.exp(- 10 * coeff*t)

        def model(x,y=None):
          return torch.mean((x-y)**2)

        coeff = 0.01
        y = fun(coeff)
        x = torch.randn((1,1000))
        st = time.time()
        x_new = LMtorch(device='cpu').solve(f=model,x0=x,y=y,bounds=[-10,0.5],max_iter=100,delta=1e-1)
        print(time.time()-st)

        plt.plot(y)
        plt.plot(x_new.numpy().T)
        plt.show()
        assert torch.norm(x_new-y) < 10
        # def test_lm_cuda(self):



    #     def model(x):
    #         return (x ** 2 - 4)
    #
    #     x = torch.randn(1)
    #     st = time.time()
    #     x_pre = (LMtorch(device='cuda').solve(model, x, [-15, 15], max_iter=1000, delta=0.1))
    #     print(x_pre)
    #     print(time.time() - st)
    #     assert torch.abs(torch.sqrt(torch.FloatTensor([4]))) - torch.abs(x_pre) < 0.0001  # st = time.time()


        # add assertion here


if __name__ == '__main__':
    unittest.main()
