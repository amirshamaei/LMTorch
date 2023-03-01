# A comment.
#
import torch
from matplotlib import pyplot as plt


class LMtorch():
    def __init__(self,device='cpu'):
        self.device = device

    def solve(self,f=None, x0=None, y=torch.FloatTensor([0]), bounds=None, max_iter=100, tol=1e-6, lambda0=1e-3, delta=1e-6):
        """
        Minimize a function `f` using the modified Levenberg-Marquardt algorithm.

        Parameters
        ----------
        f : callable
            The function to minimize. Should take a tensor `x` as input and return a scalar tensor.
        J : callable
            The Jacobian of the function `f`. Should take a tensor `x` as input and return a tensor with shape (m, n),
            where `m` is the number of outputs of `f` and `n` is the number of inputs.
        x0 : torch.Tensor
            The initial guess for the minimum.
        max_iter : int, optional
            The maximum number of iterations. Default is 100.
        tol : float, optional
            The tolerance for the change in the loss. Default is 1e-6.
        lambda0 : float, optional
            The initial value of the damping parameter. Default is 1.0.
        delta : float, optional
            The step size used to compute the approximate Hessian. Default is 1e-6.

        Returns
        -------
        x_min : torch.Tensor
            The estimated minimum.
        """

        x = x0.clone().detach().requires_grad_(True).to(self.device)

        J = torch.autograd.functional.jacobian

        # initialization
        f_val = f(x, y).reshape(-1, 1)
        J_val = J(f, (x,y))[0]
        JtJ = torch.matmul(J_val.T, J_val)
        Jtf = torch.matmul(J_val.T, f_val)
        lambda_ = lambda0 * torch.max(torch.diag(Jtf))
        multiplier = 1.5
        x_list = []
        max_lambda = torch.FloatTensor([1e10])

        for i in range(max_iter):
            # diag_JtJ = torch.diag(JtJ)
            # torch.diag(diag_JtJ)
            h = (JtJ + lambda_ * torch.eye(JtJ.shape[0])).inverse().matmul(Jtf)
            dparam = -(delta * h)
            x_new = torch.clamp(x + dparam.T, bounds[0], bounds[1])
            # x_new = x + dparam.T
            f_new_val = f(x_new,y).reshape(-1, 1)
            rho_denom = torch.matmul(h.T, lambda_ * h - Jtf)
            rho_nom = torch.matmul(f_val, f_val.T) - torch.matmul(f_new_val, f_new_val.T)
            rho = rho_nom / rho_denom if rho_denom > 0 else 10e15 if rho_nom > 0 else -10e15
            # rho = (torch.norm(f_val) - torch.norm(f_new_val) ) / torch.matmul(h.T, lambda_ * h - Jtf)

            if rho > 0:
                x = x_new
                J_val = J(f, (x,y))[0]
                JtJ = torch.matmul(J_val.T, J_val)
                Jtf = torch.matmul(J_val.T, f_val)
                lambda_, v = (lambda_ * torch.max(torch.tensor([1 / 3, 1 - (2 * rho - 1) ** 3])), 2)
                if torch.norm(f_val - f_new_val) < tol:
                    break
            else:
                lambda_= (torch.min(lambda_ * multiplier,max_lambda))
                multiplier *= 2

            x_list.append(x.clone())
            f_val = f(x, y).reshape(-1, 1)
            if i == max_iter - 1:
                print("Warning: Maximum number of iterations reached.")

        return x.detach()


