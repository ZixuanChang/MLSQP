import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from qpsolvers import Problem, solve_problem
def griewank_function(x):
    sum_term = np.sum(x ** 2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    y = sum_term - prod_term + 1
    return y
def six_hump_camel(x):
    r=1
    r1=0
    if isinstance(x,torch.Tensor):
        for i in range(5):
            r = r * torch.cos(x[i] / (i+1) ** 0.5)
            r1 = r1 + x[i] ** 2
        r = r1 / 4000 - r + 1
    else:
        r = np.apply_along_axis(griewank_function, 1, x)
    return r
class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()
    def forward(self, x):
        x = x * torch.sigmoid(x)
        # x=torch.tanh(x)
        return x
swish = Act_op()
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 第一个隐藏层
        self.fc1 = nn.Linear(input_size, 256)
        self.hidden_layers_1 = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])

        # 第二个隐藏层
        self.fc2 = nn.Linear(256, 128)
        self.hidden_layers_2 = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])

        # 输出层
        self.fc_out = nn.Linear(128, output_size)



    def forward(self, x):
        # 应用第一个隐藏层和激活函数
        x = swish(self.fc1(x))

        # 应用第一个隐藏层组中的所有层，带 dropout 和激活函数
        for layer in self.hidden_layers_1:
            x = swish(layer(x))

        x = swish(self.fc2(x))
        # 应用第二个隐藏层组中的所有层，带 dropout 和激活函数
        for layer in self.hidden_layers_2:
            x = swish(layer(x))

        # 应用输出层
        output = self.fc_out(x)

        return output

class MLsqp(object):
    def Objective(self,x,y):
        # self.obj=obj
        # self.jac_obj=jac_obj#向量
        # self.hes_obj=hes_obj#矩阵
        print('Warning:继承类后重写方法定义目标函数')
        return x[0]+2*x[1]+y[0]*y[0]

    def Penalty(self,lamda,mu_h,mu_b,xk,yk,gxk,d,lb,ub,flag):
        self.rho_k = np.array([max((self.rho_k0[i] + abs(lamda[i])) / 2, abs(lamda[i])) for i in range(len(lamda))])
        self.nu_k_h = np.array([max((self.nu_k0_h[i] + abs(mu_h[i])) / 2, abs(mu_h[i])) for i in range(len(mu_h))])
        if flag==1:
            self.rho_k0=self.rho_k.copy()
            self.nu_k0_h=self.nu_k_h.copy()
        x=xk.detach().numpy()
        y=yk
        self.P=self.obj
        for i in range(len(self.rho_k)):
            self.P=self.P+self.rho_k[i]*abs(np.array(self.A[i])@x-self.b[i])

        for i in range(self.len_G_c+self.len_G_bx):
            self.P = self.P + self.nu_k_h[i] * max(0,np.array(self.G[i]) @ x-self.h[i])
        for i in range(len(self.G_order_y[0])):
            if self.G_order_y[0][i]=='ub':
                self.P=self.P+self.nu_k_h[i+self.len_G_c+self.len_G_bx]*max(0,y[self.G_order_y[1][i]-self.len_x]-self.ub[self.G_order_y[1][i]])
            else:
                self.P=self.P+self.nu_k_h[i+self.len_G_c+self.len_G_bx]*max(0,-y[self.G_order_y[1][i]-self.len_x]+self.lb[self.G_order_y[1][i]])

        self.DP=gxk@d-self.P+self.obj
        return {'P': self.P.detach().numpy(), 'DP':self.DP.detach().numpy()}

    def setMLmodel(self,mean=None,std=None,model_path=None):
        self.model = torch.load(model_path)
        self.model.eval()
        self.std=std
        self.mean = mean


    def setConstraints(self,G=None,h=None,A=None,b=None,lb=None,ub=None):
        self.lb=lb
        self.ub=ub
        self.len_G_c=len(h)
        self.G_order_x=[[],[]]
        for i in range(self.len_x):
            if self.ub[i]!=np.inf :
                G=np.vstack((G,np.array([int(x==i) for x in range(self.len_x)])))
                h=np.append(h,self.ub[i])
                self.G_order_x[0].append('ub')
                self.G_order_x[1].append(i)
            if self.lb[i]!=-np.inf:
                G = np.vstack((G, [-int(x==i) for x in range(self.len_x)]))
                h=np.append(h,-self.lb[i])
                self.G_order_x[0].append('lb')
                self.G_order_x[1].append(i)
        self.len_G_bx=len(h)-self.len_G_c
        self.G=G
        self.h=h
        self.A=A
        self.b=b


    def MLHessian(self,Xk):
        x=torch.tensor(Xk.detach().numpy(),dtype=torch.float32,requires_grad=True)
        input=(x-torch.tensor(self.mean[0],dtype=torch.float32))/torch.tensor(self.std[0],dtype=torch.float32)
        output = self.model(input)
        y=output*torch.tensor(self.std[1],dtype=torch.float32)+torch.tensor(self.mean[1],dtype=torch.float32)
        y.requires_grad_(True)
        self.obj=self.Objective(x,y)
        self.L=self.Lagrange(x,y)

        g=grad(self.obj, x, create_graph=True)[0]
        gL=grad(self.L, x, create_graph=True)[0]
        HL=torch.stack([grad(gL, x, grad_outputs=arr, retain_graph=True)[0] for arr in torch.eye(self.len_x)])
        HL=0.5*(HL+HL.T)
        H=torch.stack([grad(g, x, grad_outputs=arr, retain_graph=True)[0] for arr in torch.eye(self.len_x)])
        H=0.5*(H+H.T)

        if self.is_psd(HL):
            pass
        else:
            HL=self.F_norm_modification()   #

        return {'g':g.detach().numpy(),'H':HL.detach().numpy(),'prediction':y.detach().numpy(),'objective':self.obj.detach().numpy()}
    def Lagrange(self,x,y):
        L=self.Objective(x,y)                            #haven't adding +lambda.T@(Ax-b)-mu.T@(Gx-h)

        for i in range(len(self.G_order_x[0])):
            if self.G_order_x[0][i]=='ub':
                L=L-self.mu_h[i+self.len_G_c]*(x[self.G_order_x[1][i]]-self.ub[self.G_order_x[1][i]])
            else:
                L=L+self.mu_h[i+self.len_G_c]*(x[self.G_order_x[1][i]]-self.lb[self.G_order_x[1][i]])

        for i in range(len(self.G_order_y[0])):
            if self.G_order_y[0][i]=='ub':
                L=L-self.mu_h[i+self.len_G_c+self.len_G_bx]*(y[self.G_order_y[1][i]-self.len_x]-self.ub[self.G_order_y[1][i]])
            else:
                L=L+self.mu_h[i+self.len_G_c+self.len_G_bx]*(y[self.G_order_y[1][i]-self.len_x]-self.lb[self.G_order_y[1][i]])

        L=L+torch.tensor(self.lamda,dtype=torch.float32)@(torch.tensor(self.A,dtype=torch.float32)@x-torch.tensor(self.b,dtype=torch.float32))-torch.tensor(self.mu_h[:self.len_G_c],dtype=torch.float32)@(torch.tensor(self.G[:self.len_G_c],dtype=torch.float32)@x-torch.tensor(self.h[:self.len_G_c],dtype=torch.float32))

        return L
    def qp(self,H,g,xk):
        xk=xk.detach().numpy()
        P = np.array(H)
        q = np.array(g)
        G = self.G
        h = self.h-self.G@xk
        A = self.A
        b = self.b-self.A@xk

        x=torch.tensor(xk[0:self.len_x],dtype=torch.float32,requires_grad=True)
        input=(x-torch.tensor(self.mean[0],dtype=torch.float32))/torch.tensor(self.std[0],dtype=torch.float32)
        output = self.model(input)
        y=output*torch.tensor(self.std[1],dtype=torch.float32)+torch.tensor(self.mean[1],dtype=torch.float32)
        dG=torch.stack([grad(y, x, grad_outputs=arr, retain_graph=True)[0] for arr in torch.eye(len(y))])

        for i in range(self.len_x,len(self.ub)):
            if self.ub[i]!=np.inf:
                G=np.vstack((G,dG[i-self.len_x]))
                h=np.append(h,self.ub[i]-y.detach().numpy()[i-self.len_x])
            if self.lb[i] != -np.inf:
                G=np.vstack((G,-dG[i-self.len_x]))
                h = np.append(h, y.detach().numpy()[i-self.len_x]-self.lb[i])
        self.G_expand=G
        self.h_expand=h

        problem = Problem(P, q, G=G,h=h,A=A,b=b)
        try:
            solution = solve_problem(problem, solver="proxqp",max_iter=100)
        except ValueError:
            problem = Problem(np.eye(len(P)), q, G=G, h=h, A=A, b=b)
            solution = solve_problem(problem, solver="proxqp", max_iter=100)
        return solution

    def setInitialPoint(self,x):
        self.IP=x
        self.len_x=len(self.IP)

    def Acc_Inf(self,x,y):
        acc=0
        x=x.detach().numpy()
        for i in range(len(self.G_order_x[0])):
            if self.G_order_x[0][i]=='ub':
                acc=acc+max(0,(x[self.G_order_x[1][i]]-self.ub[self.G_order_x[1][i]]))
            else:
                acc=acc+max(0,-x[self.G_order_x[1][i]]+self.lb[self.G_order_x[1][i]])
        for i in range(len(self.G_order_y[0])):
            if self.G_order_y[0][i]=='ub':
                acc=acc+max(0,y[self.G_order_y[1][i]-self.len_x]-self.ub[self.G_order_y[1][i]])
            else:
                acc=acc+max(0,-y[self.G_order_y[1][i]-self.len_x]+self.lb[self.G_order_y[1][i]])

        acc = acc + sum(abs(self.A @ x - self.b)) + sum([max(0,t) for t in self.G[:self.len_G_c] @ x - self.h[:self.len_G_c]])
        return acc

    def is_psd(self,mat):
        self.L,self.V=torch.linalg.eig(mat)
        self.L=self.L.real
        self.V=self.V.real
        return bool((mat == mat.T).all() and (self.L >= 0).all())

    def F_norm_modification(self):
        self.D=torch.tensor(np.array([0 for _ in self.L]), dtype=torch.float32)
        for i in range(len(self.L)):
            self.D[i]=torch.max(torch.tensor(0),1e-8-self.L[i])
        return self.V @ torch.diag_embed(self.D) @ torch.linalg.inv(self.V)
    def sqp_solve(self):
        maxk = 5000
        k = 0
        tol = 5e-5
        x0=self.IP
        self.len_G_expand=len(self.h)
        self.len_y_order=0

        self.G_order_y=[[],[]]
        for i in range(self.len_x,len(self.ub)):
            if self.ub[i]!=np.inf :
                self.G_order_y[0].append('ub')
                self.G_order_y[1].append(i)
                self.len_G_expand = self.len_G_expand + 1
            if self.lb[i] != -np.inf:
                self.G_order_y[0].append('lb')
                self.G_order_y[1].append(i)
                self.len_G_expand = self.len_G_expand + 1
        self.len_G_expand=self.len_G_expand+self.len_y_order
        self.rho_k0=[0 for i in range(len(self.b))]         #equation
        self.nu_k0_h = [0 for i in range(self.len_G_expand)] #inequation
        self.mu_h = [0 for i in range(self.len_G_expand)]
        self.lamda = [0 for i in range(len(self.A))]

        global t0,t1
        t0=time.time()
        while k < maxk:                                    #回溯Armijo condition
            if k==0:
                obj=1e20
                P0=obj
                x_temp=self.IP
                reH_flag=0
            else:
                x0=x
            diff=self.MLHessian(Xk=x0)
            qpsol = self.qp(H=diff['H'], g=diff['g'], xk=x0)
            self.lamda=np.array(qpsol.y)
            self.mu_h=np.array(qpsol.z)

            diffP=self.Penalty(lamda=qpsol.y,mu_h=qpsol.z,mu_b=qpsol.z_box,xk=x0,yk=diff['prediction'],gxk=diff['g'],d=qpsol.x,lb=self.lb,ub=self.ub,flag=1)
            dk=qpsol.x[0:self.len_x]

            acc_inf=self.Acc_Inf(x0,diff['prediction'])
            acc_step=abs(dk@dk)
            acc_opt=abs(obj-diff['objective'])


            if (acc_inf <= tol and acc_opt <= tol and k != 0) or (acc_inf <= tol and acc_step <= tol and k != 0):
                break

            rho = 0.1
            p = 0.618
            alpha=1
            while True:
                diff_alpha = self.MLHessian(Xk=x0 + alpha * dk)
                diffP_alpha = self.Penalty(lamda=qpsol.y, mu_h=qpsol.z, mu_b=qpsol.z_box, xk=x0 + alpha * dk,
                                           yk=diff_alpha['prediction'], gxk=diff['g'], d=qpsol.x, lb=self.lb,
                                           ub=self.ub,flag=0)
                if ((diffP_alpha['P'] - diffP['P']) < (rho * alpha * diffP['DP'])):#ill
                    break
                elif alpha<1e-2:
                    break
                else:
                    alpha = p*alpha


            t1= time.time()

            x = x0 + alpha * dk
            obj = diff['objective']
            k += 1
            y_pred = self.model(x.to(torch.float))
            print('***********************************************')
            print('iteration: ',k)
            print('solution:  ',[round(item,2) for item in x.tolist()])
            print('prediction:',[round(item,2) for item in y_pred.tolist()])
            print('objective: ',round(self.MLHessian(Xk=x)['objective'][0],2))
            print('time:      ',round(t1 - t0,2), ' sec' )
            print('optimal condition:')
            print(f'acc_inf  = {acc_inf:.2e}')
            print(f'acc_opt  = {list(acc_opt)[0]:.2e}')
            print(f'acc_step = {acc_step.item():.2e}')
        self.result = {'obj':self.MLHessian(Xk=x)['objective'],'solution_x':x,'solution_y':diff['prediction']}



class Absorption(MLsqp):
    def Objective(self,x,y):

        return y


def sqp4ml():

    x_mean = [ 0 for _ in range(5)]
    x_std = [1 for _ in range(5)]
    y_mean = [0]
    y_std = [1]
    mean=[x_mean,y_mean]
    std=[np.array(x_std),np.array(y_std)]
    test=Absorption()
    ip = torch.tensor([2 for _ in range(5)])#局部最优
    test.setInitialPoint(ip)
    lb=np.array([-2 for _ in range(5)]+[-np.inf])
    ub=np.array([2 for _ in range(5)]+[np.inf])
    A=np.array([[0 for _ in range(5)]])
    G = np.array([[0 for _ in range(5)]])
    h = np.array([0])
    b = np.array([0])
    test.setConstraints(G=G, h=h, A=A, b=b, lb=lb, ub=ub)
    test.setMLmodel(mean=mean, std=std, model_path='./241210_griewank.pth')
    test.sqp_solve()





if __name__ == '__main__':
    sqp4ml()
    global t0, t1
    print('time cost: ', (t1-t0),'(sec)')