import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd


def distmat(x, y):
    return np.sum(x**2, 0)[:, None] + np.sum(y**2, 0)[None, :] - 2*x.transpose().dot(y)


def normalize(a):
    return a/np.sum(a)


def add_intercept(dfX):
    df_temp = dfX.copy()
    df_temp['intercept'] = 1
    l_col = list(df_temp.columns)
    l_col.remove('intercept')
    return df_temp[['intercept']+l_col]


def add_order(dfX, k):
    df_temp = dfX.copy()
    l_col = list(df_temp.columns)
    if 'intercept' in l_col:
        l_col.remove('intercept')
    for col in l_col:
        for i in range(k-1):
            df_temp[str(col)+'_order_'+str(i+2)] = df_temp[col].apply(lambda x: x**(i+2))
    return df_temp


class VectorQuantileRegression:

    def __init__(self, order=1):

        self.X = None
        self.Y = None
        self.U = None
        self.d = None
        self.m = None
        self.n = None
        self.q = None
        self.step = None
        self.df = None
        self.order = order

    def get_U(self, d, step):

        if d > 6:
            print("Only d<=6 is yet supported")
            return None

        elif d == 1:
            u = np.arange(0, 1+step, step).T

        elif d == 2:
            x = np.arange(0, 1+step, step)
            x, y = np.meshgrid(x, x)
            u = np.array([x.flatten(), y.flatten()]).T

        elif d == 3:
            x = np.arange(0, 1+step, step)
            x, y, z = np.meshgrid(x, x, x)
            u = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        elif d == 4:
            x = np.arange(0, 1+step, step)
            x, y, z, x1 = np.meshgrid(x, x, x, x)
            u = np.array([x.flatten(), y.flatten(),
                          z.flatten(), x1.flatten()]).T

        elif d == 5:
            x = np.arange(0, 1+step, step)
            x, y, z, x1, y1 = np.meshgrid(x, x, x, x, x)
            u = np.array([x.flatten(), y.flatten(),
                          z.flatten(), x1.flatten(), y1.flatten()]).T

        elif d == 6:
            x = np.arange(0, 1+step, step)
            x, y, z, x1, y1, z1 = np.meshgrid(x, x, x, x, x, x)
            u = np.array([x.flatten(), y.flatten(), z.flatten(), x1.flatten(),
                          y1.flatten(), z1.flatten()]).T
        return u

    def fit(self, X, Y, step=0.05, verbose=False):
        Y = Y.to_numpy().T
        if self.order > 1:
            X = add_order(X, self.order)
        X = add_intercept(X).to_numpy()

        self.X = X
        self.Y = Y

        self.q = X.shape[1]

        d = Y.shape[0]
        self.d = d
        self.step = step

        u = self.get_U(d, step)
        U = u.T
        self.U = U

        n = Y.shape[1]
        m = U.shape[1]

        self.n = n
        self.m = m

        nu = normalize(np.random.rand(n, 1))
        mu = normalize(np.random.rand(m, 1))

        C = distmat(U, Y)
        P = cp.Variable((m, n))
        ind_m = np.ones((m, 1))
        constraints = [0 <= P,
                       cp.matmul(P.T, ind_m) == nu,
                       cp.matmul(P, X) == cp.matmul(cp.matmul(mu, nu.T), X)]

        objective = cp.Minimize(cp.sum(cp.multiply(P, C)))
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=verbose)
        psi = constraints[1].dual_value
        b = constraints[2].dual_value

        self.result = result
        self.b = b
        self.psi = psi

    def get_dfU(self, U, b, step):
        u = U.T
        d = u.shape[1]
        dfU = pd.DataFrame(u)
        dim = [i for i in range(d)]
        self.dim = dim
        dfU[[str(i)+"_follower" for i in list(dfU.columns)]] = dfU[dfU.columns]

        for k in range(d):

            dfU_temp = dfU.copy()
            dfU_temp[k] = dfU_temp[k].apply(
                                        lambda x: x+step if x < 1 else x-step)

            find_in = list(dfU[dim].apply(
                                    lambda x: list(np.around(x, 3)), axis=1))
            dfU[str(k)+"_follower"] = dfU_temp[dim].apply(
                        lambda x: list(np.around(x, 3)), axis=1
                                                            ).apply(
                                                    lambda x: find_in.index(x)
                                                                    )

        dfU['b'] = pd.DataFrame(b).apply(np.array, axis=1)

        for i in range(d):
            dfU['beta_'+str(i)] = (dfU.loc[list(dfU[str(i)+"_follower"])][['b']].reset_index(drop=True) - dfU[['b']])/step

        beta = ['beta_'+str(i) for i in range(2)]
        dfU['beta'] = dfU[beta].apply(lambda x: np.vstack(x), axis=1)

        return dfU

    def predict(self, X=None, u_quantile=None, argument="U"):

        '''
        argument in {"U", "X"}
        u_quantile liste with quantiles
        '''
        U = self.U
        b = self.b
        step = self.step
        X = add_intercept(X)
        if self.order > 1:
            X = add_order(X, self.order)
        m = self.m

        if argument == "X":

            if self.df is None:
                df = self.get_dfU(U, b, step)
                self.df = df
            else:
                df = self.df

            ser = pd.Series([u_quantile]*m)

            pos = df[self.dim].apply(lambda x: list(np.around(x, 3)), axis=1)
            beta = df['beta'][pos == ser].iloc[0]

            xeval = X.apply(lambda x: np.array(x).reshape(-1,1), axis=1).to_frame()
            xeval.columns = ['X']
            df_res = xeval.copy()

            if self.q == 1:
                df_res['y_pred'] = df_res['X'].apply(lambda x: beta*x)
            else:
                df_res['y_pred'] = df_res['X'].apply(lambda x: np.matmul(beta,x))

            return df_res

        elif argument == "U":
            if xeval.shape != (self.q,):
                print("If argument = U then you can only give one observation.")
                return

            if self.df is None:
                df = self.get_dfU(U, b, step)
                self.df = df
            else:
                df = self.df

            df['y_pred'] = df['beta'].apply(lambda x: np.matmul(x, X))

            return df[self.dim + ['y_pred']]

        else:
            print("argument not recognized")
            return None

#    def plot_surface():
#        "une fonction pour ploter les surfaces"

#    def plot_lines():
