import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from pypower.ppoption import ppoption
from pypower.runopf import runopf
from scipy.sparse import lil_matrix
from pypower.case24_ieee_rts import case24_ieee_rts
import pandas as pd
import copy

## Import data format from Pypower
from pypower.idx_bus import PD, QD

def case24_failrate():
    failrate = {"genmttf": [], "genmttr": [], "genweeks": [], "brlambda": [], "brdur": []}  # 建立dict存放故障数据
    failrate["genmttf"] = np.array([450, 450, 1960, 1960, 450,
                                    450, 1960, 1960, 1200, 1200,
                                    1200, 950, 950, 950, 10000,
                                    2940, 2940, 2940, 2940, 2940,
                                    960, 960, 1100, 1100, 1980,
                                    1980, 1980, 1980, 1980, 1980,
                                    960, 960, 1150])    # 33个机组的平均正常运行时间
    failrate["genmttr"] = np.array([50, 50, 40, 40, 50,
                                    50, 40, 40, 50, 50,
                                    50, 50, 50, 50, 0.1,
                                    60, 60, 60, 60, 60,
                                    40, 40, 150, 150, 20,
                                    20, 20, 20, 20, 20,
                                    40, 40, 100])       # 33个机组的平均故障修复时间
    failrate["genweeks"] = np.array([2, 2, 3, 3, 2,
                                     2, 3, 3, 3, 3,
                                     3, 4, 4, 4, 0.1,
                                     2, 2, 2, 2, 2,
                                     4, 4, 6, 6, 2,
                                     2, 2, 2, 2, 2,
                                     4, 4, 5])          # 33台机组的计划维护时间
    failrate["brlambda"] = np.array([0.24, 0.51, 0.33, 0.39, 0.48, 0.38,
                                     0.02, 0.36, 0.34, 0.33, 0.30, 0.44,
                                     0.44, 0.02, 0.02, 0.02, 0.02, 0.40,
                                     0.39, 0.40, 0.52, 0.49, 0.38, 0.33,
                                     0.41, 0.41, 0.41, 0.35, 0.34, 0.32,
                                     0.54, 0.35, 0.35, 0.38, 0.38, 0.34,
                                     0.34, 0.45])                    # 38条线路平均一年故障次数
    failrate["brdur"] = np.array([16, 10, 10, 10, 10, 768, 10, 10, 35, 10, 10, 10,
                                  10, 768, 768, 768, 768, 11, 11, 11, 11, 11, 11, 11,
                                  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                                  11, 11])                           # 38条线路故障平均持续时间
    return failrate
def failprob():
    failrate = case24_failrate()
    probgen = failrate["genmttr"] / (failrate["genmttf"]+ failrate["genmttr"])   # 每台机组的事故停机率
    brmiu = 8760 / failrate["brdur"]
    probbr = failrate["brlambda"] / (failrate["brlambda"] + brmiu)               # 计算线路的事故停机率=年停机频率*停机时间/一年
    totalprob = np.hstack((probgen, probbr))                                     # 33个机组故障率和38条线路的故障率共71个故障率数据
    return totalprob
def mc_sampling(totalprob, SIMUNIT, Ng, Nl):
    eqstatus = np.random.rand(SIMUNIT, Ng + Nl) < np.tile(totalprob, (SIMUNIT,1))
    # 生成随机数与故障率进行比较,一次mc_sampling抽样100次，eqstatus为1表示原件故障，为0表示原件正常
    eqstatus[:, 14] = 0                          #   忽略节点14的机组的故障率
    """""
    # eqstatus = sparse(doubl(eqstatus))
    此处未转化稀疏矩阵
    """
    return eqstatus
def mc_simulation(para):
    ppopt = para[2]
    Ng = para[3]
    Nl = para[4]
    statusgen = para[0][0: Ng]
    Testsys = para[1]
    Testsys["gen"][0: Ng, 7] = 1-statusgen                                       # 此处已改
    statusbranch = para[0][Ng:Ng+Nl]
    Testsys["branch"][0:Nl, 10] = 1-statusbranch
    Result =runopf(casedata=Testsys, ppopt=ppopt)
    dns = Result["f"] + Testsys["load"]
    if dns < 0.1:
        dns = 0
    return dns


class NonSequentialMonteCarlo():
    def __init__(self):
        self.pwd = os.getcwd()
        self.n_processors = os.cpu_count()                                      # 表明程序开始
    def run( self, Testsys = case24_ieee_rts(), BETAlimit = 0.0017, ITER_max = 10000, SIMUNIT = 1000):
        Nb = Testsys["bus"].shape[0]                                            # Load test system，Nb为节点数，Ng为发电机组数，Nl为馈线数
        Ng = Testsys["gen"].shape[0]
        Nl = Testsys["branch"].shape[0]

        # Set initial value
        iter = 0
        betavalue = float('inf')          # The stopping criteria停止迭代的标准
        row_index = 0
        # Build matrices that have fix dimension to avoid changing size in each loop
        eqstatus_total = np.zeros((ITER_max, Ng + Nl + 3))# 建一个100000*（33+38+3）的矩阵
        beta_table = np.zeros((1, ITER_max // SIMUNIT))  # "//"除法得到的才是整数
        edns_table = np.zeros((1, ITER_max // SIMUNIT))  # 存放评价指标，大小为1*1000
        lole_table = np.zeros((1, ITER_max // SIMUNIT))
        plc_table = np.zeros((1, ITER_max // SIMUNIT))
        genbus = np.nonzero((Testsys["bus"][:, PD]))[0]       # 第三列（python中坐标是2）是节点的有功功率，表示节点有有功负荷,此处返回该列非零元素的索引，共有17个元素非零
        sizegenbus = genbus.shape[0]                         # 有负荷的节点数量赋值给sizegenbus
        Testsys["load"] = sum(Testsys["bus"][:, PD])          # 系统需要的总有功功率
        Testsys["gencost"] = np.tile([2, 0, 0, 3, 0, 0, 0], (Ng, 1))  # np.tile建立重复矩阵块（设置机组费用）
        # treat all load as negtive generator and set their parameters, then add these vitual generators to real gens
        # 将所有载荷视为负发电机，并设置其参数，然后将这些发电机加到实际的发电机中
        loadcost = np.tile([2, 0, 0, 3, 0, 1, 0], (sizegenbus, 1))    # np.tile建立重复矩阵块（负荷的“机组费用”）
        Testsys["gencost"] = np.append(Testsys["gencost"], loadcost, axis=0)
        Index = copy.deepcopy(Testsys["gen"][0:sizegenbus, :])  # 将前17台机组的数据取出
        Index[:, 0:10] = np.hstack((Testsys["bus"][genbus, 0].reshape(-1, 1), -Testsys["bus"][genbus, 2].reshape(-1, 1),
                            -Testsys["bus"][genbus, 3].reshape(-1, 1), np.zeros((sizegenbus, 1)),
                            -Testsys["bus"][genbus, 3].reshape(-1, 1), np.zeros((sizegenbus, 1)),
                            Testsys["baseMVA"] * np.ones((sizegenbus, 1)), np.ones((sizegenbus, 1)),
                            np.zeros((sizegenbus, 1)), -Testsys["bus"][genbus, 2].reshape(-1, 1)))
        #  负荷参数代替取出的机组数据，将负荷套入机组模型,上面矩阵取数注意与matlab相比坐标要减一
        Testsys["gen"] = np.append(Testsys["gen"], Index, axis=0)
        del Index
        Testsys["bus"][genbus, 2:4] = 0  # 将原来节点中的第3、4列（有功、无功）负荷设为零
        totalprob = failprob()           # 引用前面定义的函数
        ppopt = ppoption(PF_DC=1, VERBOSE=0, OUT_ALL=0, OPF_ALG_DC=200, OPF_FLOW_LIM=1) # 可以通过ppoption()采用默认变量来看里面需要什么样的输入，这个按照matlab来输入没问题吧？
        result = runopf(casedata=Testsys, ppopt=ppopt)
        while (betavalue > BETAlimit) & (iter < ITER_max):
            eqstatus_indi = mc_sampling(totalprob, SIMUNIT, Ng, Nl)   # eqstatus为元件的状态矩阵，为1表示元件故障，为0表示原件正常
            eqstatus_indi = np.hstack((eqstatus_indi, np.ones((eqstatus_indi.shape[0], 1)), np.zeros((eqstatus_indi.shape[0], 2))))
            # 在eqstatus_indi矩阵中加入三列，第一列代表状态重复次数，第二列记载切负荷量大小（没有切负荷则为零），第三列记载是否为容量不足
            eqstatus_indi, ia1 = np.unique(eqstatus_indi, axis=0, return_inverse=True)  # 找出抽样中的相同结果
            for i in range(eqstatus_indi.shape[0]):
                eqstatus_indi[i, Ng + Nl] = sum(ia1 == i)  # 将重复记录次数在第Ng + Nl + 1
            if iter:
                x = 0
                y = eqstatus_indi.shape[0]
                for i in range(y):
                    indi_x = eqstatus_indi[x, 0:Ng + Nl]
                    for j in range(row_index):
                        if (indi_x == eqstatus_total[j, 0:Ng + Nl]).all():
                            eqstatus_total[j, Ng + Nl] = eqstatus_total[j, Ng + Nl] + eqstatus_indi[x, Ng + Nl]  # 遇见相同的，就在eqstatus_total的计数中累加次数
                            eqstatus_indi = np.delete(eqstatus_indi, x, axis=0)
                            x = x - 1
                            break
                    x = x + 1
                parfortemp = np.zeros((eqstatus_indi.shape[0], 2))
                para = [0] * eqstatus_indi.shape[0]
                n_sample = [0] * eqstatus_indi.shape[0]
                for i in range(eqstatus_indi.shape[0]):
                    para[i] = [0] * 5
                    para[i][0] = eqstatus_indi[i, 0: Ng + Nl]
                    para[i][1] = Testsys
                    para[i][2] = ppopt
                    para[i][3] = Ng
                    para[i][4] = Nl
                with Pool(self.n_processors) as p:
                    load_shedding = list(p.map(mc_simulation, para))
                parfortemp[:, 0] = np.asarray(load_shedding)
                parfortemp[:, 1] = (parfortemp[:, 0]) != 0
                eqstatus_indi[:, Ng + Nl + 1: Ng + Nl + 3] = parfortemp
                eqstatus_total[row_index: row_index + eqstatus_indi.shape[0], :] = eqstatus_indi
                row_index = row_index + eqstatus_indi.shape[0]
            else:
                parfortemp = np.zeros((eqstatus_indi.shape[0], 2))
                para = [0] * eqstatus_indi.shape[0]
                c = [0] * eqstatus_indi.shape[0]
                for i in range(eqstatus_indi.shape[0]):
                    para[i] = [0] * 5
                    para[i][0] = eqstatus_indi[i, 0: Ng + Nl]
                    para[i][1] = Testsys
                    para[i][2] = ppopt
                    para[i][3] = Ng
                    para[i][4] = Nl
                with Pool(self.n_processors) as p:
                    load_shedding = list(p.map(mc_simulation, para))            # 计算负荷短缺值
                parfortemp[:, 0] = np.asarray(load_shedding)
                parfortemp[:, 1] = (parfortemp[:, 0]) != 0                      # 记录是否负荷短缺
                eqstatus_indi[:, Ng + Nl + 1: Ng + Nl + 3] = parfortemp
                eqstatus_total[row_index: row_index + eqstatus_indi.shape[0], :] = eqstatus_indi
                row_index = row_index + eqstatus_indi.shape[0]
            ## Update index
            edns = sum(eqstatus_total[0:row_index, Ng + Nl] * eqstatus_total[0: row_index, Ng + Nl + 1]) / (iter + SIMUNIT)
            lole = sum(eqstatus_total[0:row_index, Ng + Nl] * eqstatus_total[0: row_index, Ng + Nl + 2]) / (
                    iter + SIMUNIT) * 8760
            plc = sum(eqstatus_total[0:row_index, Ng + Nl] * eqstatus_total[0: row_index, Ng + Nl + 2]) / (iter + SIMUNIT)
            betavalue = (sum(eqstatus_total[0:row_index, Ng + Nl]*(
                    eqstatus_total[0:row_index, Ng+Nl+1] - edns)**2)) ** 0.5/(iter + SIMUNIT) / edns

            beta_table[0, ((iter + SIMUNIT) // SIMUNIT)-1] = betavalue
            edns_table[0, ((iter + SIMUNIT) // SIMUNIT)-1] = edns
            lole_table[0, ((iter + SIMUNIT) // SIMUNIT)-1] = lole
            plc_table[0, ((iter + SIMUNIT) // SIMUNIT)-1] = plc
            iter = iter + SIMUNIT
        return edns



if __name__=="__main__":
    ## Start the function
    non_sequential_monte_carlo = NonSequentialMonteCarlo()
    non_sequential_monte_carlo.run()

