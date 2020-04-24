import numpy as np
from pypower.ppoption import ppoption
from pypower.runopf import runopf
from scipy import sparse
from pypower.case24_ieee_rts import case24_ieee_rts
def case24_failrate():
    failrate = {"genmttf": [], "genmttr": [], "genweeks": [], "brlambda": [], "brdur": []}
    failrate["genmttf"] = np.array([450,450,1960,1960,450,450,1960,1960,1200,1200,1200,950,950,950,10000,2940,2940,2940,2940,2940,960,960,1100,1100,1980,1980,1980,1980,1980,1980,960,960,1150])
    failrate["genmttr"] = np.array([50,50,40,40,50,50,40,40,50,50,50,50,50,50,0.1,60,60,60,60,60,40,40,150,150,20,20,20,20,20,20,40,40,100])     # 33个机组的平均修复时间
    failrate["genweeks"] = np.array([2,2,3,3,2,2,3,3,3,3,3,4,4,4,0.1,2,2,2,2,2,4,4,6,6,2,2,2,2,2,2,4,4,5])                       #33台机组的计划维护时间
    failrate["brlambda"] = np.array([0.24,0.51,0.33,0.39,0.48,0.38,0.02,0.36,0.34,0.33,0.30,0.44,0.44,0.02,0.02,0.02,0.02,0.40,0.39,0.40,0.52,0.49,0.38,0.33,0.41,0.41,0.41,0.35,0.34,0.32,0.54,0.35,0.35,0.38,0.38,0.34,0.34,0.45])    #平均一年故障次数
    failrate["brdur"] = np.array([16,10,10,10,10,768,10,10,35,10,10,10,10,768,768,768,768,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11])   #线路故障平均持续时间
    return failrate
def failprob():
    failrate = case24_failrate()
    probgen = failrate["genmttr"] / (failrate["genmttf"]+ failrate["genmttr"])   # 机组事故停机率
    brmiu = 8760 / failrate["brdur"]
    probbr = failrate["brlambda"] / (failrate["brlambda"] + brmiu)
    totalprob = np.hstack((probgen, probbr))     #33个机组故障率和38条线路的故障率共71个故障率数据
    return totalprob
def mc_sampling(totalprob, SIMUNIT, Ng, Nl):
    eqstatus = np.random.rand(SIMUNIT, Ng + Nl) < np.tile(totalprob, (SIMUNIT,1))   # 生成随机数与故障率进行比较
    eqstatus[:, 14] = 0                          #   忽略节点14的机组的故障率
    #eqstatus = sparse.csr_matrix(eqstatus)       #   产生MCS抽样（是否故障）的的结果，一次实验抽样100次
    return eqstatus
def mc_simulation(eqstatus, Testsys, ppopt, Ng, Nl):
    statusgen = eqstatus[0: Ng]
    Testsys["gen"][0: Ng, 8] = 1-statusgen
    statusbranch = eqstatus[Ng:Ng+Nl]
    Testsys["branch"][0:Nl,10] = 1-statusbranch

    Result =runopf(casedata=Testsys, ppopt=ppopt)
    dns = Result["f"] + Testsys["load"]
    if dns < 0.1:
        dns = 0
        return dns

Testsys=case24_ieee_rts()
# Load test system，Nb为节点数，Ng为发电机组数，Nl为馈线数
Nb = Testsys["bus"].shape[0]
Ng = Testsys["gen"].shape[0]
Nl = Testsys["branch"].shape[0]
# Set initial value
BETAlimit = 0.0017
ITER_max = 100000
SIMUNIT = 100
iter = 0
lole = 0
edns = 0
betavalue = float('inf')
row_index = 0
# Build matrices that have fix dimension to avoid changing size in each loop
eqstatus_total = np.zeros((ITER_max, Ng+Nl+3))
beta_table = np.zeros((1, ITER_max//SIMUNIT))             # "//"除法得到的才是整数
edns_table = np.zeros((1, ITER_max//SIMUNIT))
lole_table = np.zeros((1, ITER_max//SIMUNIT))
plc_table = np.zeros((1, ITER_max//SIMUNIT))
genbus = np.nonzero(Testsys["bus"][:, 3])                 # 第三列是节点的有功功率，表示节点有有功负荷,此处genbus为tuple格式
sizegenbus = (genbus[0]).shape[0]
Testsys["load"]=sum(Testsys["bus"][:, 3])                 # 系统需要的总有功功率
Testsys["gencost"]= np.tile([2,0,0,3,0,0,0], (Ng,1))      # np.tile建立重复矩阵块

# treat all load as negtive generator and set their parameters, then add these vitual generators to real gens将所有载荷视为负发电机，并设置其参数，然后将这些发电机加到实际的发电机中
loadcost=np.tile([2,0,0,3,0,1,0], (sizegenbus,1))
Testsys["gencost"]=np.append(Testsys["gencost"],loadcost, axis=0)
Index = Testsys["gen"][0:sizegenbus,:]                    # 将前17台机组的数据取出
Index[:, 0:10] = np.hstack((Testsys["bus"][genbus, 1].reshape(-1,1),-Testsys["bus"][genbus, 3].reshape(-1,1),
                   -Testsys["bus"][genbus, 4].reshape(-1,1), np.zeros((sizegenbus, 1)),
                   -Testsys["bus"][genbus, 4].reshape(-1,1), np.zeros((sizegenbus, 1)).reshape(-1,1),
                   Testsys["baseMVA"] * np.ones((sizegenbus, 1)), np.ones((sizegenbus, 1)),
                   np.zeros((sizegenbus, 1)), -Testsys["bus"][genbus, 3].reshape(-1,1)))
Testsys["gen"]=np.append(Testsys["gen"], Index, axis=0)
del Index
Testsys["bus"][genbus, 2:4]=0                            # 将原来的节点有功、无功负荷设为零
totalprob = failprob()
ppopt = ppoption(PF_DC=1, VERBOSE=0, OUT_ALL=0, OPF_ALG_DC=200, OPF_FLOW_LIM= 1)
result = runopf(casedata=Testsys, ppopt=ppopt)
while (betavalue > BETAlimit) & (iter < ITER_max):
    eqstatus_indi = mc_sampling(totalprob, SIMUNIT, Ng, Nl)
    aa = np.ones((eqstatus_indi.shape[0], 1))
    bb = np.zeros((eqstatus_indi.shape[0], 2))
    eqstatus_indi = np.hstack((eqstatus_indi, aa, bb))
    eqstatus_indi, ia1 = np.unique(eqstatus_indi, axis=0, return_inverse=True)    # 找出抽样中的相同结果
    for i in range(eqstatus_indi.shape[0]):
        eqstatus_indi[i, Ng + Nl] = sum(ia1 == i)        # 将重复记录次数在第Ng + Nl + 1
    if iter:
        x = 0
        for i in range(eqstatus_indi.shape[0]):
            aa = eqstatus_indi[x, 0:Ng+Nl]
            for j in range(row_index-1):
                if (aa == eqstatus_total[j, 0:Ng+Nl]).all():
                    eqstatus_total[j, Ng+Nl] = eqstatus_total[j, Ng+Nl]+1
                    np.delete(eqstatus_indi, x, axis=0)
                    x = x-1
                    break
            x = x+1
        parfortemp = np.zeros((eqstatus_indi.shape[0], 2))
        for i in range(eqstatus_indi.shape[0]):
            parfortemp[i, 0] = mc_simulation(eqstatus_indi[i, 0:Ng + Nl], Testsys, ppopt, Ng, Nl)  # 此处计算切负荷的大小
        parfortemp[:, 1] = (parfortemp[:, 1]) != 0
        eqstatus_indi[:, Ng + Nl + 1: Ng + Nl + 3] = parfortemp
        eqstatus_total[row_index: row_index + eqstatus_indi.shape[0], :] = eqstatus_indi
        row_index = row_index + eqstatus_indi.shape[0]
        '''''
        for i in range(eqstatus_temp.shape[0]):
            eqstatus_temp[i, Ng + Nl] = sum(aa[ia1 == i, Ng + Nl])  # 记录重复数
        parfortemp = np.zeros((eqstatus_indi.shanpe[0], 2))
        for i in range(eqstatus_indi.shanpe[0]):
            parfortemp[i, 0] = mc_simulation(eqstatus_indi[i, 0:Ng + Nl], Testsys, ppopt, Ng, Nl)  # 此处计算切负荷的大小
        parfortemp[:, 1] = (parfortemp[:, 1]) != 0
        eqstatus_indi[:, Ng + Nl + 1: Ng + Nl + 3] = parfortemp
        eqstatus_total[row_index: row_index + eqstatus_indi.shape[0], :] = eqstatus_indi
        row_index = row_index + eqstatus_indi.shape[0]
        '''
    else:
        parfortemp = np.zeros((eqstatus_indi.shape[0], 2))
        for i in range(eqstatus_indi.shape[0]):
            parfortemp[i, 0] = mc_simulation(eqstatus_indi[i, 0:Ng + Nl], Testsys, ppopt, Ng, Nl)  # 此处计算切负荷的大小
        parfortemp[:, 1] = (parfortemp[:, 1]) != 0
        eqstatus_indi[:, Ng + Nl + 1: Ng + Nl + 3] = parfortemp
        eqstatus_total[row_index: row_index + eqstatus_indi.shape[0], :] = eqstatus_indi
        row_index = row_index + eqstatus_indi.shape[0]

    edns = sum(eqstatus_total[0:row_index, Ng + Nl]* eqstatus_total[0: row_index,Ng+ Nl+1]) / (iter + SIMUNIT)
    lole = sum(eqstatus_total[0:row_index, Ng + Nl]* eqstatus_total[0: row_index, Ng + Nl + 2]) / (iter + SIMUNIT) * 8760
    plc = sum(eqstatus_total[0:row_index, Ng + Nl]*eqstatus_total[0: row_index, Ng + Nl + 2]) / (iter + SIMUNIT)
    beta_table[0, (iter + SIMUNIT) // SIMUNIT] = betavalue
    edns_table[0, (iter + SIMUNIT) // SIMUNIT] = edns
    lole_table[0, (iter + SIMUNIT) // SIMUNIT] = lole
    plc_table[0, (iter + SIMUNIT) // SIMUNIT] = plc
    iter = iter + SIMUNIT
xx = 3











