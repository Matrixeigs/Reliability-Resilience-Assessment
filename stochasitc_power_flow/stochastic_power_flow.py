

from pypower import runpf
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from numpy import zeros, ones, shape, random

from scipy.sparse import csr_matrix as sparse

from matplotlib import pyplot

class StochasticPowerFlow():
    def __init__(self):
        self.name = "Stochastic Power Flow"
        from pypower.ppoption import ppoption
        opt = ppoption()
        opt["VERBOSE"] = 0
        opt["OUT_ALL"] = 0
        self.opt = opt

    def monte_carlo_simulation(self, power_networks, ns = 100, beta = 0.05):

        # mpc = ext2int(power_networks)
        mpc = power_networks
        nb = shape(mpc['bus'])[0]  ## number of buses
        result = zeros((ns, nb))
        base_load_P = mpc["bus"][:,PD]
        base_load_Q = mpc["bus"][:,QD]

        for i in range(ns):
            load_variation = random.randn(nb)
            mpc["bus"][:,PD] = base_load_P*(1+load_variation*0.5)
            mpc["bus"][:,QD] = base_load_Q*(1+load_variation*0.5)

            power_result_temp = runpf.runpf(mpc, ppopt=self.opt)

            result[i,:] = power_result_temp[0]["bus"][:,VM]
        pyplot.hist(result)
        pyplot.show()

        return result


if __name__=="__main__":


    from pypower import case4gs
    mpc = case4gs.case4gs()

    stochastic_power_flow = StochasticPowerFlow()

    result = stochastic_power_flow.monte_carlo_simulation(power_networks=mpc)
    print(result)