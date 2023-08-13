""" example of integration with the py-modelrunner package """
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

import time
import mpi4py
import numba_mpi

# from modelrunner import ModelBase, submit_job
from PyMPDATA import Options

import scenarios
from PyMPDATA_MPI.domain_decomposition import subdomain
from PyMPDATA_MPI.hdf_storage import HDFStorage
from PyMPDATA_MPI.utils import setup_dataset_and_sync_all_workers


class SimulationModel:
    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        # output: Optional[str] = None,
        # *,
        # strict: bool = False,
    ):
        # super().__init__(parameters, output, strict=strict)
        self.parameters = parameters
        scenario_class = getattr(scenarios, parameters["scenario"])
        self.simulation = scenario_class(
            mpdata_options=Options(**parameters["mpdata_options"]),
            n_threads=parameters["n_threads"],
            grid=parameters["grid"],
            size=numba_mpi.size(),
            courant_field_multiplier=parameters["courant_field"],
            rank=numba_mpi.rank(),
        )
        if numba_mpi.rank() == 0:
            HDFStorage.create_dataset(
                name=parameters["dataset_name"],
                path=Path(parameters["output_datafile"]),
                grid=parameters["grid"],
                steps=parameters["output_steps"],
            )
        numba_mpi.barrier()
        self.storage = np.ones((*parameters["grid"], len(parameters["output_steps"])))#HDFStorage.mpi_context(
            #parameters["output_datafile"], "r+", mpi4py.MPI.COMM_WORLD
        #)

    def __call__(self):
        steps = self.parameters["output_steps"]
        x_range = slice(
            *subdomain(self.parameters["grid"][0], numba_mpi.rank(), numba_mpi.size())
        )
        print("grid: ",self.parameters["grid"], ' x_range: ', x_range)
        start1 = time.time()
        dataset = self.storage #setup_dataset_and_sync_all_workers(
            #self.storage, self.parameters["dataset_name"]
        #)
        #print("aftet db sync, rank: ", numba_mpi.rank(), ' time taken: ', time.time() - start1) 
        start = time.time()
        exec_time = self.simulation.advance(
            dataset=dataset, output_steps=steps, x_range=x_range
        )
        # print("exec_time: ", exec_time)
        return (exec_time/1000000, time.time() - start)

def benchmark(grids, results):
    for i, grid in enumerate(grids):
        for k in range(2):
            model = SimulationModel(
                parameters={
                "scenario": "CartesianScenario",
                "mpdata_options": {"n_iters": 1},
                "n_threads": 1,
                #"grid": (numba_mpi.size(), numba_mpi.size()),
                "grid": grid,
                "courant_field": (0.5, 0.5),
                "output_steps": (36,),
                #"output_steps": tuple(i for i in range(0, 25, 2)),
                "output_datafile": "output_psi"+str(i)+".hdf5",
                "dataset_name": "psi",
                }
            )
            res=model()
            print("exec time ", int(res[0]), ' time.time: ', res[1], " rank: ", numba_mpi.rank())
            if k==1 and numba_mpi.rank() == 0:
                results[str(grid)] = res[0]
        numba_mpi.barrier()
if __name__ == "__main__":
    # submit_job(
    #     __file__,
    #     parameters={
    #         "scenario": "CartesianScenario",
    #         "mpdata_options": {"n_iters": 2},
    #         "n_threads": 1,
    #         "grid": (96, 96),
    #         "courant_field": (0.5, 0.5),
    #         "output_steps": tuple(i for i in range(0, 25, 2)),
    #         "output_datafile": "output_psi.hdf5",
    #         "dataset_name": "psi"
    #     },
    #     output="output_times.json",
    #     method="foreground",
    # )
    results = {}
    grids = ((256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (9192, 9192)))
    #grids = ((4096, 4096),)
    benchmark(grids, results)
    #benchmark(((1024, 1024),), results)
    if numba_mpi.rank() == 0:
        plt.plot(results.keys(), results.values(),  marker='o', linestyle='-')
        #plt.xticks(range(len(results)), list(results.keys()))
        plt.savefig('results.png')
        print(results) 

