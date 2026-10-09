from .factory import TemperatureSolverFactory
from .tempest_standard import TempestStandardSolver
from .tempestovsky import YarkovskySolver
from .tempestovsky_implicit import YarkovskyImplicitSolver

# Register available solvers
TemperatureSolverFactory.register("tempest_standard", TempestStandardSolver) 
TemperatureSolverFactory.register("tempestovsky", YarkovskySolver)
TemperatureSolverFactory.register("tempestovsky_implicit", YarkovskyImplicitSolver)
