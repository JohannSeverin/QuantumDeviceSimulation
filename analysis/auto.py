from .analysis import Analysis, SweepAnalysis


class AutomaticAnalysis(Analysis):

    def __init__(self, results):
        if results.number_of_expvals > 0:
            if results.number_of_sweeps in [0, 1, 2] and results.only_store_final:
                self.__class__ =  SweepAnalysis
                self.__init__(results)
        else:
            raise NotImplementedError("No automatic analysis implemented for this type of results")

