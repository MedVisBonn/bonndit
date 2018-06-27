class CylKurtosisModel(object):

    def __init__(self, gtab):
        self.gtab = gtab

    def fit(self, data):
        return CylKurtosisFit()


class CylKurtosisFit(object):

    def __init__(self):
        pass
