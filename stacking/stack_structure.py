import numpy as np


class StackStructure:
    def __init__(self, sensor, t_window_length, t_series_length):
        self.sensor = sensor
        self.t_window_length = t_window_length
        self.t_series_length = t_series_length
        self.days1 = np.arange(self.t_series_length)
        self.days2 = self.days1 if self.t_series_length <= self.t_window_length else np.arange(self.t_window_length)
        self.master = self.initialize_master()
        self.scheme = self.initialize_scheme()

    def initialize_master(self):
        master = {}
        if self.sensor == 'icesat2':
            beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
            for beam in beams.tolist():
                master[beam] = {}
                for day1 in self.days1.tolist():
                    master[beam][day1] = {}
                    for day2 in self.days2.tolist():
                        master[beam][day1][day2] = {}
        else:
            for day1 in self.days1.tolist():
                master[day1] = {}
                for day2 in self.days2.tolist():
                    master[day1][day2] = {}
        return master

    def initialize_scheme(self):
        if self.sensor == 'icesat2':
            self.scheme = np.zeros([len(self.master), len(self.days1.tolist()), len(self.days2.tolist())])
        else:
            self.scheme = np.zeros([len(self.days1.tolist()), len(self.days2.tolist())])
        return self.scheme

    def get_master(self):
        return self.master

    def get_scheme(self):
        return self.scheme
