

class ActionEMGRecord(object):
    def __init__(self, tup, dataset_conf):
        #self._index = str(tup[0])
        self._series = tup[1]       #forse vuole tup[0]
        self.dataset_conf = dataset_conf

    @property
    def id(self):
        return self._series['id']

    @property
    def myo_right_readings(self):
        return self._series['right_readings']

    @property
    def myo_left_readings(self):
        return self._series['left_readings']


    @property
    def label(self):
        if 'label' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['label']
