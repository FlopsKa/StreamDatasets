from changeds.abstract import ChangeStream, RegionalChangeStream, QuantifiesSeverity
from changeds.datastreams.abrupt import *
from changeds.datastreams.gradual import *
from changeds.datastreams.synthetic import Hypersphere, Gaussian

if __name__ == '__main__':
    stream = Hypersphere()
    print(stream.id())
    current_change = 0
    while stream.has_more_samples():
        x, y, is_change = stream.next_sample()
        if is_change:
            current_change += 1
            print("Change Nr. {} at index {}".format(current_change, stream.sample_idx))
            if isinstance(stream, QuantifiesSeverity):
                print("Severity is {}".format(stream.get_severity()))

    if isinstance(stream, RegionalChangeStream):
        change_regions = stream.approximate_change_regions()
        print(change_regions)
