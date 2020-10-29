predict_names = ['Jessie', 0.97]

time_stamp = []
time_stamp.append(['10/26 19:00', 3])
time_stamp.append(['10/26 19:10', 4])
time_stamp.append(['10/26 19:20', 6])
time_stamp.append(['10/26 19:30', 5])
time_stamp.append(['10/26 19:40', 4])
time_stamp.append(['10/26 19:50', 1])
time_stamp.append(['10/26 20:00', 2])
time_stamp.append(['10/26 20:10', 3])
time_stamp.append(['10/26 20:20', 1])
time_stamp.append(['10/26 20:30', 2])
time_stamp.append(['10/26 20:40', 3])
time_stamp.append(['10/26 20:50', 1])
time_stamp.append(['10/26 21:00', 2])

from matplotlib import pyplot
import pandas as pd

time_stamp = pd.DataFrame(time_stamp)
pyplot.figure(figsize=(10, 5))
pyplot.plot(time_stamp[0], time_stamp[1])
pyplot.savefig('distract_score_' + predict_names[0] + '.png')
