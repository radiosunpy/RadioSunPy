from radiosun.time import TimeRange
from radiosun.client import *

t = TimeRange('2017-09-01', '2017-09-03')
ratan_client = RATANClient()
ar_data = ratan_client.get_data(t)
t = ar_data[ar_data['Number'] == 2674]
print(t)