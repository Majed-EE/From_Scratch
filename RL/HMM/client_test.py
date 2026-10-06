#testHMM

import requests
import time
from HMM_class import HMMModel

channel_instance=HMMModel()
channel_instance.print_model_parameters()

def send_hi_to_server(msg: str):
    url = "http://127.0.0.1:8000/send/"
    response = requests.post(url, json={"content": msg})
    print(response.json())
    # time.sleep(d)












observations = ["L1", "L2", "L3", "L4", "L5"]
delay_list=[0.1,0.2,0.3,0.4,0.5] *1000 # delay in mili seconds
# delay_dict = {}
states = ["R3D3","R3D2","R3D1","R2D3","R2D2","R2D1","R1D3","R1D2","R1D1"]

obs_seq,state_seq= channel_instance.generate_sequence()
for obs in range(len(obs_seq)):
    
    print(f"Sample {obs}\nObervation: {observations[obs_seq[obs]]}, State: {states[state_seq[obs]]}")
    latency=delay_list[obs_seq[obs]]
    print(f"delay: {latency}")
    msg="hi"
    try:
        send_hi_to_server(msg)
    except Exception as e:
        print(f"Error while sending {msg}: {e}")

    time.sleep(latency)




