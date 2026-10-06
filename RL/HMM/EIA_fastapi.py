from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()
res_list=["R1","R2", "R3"]
r1, r2=50, [25,25]
act_set = {"c_1":-1,"c0":0,"c+1":1}
act_keys= list(act_set.keys())
reward_theta = []
q_table = np.random.rand(len(res_list),1)


def avg_periodic_episode(list_theta):
    pass



def get_transition(res,list_theta):
    # action  
    # define policy (ask sir)
    action =  act_set[np.random.randint(0,3)] # rand int between 0 to 3-1 (including 2)
    reward = r1*avg_periodic_episode(list_theta)
    
     







def pseudo_model(res):
    if res==res_list[0]:
        return 0.5
    elif res==res_list[1]:
        return 0.65
    elif res==res_list[2]:
        return 0.9
    else:
        raise Exception(f"Valid Resolution {res_list}")





class RequestData(BaseModel):
    resolution: str
    efficiency: str

@app.post("/send/")
async def receive_message(data: RequestData):
    print(f"request data: {data}")
    get_transition(data.resolution, data.efficiency)
    return {"message": f"Received: {data.resolution}","control":"c2"}