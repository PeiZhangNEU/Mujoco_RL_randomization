import os
import numpy as np

from gym.utils import EzPickle
from gym.envs.mujoco.hopper import HopperEnv

class Hopper2(HopperEnv, EzPickle):
    '''加入可以随机化参数的Hopper'''
    def __init__(self):
        HopperEnv.__init__(self)
        EzPickle.__init__(self)
    
    def object_ids(self, obj_name):
        '''根据xml里面的物体名称，得到该物体的bodyid和geomid，比如'object0, 得到的是 {body_id:32, geom_id:23}'’'''
        obj_id = {}

        try:
            obj_id['body_id'] = self.sim.model.body_name2id(obj_name)
        except:
            # print('Exception1')
            pass

        try:
            obj_id['geom_id'] = self.sim.model.geom_name2id(obj_name)
        except:
            # print('Exception2')
            pass

        return obj_id


    def set_property(self, obj_name, prop_name, prop_value):
        '''可以修改self.sim.model里面的任意属性，'''
        obj_id = self.object_ids(obj_name)     # {'body_id': 32, 'geom_id': 23}   一共有33个body， 22个geom，这里是指我们的object0他的body序号是32，geom序号是23
        object_type = prop_name.split('_')[0]
        object_type_id = object_type + '_id'

        prop_id = obj_id[object_type_id]
        prop_all = getattr(self.sim.model, prop_name)   # 是一个大列表， 我们需要根据刚刚得到的索引来修改对应的值
        # print('***',prop_name, object_type_id, prop_all[prop_id])#, prop_all[obj_id])

        prop_all[prop_id] = prop_value
        prop_all = getattr(self.sim.model, prop_name)


    def get_property(self, obj_name, prop_name):
        obj_id = self.object_ids(obj_name)

        object_type = prop_name.split('_')[0]
        object_type_id = object_type + '_id'

        prop_id = obj_id[object_type_id]
        prop_all = getattr(self.sim.model, prop_name)
        prop_val = prop_all[prop_id]
        return prop_val