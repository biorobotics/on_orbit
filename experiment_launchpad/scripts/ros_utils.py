#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import numpy as np


class ur_controllers():
    def __init__(self):
        self.prev_msgs=[]
        self.curr_msgs=[]
        self.moveon=False
        self.new_msg=String()
        self.statefull_l = []
        self.statefull_r = []
        self.timestam = []

    def movej(self,q,pub,a=1.4,v=1.05,t=0,r =0):
        new_msg = String()
        cmd_str="movej(["
        for i in q:
            cmd_str=cmd_str+str(i)+","

        cmd_str=cmd_str[:-1]+"],a="+str(a)+", v="+str(v)+", t="+str(t)+", r="+str(r)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def movel(self,q,pub,a=1.2,v=0.25,t=0,r =0):
        new_msg = String()
        cmd_str="movel(["
        for i in q:
            cmd_str=cmd_str+str(i)+","

        cmd_str=cmd_str[:-1]+"],a="+str(a)+", v="+str(v)+", t="+str(t)+", r="+str(r)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def servoj(self,q,pub,t=0.02,lookahead=0.1,gain=150):
        new_msg = String()
        cmd_str="servoj(["
        for i in q:
            cmd_str=cmd_str+str(i)+","
        cmd_str=cmd_str[:-1]+"], a=0, v=0,t="+str(t)+", lookahead_time="+str(lookahead)+", gain="+str(gain)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def speedj(self,qvel,pub,a=0.05,t=0.03):
        '''Inputs:
        t - command lifetime in seconds'''
        new_msg = String()
        cmd_str="speedj(["
        for i in qvel:
            cmd_str=cmd_str+str(i)+","

        cmd_str=cmd_str[:-1]+"], a="+str(a)+", t="+str(t)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def servo16j(self,q,pub,t=0.02,lookahead=0.1,gain=50):
        new_msg = String()
        cmd_str="servoj(["
        for i in q:
            cmd_str=cmd_str+str(i)+","
        cmd_str=cmd_str[:-1]+"], a=0, v=0,t="+str(t)+", lookahead_time="+str(lookahead)+", gain="+str(gain)+")"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg
    
    def servoc(self,q,pub,a=0.1, v=0.35, r=0):
        print(q)
        new_msg = String()
        cmd_str="servoc(["
        for i in q:
            cmd_str=cmd_str+str(i)+","
        cmd_str=cmd_str[:-1]+"], a=" + str(a) + ", v=" + str(v) +",r=0.01)"
        new_msg.data=cmd_str
        pub.publish(new_msg)
        self.new_msg=new_msg

    def jstate_in_left(self,data):
        jin=np.array(data.position)
        jvel=np.array(data.velocity)
        self.jstate_left=np.array([jin[2],jin[1],jin[0],jin[3],jin[4],jin[5]])
        self.jvel_left=np.array([jvel[2],jvel[1],jvel[0],jvel[3],jvel[4],jvel[5]])
        self.statefull_l.append(self.jstate_left)
        self.timestam.append(data.header.stamp.nsecs/1000000000 + data.header.stamp.secs)

    def jstate_in_right(self,data):
        jin=np.array(data.position)
        jvel=np.array(data.velocity)
        self.jstate_right=np.array([jin[2],jin[1],jin[0],jin[3],jin[4],jin[5]])
        self.jvel_right=np.array([jvel[2],jvel[1],jvel[0],jvel[3],jvel[4],jvel[5]])
        self.statefull_r.append(self.jstate_right)
        self.timestam.append(data.header.stamp.nsecs/1000000000 + data.header.stamp.secs)

    # def msgs_published(self,data):
    #     if data.data==self.new_msg.data:
    #         self.moveon=True
    #     else:
    #         self.moveon=False
        # self.curr_msgs=data.data
        # if self.curr_msgs==self.prev_msgs:
        #     self.moveon=False
        # else:
        #     self.prev_msgs=data.data
        #     self.moveon=True
        # jin=np.array(data.position) 
        # self.jstate=np.array([jin[2],jin[1],jin[0],jin[3],jin[4],jin[5]])

