import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import sys
import TrainAndRoadCharacter as trc
import trainRunningModel as trm

UNIT_H = 20   # pixels
SPEED_H = 25  # grid height
DISTANCE_W = trc.SLStartPoint[-1]-trc.SLStartPoint[0] # grid width
UNIT_W=1
class TrainLine(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 30
	}
	def __init__(self,time):
		print("path:"+sys.path[0])
		#线路静态信息
		self.startPoint=trc.SLStartPoint[0]
		self.endPoint=trc.SLStartPoint[-1]
		self.S=self.endPoint-self.startPoint
		self.max_speed= 80 / 3.6
		self.T = time		  #运行总时间
		self.avg_speed= self.S/self.T   #平均速度
		self.dt = 0.2		  #运行时间步长
		self.low = np.array([0, 0])
		self.high = np.array([self.S, self.max_speed])
		self.ac = 0.8		#最大加速度
		self.de = 1		  #最大减速度
		self.viewer = None
		self.n_actions= 9		#-0.8 :0.2 :0.8
		self.n_features=4		#用于训练的特征，和self.state相对应
		self.action_space = ['-0.8', '-0.6', '-0.4', '-0.2', '0',  '0.2', '0.4', '0.6', '0.8']
		self.action_space = spaces.Discrete(9)     #下标从0开始
		self.observation_space = spaces.Box(self.low, self.high)
		self.seed()
		self.done=False
		self.filterFactor = 0.8
		#线路动态信息
		self.trd=trc.TrainAndRoadData()
		self.reset()
		
	def subFilterFactor(self,MaxEpisode):
		self.filterFactor -= 0.9/MaxEpisode
	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		reward =  0
		reward0 = 0
		reward1 = 0
		reward2 = 0
		reward3 = 0  #限速奖励
		reward4 = 0   #舒适度奖励
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
		du = (action - 4) *0.2  # 转化为[-0.3, 0.3]
#		u = round((self.u + action*self.dt), 1)
		u =	du * (1- self.filterFactor) + self.u  * self.filterFactor  #滤波思想
		u = max(-0.8, u)
		u = min(0.8, u)
		#上限速
		ebkv=self.trd.getEmerencyBrakeSpeed(self.pos +  self.veo * self.dt)
		if (self.veo > ebkv) or self.veo**2 >2* (self.endPoint -self.pos) * 0.6:
			u = -self.de
			reward3 += -0.1
			action = 0
		#下限速	
		resDis = self.endPoint - self.pos
		lowVeo = min(8,math.sqrt(2* 0.2* resDis))
		if (self.veo < lowVeo): 
			self.veo = lowVeo 
			u = self.ac	
			reward3 += -0.1
			action = 8
		reward4 += -1* abs(self.u - u) * self.dt * 0.3
		self.u = u
		trainState=self.train.Step(self.u) #更新列车状态
		pos_cha =trainState['S'] -self.pos
		self.pos=trainState['S']
		self.veo=trainState['v']
		dE=trainState['P']/trc.M	 #单位质量的能耗
		self.EC=self.EC+dE*self.dt	
		if self.pos>=self.endPoint:
			self.done = True
			reward += (self.endPoint-self.startPoint) *5.0  #终点奖励
		elif self.step1*self.dt >= 300:
			self.done = True
		else:
			self.done = False
		if u > 0:
			reward0 += -dE *self.dt * 3.0	#做功
			reward2 += pos_cha * 0.5 #位移
		t = self.step1 *self.dt
#		Tbar = t + (self.endPoint - self.pos)/(self.veo)   
		d1dv = 1/(self.veo + 0.1)
		distance = self.endPoint-self.startPoint
		dT_error = (2*t*d1dv -2*self.T*d1dv + self.T**2/distance) * pos_cha
		self.TErrorSum = self.TErrorSum * 0.99 + dT_error *1.0
		reward1 +=  -1 * dT_error * 3.0	
		reward += reward0 +reward1 + reward2 + reward3 + reward4 
#		self.state = (self.pos - self.startPoint,self.veo, t, self.u) #位置,速度,时间,能耗
		s = (self.pos - self.startPoint)/self.S
		self.state = (s,self.veo/self.max_speed,t/self.T,self.u) #位置,速度,能耗
		self.step1 += 1
		return np.array(self.state), self.EC, reward, self.done, action
		
	def get_refer_time(self, position):
		position = position - self.startPoint #转换成相对距离
		distance = self.endPoint - self.startPoint
		s1 = (80 / 3.6) ** 2 / 2 / self.ac
		s2 =  distance - ((80 / 3.6) ** 2 / 2 / self.de)
		t1 = 80/3.6/self.ac
		t2 = (s2 - s1) / (80/3.6)
		t3 = 80/3.6/self.de
		tz = t1 + t2 + t3
		if position <= s1:  #从s=10,v=4 开始
			v_max = math.sqrt((position )*2*self.ac )
			t_min = (v_max)/self.ac
		elif position <= s2:
			t_min = (position - s1) / (80/3.6)
			t_min = t1 + t_min
		elif position <= self.endPoint:
			temp = max(((80/3.6)**2 - (position - s2)*2*self.de),0)
			v_max = math.sqrt(temp)
			t_min = (80/3.6 - v_max)/self.de
			t_min = t1 + t2 + t_min
		else:
			t_min = tz
		tr = (t_min / tz) * self.T
		self.tr = tr
		return self.tr		


	def reset(self):
		self.train=trm.Train_model(self.startPoint,1,0.6,self.dt)  #列车模型
		self.EC=0   #消耗的能耗
		self.pos=self.startPoint
		self.veo=1
		self.u = 0.6 #初始加速度为0.6m/s2
		self.step1=0  #仿真步数
		s = (self.pos - self.startPoint)/self.S
		self.state = np.array([s,1/self.max_speed,0,0.6])  #列车状态:起点位置,速度,时间
		self.TErrorSum = 0
		#self.state = np.array([restPos,0])  #列车状态：相对起点位置,速度,时间
		return np.array(self.state)


	def bef_print(self):
		for i in range(2):
			position = 0.5 * self.ac * i**2
			velocity = i * self.ac
			f1 = open('datat.txt', 'r+')
			f1.read()
			print(position, velocity, file=f1)
			f1.close()

	def render(self, mode = 'human'):
			screen_width = 1300
			screen_height = 500

			world_width = 2400
			world_height=25
			scale_w = screen_width / world_width
			scale_h=screen_height/world_height
			trainwidth = 60
			trainheight = 20

			self.store=  [0 for x in range(0, 1000)]
			self.store_= [0 for x in range(0, 1000)]

			if self.viewer is None:
				from gym.envs.classic_control import rendering
				self.viewer = rendering.Viewer(screen_width, screen_height)
				xs = np.linspace(0, 2400, 2400)
				ys = np.linspace(0,25,25)
				xys = list(zip(xs*scale_w, ys * scale_h))

				self.track = rendering.make_polyline(xys)
				self.track.set_linewidth(8)
				#self.viewer.add_geom(self.track)

				clearance = 10

				l, r, t, b = -trainwidth / 2, trainwidth / 2, trainheight, 0
				train = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
				train.add_attr(rendering.Transform(translation=(0, clearance)))
				self.traintrans = rendering.Transform()
				train.add_attr(self.traintrans)
				self.viewer.add_geom(train)
				frontwheel = rendering.make_circle(trainheight / 2.5)
				frontwheel.set_color(.5, .5, .5)
				frontwheel.add_attr(rendering.Transform(translation=(trainwidth / 4, clearance)))
				frontwheel.add_attr(self.traintrans)
				self.viewer.add_geom(frontwheel)
				backwheel = rendering.make_circle(trainheight / 2.5)
				backwheel.add_attr(rendering.Transform(translation=(-trainwidth / 4, clearance)))
				backwheel.add_attr(self.traintrans)
				backwheel.set_color(.5, .5, .5)
				self.viewer.add_geom(backwheel)
				flagx = 2350 * scale_w
				flagy1 = 0
				flagy2 = flagy1 + 50
				flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
				self.viewer.add_geom(flagpole)
				flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
				flag.set_color(.8, .8, 0)
				self.viewer.add_geom(flag)


				x1=rendering.Line((0,60/3.6*scale_h),(50*scale_w,60/3.6*scale_h))
				self.viewer.add_geom(x1)
				x1.set_color(255,0,0)
				x2=rendering.Line((50*scale_w,60/3.6*scale_h),(50*scale_w,70/3.6*scale_h))
				x2.set_color(255, 0, 0)
				self.viewer.add_geom(x2)
				x3=rendering.Line((50*scale_w,70/3.6*scale_h),(1000*scale_w,70/3.6*scale_h))
				x3.set_color(255, 0, 0)
				self.viewer.add_geom(x3)
				x4=rendering.Line((1000*scale_w,70/3.6*scale_h),(1000*scale_w,80/3.6*scale_h))
				x4.set_color(255, 0, 0)
				self.viewer.add_geom(x4)
				x5=rendering.Line((1000*scale_w,80/3.6*scale_h),(2200*scale_w,80/3.6*scale_h))
				x5.set_color(255, 0, 0)
				self.viewer.add_geom(x5)
				x6=rendering.Line((2200*scale_w,80/3.6*scale_h),(2200*scale_w,60/3.6*scale_h))
				x6.set_color(255, 0, 0)
				self.viewer.add_geom(x6)
				x7=rendering.Line((2200*scale_w,60/3.6*scale_h),(2400*scale_w,60/3.6*scale_h))
				x7.set_color(255, 0, 0)
				self.viewer.add_geom(x7)
				if self.state[0]==0:
				   x0 = 0
				   y0 = 0
				   x1 =100
				   y1 = 20
				   outline = rendering.Line((x0*scale_w, y0*scale_h), (x1*scale_w, y1*scale_h))
				   outline.set_color(0,255,0)
				   self.viewer.add_geom(outline)



			pos = self.state[0]
			self.traintrans.set_translation(pos*scale_w, 0)
			#self.traintrans.set_rotation(math.cos(3 * pos))
			return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def close(self):
		   if self.viewer: self.viewer.close()
