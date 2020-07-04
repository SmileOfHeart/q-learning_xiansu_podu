#from dqn_env import TrainLine
import sys
sys.path.append('.\subway_system')

from subway_env import TrainLine
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as mplt
import tensorflow as tf
import pandas as pd
import TrainAndRoadCharacter as trc

def plot(r,ylabel):
	import matplotlib.pyplot as plt
	plt.plot(np.arange(len(r)), r, linewidth=1)
	plt.ylabel(ylabel)
	plt.xlabel('training episodes')
	plt.savefig("./img/"+ylabel+".png")
	plt.show()

def draw_mean(r,ylabel):
	import matplotlib.pyplot as plt
	x_10 = []
	temp = []
	count = 0
	for i in range (len(r)):
		temp.append(r[i])
		count += 1
		if count >= 10:
			x_10.append(sum(temp) / 10)
			temp = []
			count = 0
	plt.plot(np.arange(len(x_10)), x_10, linewidth=1)
	plt.ylabel('mean' + ylabel)
	plt.xlabel('training episodes X10')
	plt.savefig("./img/"+'mean' +ylabel+".png")
	plt.show()

def run_train():
	total_step = 0
	Max_iteras= 3000
	for episode in range(Max_iteras):
		#训练5000次
		r1_max = 0
		step = 0
		r1 = 0
		pl=[]     #位置
		vl=[]     #速度
		ul=[]     #加速度
		al=[]     #动作
		# initial observation
		observation = env.reset()
		#env.bef_print()
		while True:
			# fresh env
			#env.render()
			# RL choose action based on observation
			
			action = RL.choose_action(observation)
			#强行推上曲线
			pos = observation[0] * env.S
			veo = observation[1] * env.max_speed
			if pos <100 and veo < env.avg_speed:
				action = 8			
			# RL take action and get next observation and reward
			observation_,E,reward, done,  action = env.step(action) # action =0-6 最后会被转换到转化为[-0.3, 0.3]

			r1 = r1 * 0.99 + reward

			RL.store_transition(observation, action, reward, observation_)
			if (total_step > 5000 and total_step % 32 == 0 ):
				RL.learn()
			# swap observation
			observation = observation_
#			o1 =observation
			if episode%20==0 or episode==Max_iteras-1:
				pl.append(pos)
				vl.append(veo)
				ul.append(observation[3])
				al.append(action)
			# break while loop when end of this episode
			if done:
#				env.subFilterFactor(Max_iteras)    #减少平滑因子
				r.append(r1)
				energy.append(E)
				print(observation_[2]*env.T,env.TErrorSum,env.filterFactor,RL.epsilon)
				RL.increase_epsilon()
				tlist.append(observation_[2]*env.T)
				#曲线判定函数，决定是否保存曲线 ：旅行距离是否合适，时间是否接近，以episode_speed.csv为 文件名
				if r1 > r1_max and episode>1500 and episode%20 == 0:
					r1_max =r1
					Curve=np.mat([pl,vl,ul,al])
					CurveData=pd.DataFrame(data=Curve.T,columns=['s','v','acc','action'])
					CurveData.to_csv("./Curve/"+str(episode)+"_CurveData.csv")					
				if episode==Max_iteras-1:
					print(r1)
#					f1 = open('datat.txt', 'r+')
#					f1.read()
#					print(episode, (step + 5)/5, file=f1)
#					f1.close()
					r.append(r1)
					print('Episode finished after {} timesteps'.format((step + 5)/5))
				break
#			if (5000 > episode >= 4500):
#				 print(o1)
#				 f2 = open('vs.txt', 'r+')
#				 f2.close()
#				 break
			step += 1
			total_step += 1
		#最后打印结果
		print(episode)
		if episode%20 ==0 or episode==Max_iteras-1:
			trc.plotSpeedLimitRoadGrad('relative')
			mplt.plot(pl,vl)
			mplt.savefig("./img/"+str(episode)+"v-s.png")
			mplt.show()	
			mplt.plot(pl,ul)
			mplt.savefig("./img/"+str(episode)+"u-s.png")
			mplt.show()	
			draw_mean(al,str(episode)+"action-s")
#			mplt.savefig("./img/"+str(episode)+"action-s.png")
#			mplt.show()	
	return
			
# end of game


if __name__ == "__main__":
	print("path:"+sys.path[0])
	global r,energy,tlist,RL
	tf.reset_default_graph()
	env = TrainLine(110)
	env.seed(1)
	RL = DeepQNetwork(env.n_actions, env.n_features,
					  learning_rate=0.0001,
					  reward_decay=0.99,   #奖励折扣
					  e_greedy=0.6,         #探索效率
					  replace_target_iter=512,
					  memory_size=10000,
					  batch_size=256,
					  e_greedy_increment=0.35/3000,
					  # output_graph=True
					  )
#	RL.LoadModel()
	energy = []
	r = []
	tlist = [] 
	run_train()
	RL.plot_cost()
	plot(r,'reward')
	plot(energy,'energy')
	plot(tlist,'time')
	draw_mean(r,'reward')
	draw_mean(energy,'energy')
	draw_mean(tlist,'time')
	draw_mean(RL.cost_his,'mean_cost')
	rdata = pd.DataFrame(r)
	rdata.to_csv("reward.csv")
	tdata = pd.DataFrame(tlist)
	tdata.to_csv("timeError.csv")
	costData = pd.DataFrame(RL.cost_his)
	costData.to_csv("costData.csv")
	Edata = pd.DataFrame(energy)
	Edata.to_csv("EData.csv")