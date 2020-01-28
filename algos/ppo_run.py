import os
import argparse

def plot():
    os.system('python plot.py model/ppo')

def test():
    print('test')
    #TODO: load model, show the agent in the environment

#TODO: read plot, train or test from std_out and execute the proper function

parser = argparse.ArgumentParser(description='Helper to train, plot or visualize models.')
parser.add_argument("--cmd", default='train', help='a string for the action (plot, train, test)')
#parser.add_argument("--seeds", nargs='*', )
args = parser.parse_args()
cmd = args.cmd
if cmd=='plot':
    plot()
elif cmd=='train':
    for i in range(10):
        try:
            cmd = 'python vppo_gae_buf.py --output_dir model/vppo_walker/ppo' + str(i) + ' --exp_name vppo_walker'
            os.system(cmd)
        except:
            print('except')
            break
elif cmd=='test':
    test()
else:
    print('invalid cmd')