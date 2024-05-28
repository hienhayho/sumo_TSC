echo "Start traing DoubleDQN with reward_fn: average-speed..."
CUDA_VISIBLE_DEVICES=0 python training/double_dqn.py -train -s 10000 -r average-speed

echo "---------------------------------------------------"

echo "Start traing DoubleDQN with reward_fn: diff-waiting-time..."
CUDA_VISIBLE_DEVICES=0 python training/double_dqn.py -train -s 10000 -r diff-waiting-time