NUMS=${1:-0}

echo "Traing agent with TTMin..."

echo "Start train dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -name dqn -s 10000 -r diff-travel-time

echo "Start train double_dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -name double_dqn -s 10000 -r diff-travel-time

echo "Start train dueling_dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -name dueling_dqn -s 10000 -r diff-travel-time

echo "Start train double_dueling_dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -name double_dueling_dqn -s 10000 -r diff-travel-time