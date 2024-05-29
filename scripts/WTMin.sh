NUMS=$1

echo "Traing agent with WTMin..."

echo "Start train dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -train -name dqn -s 10000

echo "Start train double_dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -train -name double_dqn -s 10000

echo "Start train dueling_dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -train -name dueling_dqn -s 10000

echo "Start train double_dueling_dqn..."
CUDA_VISIBLE_DEVICES=$NUMS python tools/train.py -train -name double_dueling_dqn -s 10000