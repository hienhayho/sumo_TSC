apt-get install software-properties-common -y
add-apt-repository ppa:sumo/stable -y
apt-get update
apt-get install sumo sumo-tools sumo-doc -y

echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
export LIBSUMO_AS_TRACI=1

cd sumo-rl
pip install -e .

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install loguru ipython tqdm nvitop