apt-get install software-properties-common -y
add-apt-repository ppa:sumo/stable -y
apt-get update
apt-get install sumo sumo-tools sumo-doc -y

echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
export LIBSUMO_AS_TRACI=1

pip install -e .

pip install -r requirements.txt