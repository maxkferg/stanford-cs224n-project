mkdir tmp && cd tmp
git clone https://www.github.com/datalogai/recurrentshop.git
cd recurrentshop
python setup.py install
cd ../../ & rm -rf tmp
cd ./data/glove/
source download.sh