python -m virtualenv pttf_env
source pttf_env/bin/activate
pip install --no-cache-dir -r requirements.txt

python src/utils/tf.py
python src/utils/pt.py
