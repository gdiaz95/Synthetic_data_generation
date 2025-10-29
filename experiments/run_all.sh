echo "Running all synthetic data generation experiments..."

python3 script/CopulaGAN.py
python3 script/CTGAN.py
python3 script/Gauss_corr.py
python3 script/gaussian_copula.py
python3 script/TVAE.py

echo "All experiments completed!"