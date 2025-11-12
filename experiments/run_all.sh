echo "Running all synthetic data generation experiments..."

# Adults: run with 10 iterations
python3 script/CopulaGAN.py --dataset adults --iterations 10
python3 script/CTGAN.py --dataset adults --iterations 10
python3 script/Gauss_corr.py --dataset adults --iterations 10
python3 script/gaussian_copula.py --dataset adults --iterations 10
python3 script/TVAE.py --dataset adults --iterations 10

# Car Evaluation
python3 script/CopulaGAN.py --dataset car_evaluation
python3 script/CTGAN.py --dataset car_evaluation
python3 script/Gauss_corr.py --dataset car_evaluation
python3 script/gaussian_copula.py --dataset car_evaluation
python3 script/TVAE.py --dataset car_evaluation

# Balance Scale
python3 script/CopulaGAN.py --dataset balance_scale
python3 script/CTGAN.py --dataset balance_scale
python3 script/Gauss_corr.py --dataset balance_scale
python3 script/gaussian_copula.py --dataset balance_scale
python3 script/TVAE.py --dataset balance_scale

# Nursery
python3 script/CopulaGAN.py --dataset nursery
python3 script/CTGAN.py --dataset nursery
python3 script/Gauss_corr.py --dataset nursery
python3 script/gaussian_copula.py --dataset nursery
python3 script/TVAE.py --dataset nursery

# Student Performance
python3 script/CopulaGAN.py --dataset student_performance
python3 script/CTGAN.py --dataset student_performance
python3 script/Gauss_corr.py --dataset student_performance
python3 script/gaussian_copula.py --dataset student_performance
python3 script/TVAE.py --dataset student_performance

# Student Dropout & Success
python3 script/CopulaGAN.py --dataset student_dropout_success
python3 script/CTGAN.py --dataset student_dropout_success
python3 script/Gauss_corr.py --dataset student_dropout_success
python3 script/gaussian_copula.py --dataset student_dropout_success
python3 script/TVAE.py --dataset student_dropout_success

echo "All experiments completed!"
