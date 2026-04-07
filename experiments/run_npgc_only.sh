echo "Running all synthetic data generation experiments..."

# Car Evaluation
python3 script/npgc_script.py --dataset car_evaluation


# Balance Scale
python3 script/npgc_script.py --dataset balance_scale


# Nursery
python3 script/npgc_script.py --dataset nursery

# Student Performance
python3 script/npgc_script.py --dataset student_performance


# Student Dropout & Success
python3 script/npgc_script.py --dataset student_dropout_success


# Adults: run with 10 iterations
python3 script/npgc_script.py --dataset adults --iterations 10


echo "All experiments completed!"
