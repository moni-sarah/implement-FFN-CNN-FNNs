import subprocess
import sys

# Helper to run a script and capture its output
def run_and_capture(script):
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    return result.stdout

# Run each model script and extract results
print('Running FNN (Iris)...')
fnn_out = run_and_capture('fnn_iris.py')
print('Running CNN (CIFAR-10)...')
cnn_out = run_and_capture('cnn_cifar10.py')
print('Running RNN (Sine)...')
rnn_out = run_and_capture('rnn_sine.py')

# Extract metrics from output
import re
fnn_acc = re.search(r'Test accuracy: ([0-9.]+)', fnn_out)
cnn_acc = re.search(r'Test accuracy: ([0-9.]+)', cnn_out)
rnn_mse = re.search(r'Test MSE: ([0-9.]+)', rnn_out)

print('\n--- Results Summary ---')
print(f"FNN (Iris) Accuracy:   {fnn_acc.group(1) if fnn_acc else 'N/A'}")
print(f"CNN (CIFAR-10) Accuracy: {cnn_acc.group(1) if cnn_acc else 'N/A'}")
print(f"RNN (Sine) MSE:         {rnn_mse.group(1) if rnn_mse else 'N/A'}") 