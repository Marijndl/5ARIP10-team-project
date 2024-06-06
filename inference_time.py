from model import CARNet
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CARNet()
model.to(device)
dummy_input_origin = torch.randn(2, 3, 1, dtype=torch.float).to(device)
dummy_input_shape = torch.randn(2, 3, 352, dtype=torch.float).to(device)

# Initiliaze loggers
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 2000
timings=np.zeros((repetitions,1))

# GPU warm-up phase
for _ in range(int(0.5*repetitions)):
    _ = model(dummy_input_origin, dummy_input_shape, dummy_input_origin, dummy_input_shape)

print("Completed warmup phase")
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input_origin, dummy_input_shape, dummy_input_origin, dummy_input_shape)
        ender.record()
        # Wait for GPU synch
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) #Elapsed time in milliseconds
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(f"We achieve a average inference time of: {mean_syn:.3f} ms with s.d. {std_syn:.3f}")
print(timings)

x = range(timings.shape[0])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, timings)
plt.show()