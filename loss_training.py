import tensorflow as tf
from tf.math import sqrt, square

def my_loss(registered, original):
    loss = torch.mean(sqrt(square(registered[0,:] - original[0,:]) + square(registered[1,:] - original[1,:]) + square(registered[2,:] - original[2,:])))
    return loss

model = nn.Linear(2, 2)
input_3D = torch.randn(1, 2)
target = torch.randn(1, 2)
angular_deformation = model(x)
registered = perform_deformation(angular_deformation, input_3D)
input_3D = convert_back(registered)
loss = my_loss(input_3D, input_2D)
loss.backward()
print(model.weight.grad)