import torch 

x = torch.ones(5)
y = torch.zeros(3)

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("loss: ", loss)

print("gradient function for x: ", x.grad_fn)
print("gradient function for y: ", y.grad_fn)
print("gradient function for w: ", w.grad_fn)
print("gradient function for b: ", b.grad_fn)
print("gradient function for z: ", z.grad_fn)
print("gradient function for loss: ", loss.grad_fn)

print("w before backprop: ", w)

loss.backward()
print("w after backprop: ", w)
print(w.grad)
print(b.grad)

z = torch.matmul(x, w) + b
print("requires grad: ", z.requires_grad)
z_det = z.detach()
print("requires grad: ", z_det.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print("requires grad: ", z.requires_grad)
  