import torch

V_data = [1.,2.,3.]
V = torch.Tensor(V_data)
#print(V)

M_data = [[1.,2.,3.],[4.,5.,6.]]
M = torch.Tensor(M_data)

T_data = [[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]]
T = torch.Tensor(T_data)

#print(T.shape)
#print(T[0][0][1])

#x = torch.randn(2,3,4)
#print(x)
#print(x.view(2,12))
#print(x.view(2,-1))

# requires_grad=True로 설정하여 텐서의 연산을 추적
x = torch.randn(2, 2, requires_grad=True)
y = x * x

# y는 x에 대한 연산의 결과로 생성되었으므로 grad_fn 속성을 가짐
#print(y.grad_fn)

x = torch.randn(2, 2)
y = torch.randn(2, 2)
# 사용자가 생성한 Tensor는 기본적으로 ``requires_grad=False`` 를 가집니다
print(x.requires_grad, y.requires_grad)
z = x + y
# 그래서 z를 통해 역전파를 할 수 없습니다
print(z.grad_fn)

# ``.requires_grad_( ... )`` 는 기존 텐서의 ``requires_grad``
# 플래그를 제자리에서(in-place) 바꿉니다. 입력 플래그가 지정되지 않은 경우 기본값은 ``True`` 입니다.
x = x.requires_grad_()
y = y.requires_grad_()
# z는 위에서 본 것처럼 변화도를 계산하기에 충분한 정보가 포함되어 있습니다
z = x + y
print(z.grad_fn)

# 연산에 대한 입력이 ``requires_grad=True`` 인 경우 출력도 마찬가지입니다
print(z.requires_grad)
'''
# 이제 z는 x와 y에 대한 계산 기록을 가지고 있습니다
# z의 값만 가져가고, 기록에서 **분리** 할 수 있을까요?
new_z = z.detach()

# ... new_z 가 x와 y로의 역전파를 위한 정보를 갖고 있을까요?
# 아닙니다!
print(new_z.grad_fn)
# 어떻게 그럴 수가 있을까요? ``z.detach()`` 는 ``z`` 와 동일한 저장공간을 사용하지만
# 계산 기록은 없는 tensor를 반환합니다. 그 tensor는 자신이 어떻게 계산되었는지
# 아무것도 알지 못합니다.
# 본질적으로는 Tensor를 과거 기록으로부터 떼어낸 겁니다
'''

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
      print((x**2). requires_grad)