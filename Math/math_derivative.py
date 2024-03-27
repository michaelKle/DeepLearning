import numpy as np
import matplotlib.pyplot as plt

import sympy as sym
import sympy.plotting.plot as symplot

x = sym.symbols('x')
fx = 2*x**2

df = sym.diff(fx,x)
print(f"fx={fx}")
print(f"fx'={df}")

symplot(fx, (x,-4,4), title='The fuction')
plt.savefig('fuction')

symplot(df, (x,-4,4), title='The derivative')
plt.savefig('deriv')

# now with relu and sigmoid
relu = sym.Max(0,x)
sigmoid = 1 / (1+sym.exp(-x))

p = symplot(relu, (x,-4,4), label='ReLU', show=False, line_color='blue')
p.extend(symplot(sigmoid, (x,-4,4), label='Sigmoid', show=False, line_color='red'))
p.legend = True
p.title = 'The functions'
#p.show()
plt.savefig('sigrelu')

# now their derivatives
p = symplot(sym.diff(relu), (x,-4,4), label='df(ReLU)', show=False, line_color='blue')
p.extend(symplot(sym.diff(sigmoid), (x,-4,4), label='df(Sigmoid)', show=False, line_color='red'))
p.legend = True
p.title = 'The functions'
#p.show()
plt.savefig('dfsigrelu')


## example of derivative of product of functions

gx = 4*x**3 - 3*x**4

df = sym.diff(fx)
dg = sym.diff(gx)

print(f"f={fx} f'={df}")
print(f"g={gx} g'={dg}")

mult = fx*gx
dmult = sym.diff(mult)
print(f"f*g={mult} (f*g)`={dmult}")

## example of chain rule
gx = x**2 + 4*x**3
fx = (gx)**5

df = sym.diff(fx)
dg = sym.diff(gx)

print(f"g={gx} g'={dg}")
print(f"f={fx} f'={df}")
