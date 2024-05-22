from layers import GravNetLayer

l = GravNetLayer()
l['input_dense'] = (4, 2, 1)
print(type(l))
print(l.keys())
print(l['input_dense'])

l['output_dense']
