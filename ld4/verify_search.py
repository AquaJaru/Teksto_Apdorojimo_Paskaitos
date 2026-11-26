import numpy as np
from student_search import cosine_rank
d=np.array([[1.,0.],[0.,1.],[1.,1.]])
r=cosine_rank(np.array([1.,0.]),d,3)
assert [i for i,s in r]==[0,2,1]
assert np.allclose([s for i,s in r],[1.,2**-.5,0.])
r=cosine_rank(np.array([0.,0.]),d,3)
assert all(np.isfinite(s) for i,s in r)
print('PASS: cosine scores, ordering and zero-query safety')
