import numpy as np
EPS=1e-12
def cosine_rank(query,documents,k=3):
    q=np.asarray(query,dtype=float).ravel()
    m=np.atleast_2d(np.asarray(documents,dtype=float))
    if m.shape[0]==0 or k<=0:return []
    qn=q/max(np.linalg.norm(q),EPS)
    mn=m/np.maximum(np.linalg.norm(m,axis=1,keepdims=True),EPS)
    scores=mn@qn
    scores=np.where(np.isfinite(scores),scores,0.0)
    order=np.argsort(-scores,kind='stable')[:min(k,len(scores))]
    return [(int(i),float(scores[i])) for i in order]
