"""Numeric HOBO builder for the R-reactor selection problem.

Builds the penalised polynomial  E = sum_r c_r x_r^2 + f_r y_r
      + lam * [ sum_r (x_r - U + s_r)^2 + (d - sum_r a_r y_r x_r + s_d)^2 ]
directly as a monomial dictionary (no symbolic expansion), then exports to
hobotan (hobo_mix) and Amplify (Poly).  Variable order (same as the original
scripts for R=2):  y_0..y_{R-1}, x_0[n], x_1[n], ..., s_0[n], ..., s_d[n].
"""
import itertools, math
import numpy as np

class ReactorHOBO:
    def __init__(self, R=2, n=10, lam=1e4, c=(1.5,1.0), f=(6.1,5.9), alpha=(0.75,0.80), d=3.0, U=5.0, xl=0.0, drop_bounds=False):
        assert len(c)==len(f)==len(alpha)==R
        self.R,self.n,self.lam,self.c,self.f,self.alpha,self.d,self.U,self.xl = R,n,float(lam),c,f,alpha,d,U,xl
        self.drop_bounds = bool(drop_bounds)   # True: omit x<=U penalty and slacks s_1..s_R (bounds already enforced by the encoding)
        self.nvar = R + ((R+1) if self.drop_bounds else (2*R+1))*n
        self.w = (U-xl)/(2**n-1) * 2.0**np.arange(n)          # bit weights
        self.names = [f"y_{r}" for r in range(R)]
        for grp in [f"x{r+1}" for r in range(R)] + ([] if self.drop_bounds else [f"s{r+1}" for r in range(R)]) + [f"s{R+1}"]:
            self.names += [f"{grp}_{k}" for k in range(n)]
        self.poly = self._build()

    # ---- index helpers ----
    def y(self, r): return r
    def xbits(self, r): return range(self.R + r*self.n, self.R + (r+1)*self.n)
    def sbits(self, r): return range(0) if self.drop_bounds else range(self.R + (self.R+r)*self.n, self.R + (self.R+r+1)*self.n)
    def sdbits(self): return range(self.R + (1 if self.drop_bounds else 2)*self.R*self.n, self.nvar)

    # ---- polynomial algebra on dict{tuple(sorted idx): coef} with q^2=q ----
    @staticmethod
    def _mul(a, b):
        out = {}
        for ka, va in a.items():
            for kb, vb in b.items():
                k = tuple(sorted(set(ka) | set(kb)))
                out[k] = out.get(k, 0.0) + va*vb
        return out
    @staticmethod
    def _add(out, a, scale=1.0):
        for k, v in a.items():
            out[k] = out.get(k, 0.0) + scale*v
        return out
    def _lin(self, bits, extra=()):                # sum_k w_k q_k  (times optional extra vars)
        return {tuple(sorted(set(extra) | {b})): float(wk) for b, wk in zip(bits, self.w)}

    def _build(self):
        R, lam = self.R, self.lam
        E = {}
        for r in range(R):
            X = self._lin(self.xbits(r)); X[()] = self.xl
            self._add(E, self._mul(X, X), self.c[r])            # c_r x_r^2
            self._add(E, {(self.y(r),): self.f[r]})              # f_r y_r
            if not self.drop_bounds:
                # bound constraint (x_r - U + s_r)^2
                g = dict(X); g[()] = g.get((),0.0) - self.U
                self._add(g, self._lin(self.sbits(r)))
                self._add(E, self._mul(g, g), lam)
        # demand constraint (d - sum a_r y_r x_r + s_d)^2
        g = {(): self.d}
        for r in range(R):
            yx = self._lin(self.xbits(r), extra=(self.y(r),))
            if self.xl: yx[(self.y(r),)] = yx.get((self.y(r),),0.0) + self.xl
            self._add(g, yx, -self.alpha[r])
        self._add(g, self._lin(self.sdbits()))
        self._add(E, self._mul(g, g), lam)
        return {k: v for k, v in E.items() if abs(v) > 0.0}

    # ---- evaluation / decoding (float64) ----
    def decode(self, bits):
        b = np.asarray(bits, float)
        y = b[:self.R]
        x = np.array([self.xl + b[list(self.xbits(r))] @ self.w for r in range(self.R)])
        s = np.array([b[list(self.sbits(r))] @ self.w for r in range(self.R)]) if not self.drop_bounds else None
        sd = b[list(self.sdbits())] @ self.w
        obj = float(np.sum(self.c*x**2) + np.sum(self.f*y))
        prod = float(np.sum(self.alpha*y*x))
        pen = float((0.0 if self.drop_bounds else np.sum((x-self.U+s)**2)) + (self.d-prod+sd)**2)
        return dict(y=tuple(int(v) for v in y), x=x, obj=obj, prod=prod, energy=obj+self.lam*pen,
                    feasible=bool(prod >= self.d-1e-9))
    def energy(self, bits):
        b = np.asarray(bits, float)
        return sum(v*np.prod(b[list(k)]) for k, v in self.poly.items())

    # ---- exports ----
    def to_hobotan(self, device="cuda:0"):
        import torch
        terms = [(k, v) for k, v in self.poly.items() if len(k) > 0]
        offset = self.poly.get((), 0.0)
        ho = max(len(k) for k, _ in terms)
        idx = np.array([[k[0]]*(ho-len(k)) + list(k) for k, _ in terms]).T
        val = np.array([v for _, v in terms])
        indices = torch.tensor(idx, dtype=torch.int, device=device)
        values = torch.tensor(val, dtype=torch.float32, device=device)
        index_map = {nm: i for i, nm in enumerate(self.names)}
        return [indices, values, index_map, self.nvar, [], []], offset
    def to_amplify(self, gen):
        q = gen.array("Binary", self.nvar)
        from amplify import Poly
        p = Poly(0.0)
        for k, v in self.poly.items():
            t = Poly(v)
            for i in k: t = t * q[i]
            p = p + t
        return p, q
    def stats(self):
        deg = {}
        for k in self.poly: deg[len(k)] = deg.get(len(k), 0) + 1
        return dict(nvar=self.nvar, nterms=len(self.poly), terms_by_degree=dict(sorted(deg.items())))

if __name__ == "__main__":
    import time, sys
    # 1) speed / size for R=2,3,4 at several n
    for R in (2,3,4):
        for n in (5,10,20,30):
            c=(1.5,1.0,1.2,0.9)[:R]; f=(6.1,5.9,6.0,6.3)[:R]; a=(0.75,0.80,0.70,0.85)[:R]
            t=time.time(); m=ReactorHOBO(R,n,1e4,c,f,a); dt=time.time()-t
            print(f"R={R} n={n:2d} build {dt:6.2f}s  {m.stats()}")
    # 2) exact check against the original symbolic construction (R=2)
    try:
        import symengine as sp, re
        def symbolic(nlen, pen):
            list_var = sp.symbols("y_0:2")
            for i in ["x1","x2","s1","s2","s3"]: list_var += sp.symbols(i+"_0:{}".format(nlen))
            w=[5/(2**nlen-1)*2**k for k in range(nlen)]
            seg=lambda k: sum(wk*v for wk,v in zip(w,list_var[2+k*nlen:2+(k+1)*nlen]))
            y1,y2=list_var[0],list_var[1]; X1,X2,S1,S2,S3=(seg(k) for k in range(5))
            E=1.5*X1**2+X2**2+6.1*y1+5.9*y2+pen*((X1-5+S1)**2+(X2-5+S2)**2+(3-(y1*0.75*X1+y2*0.8*X2)+S3)**2)
            return list_var, sp.expand(E)
        for n in (4,6):
            lv, Es = symbolic(n, 1e4); m = ReactorHOBO(2, n, 1e4)
            rng = np.random.default_rng(0); maxerr = 0
            for _ in range(200):
                bits = rng.integers(0,2,m.nvar)
                es = float(Es.subs({v:int(b) for v,b in zip(lv,bits)}))
                maxerr = max(maxerr, abs(es - m.energy(bits))/max(1,abs(es)))
            print(f"symbolic-vs-numeric check n={n}: max rel err = {maxerr:.2e}")
    except ImportError:
        print("symengine not available; symbolic check skipped")
