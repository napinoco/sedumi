% generate_maxstep_oracle.m
%
% Runs the real Octave maxstep.m using the same mixed L+Lorentz+real-SDP
% problem and interior point `x` as generate_updtransfo_oracle.m, with
% auxx built the same way wregion.m does (uxc.tdet from vfrm.lab,
% uxc.u = psdfactor(x,K)), and a small random step direction dx so the
% step stays comfortably inside the feasible region (tp > 1).
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_maxstep_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'sdinit');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 88);

K.f = 0; K.l = 2; K.q = [3;4]; K.r = zeros(1,0); K.s = [2;3];
n = K.l + sum(K.q) + K.s(1)^2 + K.s(2)^2;
m = 4;
At = rand(n,m) - 0.5;
b = rand(m,1) - 0.5;
c = rand(n,1) - 0.5;
pars = checkpars(struct());

[At2, b2, c2, K2, prep] = pretransfo(At, b, c, K, pars);

dense.cols = zeros(0,1);
dense.A = sparse(length(b2), 0);
dense.q = zeros(0,1);
dense.l = 0;

[dIN, v, vfrm0, y, y0, R] = sdinit(At2, b2, c2, dense, K2, pars);

function xv = make_point(K2, seed_add)
    rand('seed', 88 + seed_add);
    xv = rand(K2.N,1) - 0.5;
    xv(1:K2.l) = abs(xv(1:K2.l)) + 0.5;
    lorN = length(K2.q);
    xi = K2.l + lorN;
    for k = 1:lorN
        kk = K2.q(k) - 1;
        vecpart = xv(xi+1:xi+kk);
        xv(K2.l+k) = norm(vecpart) + 1;
        xi = xi + kk;
    end
    off = K2.lq;
    A1 = rand(2) - 0.5; X1 = A1*A1' + 2*eye(2);
    xv(off+1:off+4) = X1(:);
    off = off + 4;
    A2 = rand(3) - 0.5; X2 = A2*A2' + 3*eye(3);
    xv(off+1:off+9) = X2(:);
end
x = make_point(K2, 1);

% ---- build a genuine vfrm via updtransfo (same recipe as
% generate_updtransfo_oracle.m), using x for BOTH the primal and dual
% role (xc=zc=x) purely to get a valid, well-defined `w`/vfrm --
% maxstep.m only actually needs vfrm.lab, not any deeper meaning.
w.tdetx = tdet(x, K2);
w.tdetz = w.tdetx;
w.ux = psdfactor(x, K2);
w.s = psdscale(w.ux, x, K2);
ix = K2.mainblks;
detxz = w.tdetx .* w.tdetx / 4;
halfxz = (x(ix(1):ix(2)-1).*x(ix(1):ix(2)-1) + ddot(x(ix(2):ix(3)-1),x,K2.qblkstart)) / 2;
tmp = halfxz.^2 - detxz;
lab2q = halfxz + sqrt(max(tmp,0));
w.lab = [x(1:K2.l).*x(1:K2.l); detxz./lab2q; lab2q; psdeig(w.s,K2)];
dIN.perm = [];
[dOut, vfrm] = updtransfo(x, x, w, dIN, K2);

% ---- auxx, matching wregion.m's own construction ----
auxx.u = psdfactor(x, K2);
n_lor = length(K2.q);
lo = vfrm.lab(ix(1):ix(2)-1);
hi = vfrm.lab(ix(2):2*ix(2)-ix(1)-1);
auxx.tdet = 2 * lo .* hi;

% ---- small random step direction, safely inside the feasible region ----
dx = 0.05 * (rand(K2.N,1) - 0.5);

tp = maxstep(dx, x, auxx, K2);

save('-v7', fullfile(out_dir, 'maxstep.mat'), 'K2', 'x', 'dx', 'auxx', 'tp');

fprintf('maxstep oracle written to %s\n', out_dir);
