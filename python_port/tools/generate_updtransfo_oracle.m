% generate_updtransfo_oracle.m
%
% Runs the real Octave updtransfo.m on the same mixed L+Lorentz+real-SDP
% problem/scaling point `d` as the recent sdinit/getDAtm/PopK/pcg
% fixtures, with a genuine `w` struct built the same way trydif.m/
% widelen.m do (tdet/psdfactor/psdscale/psdeig), and real (not complex)
% x, z so the PSD blocks stay within qrK/urotorder/givensrot/sqrtinv's
% existing real-symmetric-only scope.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_updtransfo_oracle"

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
dIN.q1 = dIN.q1 .* (1 + 0.1*rand(size(dIN.q1)));
dIN.q2 = dIN.q2 + 0.05*(rand(size(dIN.q2))-0.5);
dIN.perm = [];   % first iteration: no prior pivot ordering

% ---- build genuine x, z (full internal-format vectors) with PD PSD blocks ----
function xv = make_point(K2, seed_add)
    rand('seed', 88 + seed_add);
    xv = rand(K2.N,1) - 0.5;
    xv(1:K2.l) = abs(xv(1:K2.l)) + 0.5;   % keep LP part positive
    % Keep each Lorentz block strictly inside the cone (x0 > ||xvec||),
    % since tdet(x)=x0^2-||xvec||^2 must stay positive for updtransfo's
    % own sqrt(w.tdetx./w.tdetz) to stay real -- a plain random vector
    % has no reason to satisfy that.
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
z = make_point(K2, 2);

w.tdetx = tdet(x, K2);
w.tdetz = tdet(z, K2);
w.ux = psdfactor(x, K2);
w.s = psdscale(w.ux, z, K2);
lab2q = zeros(0,1);
if ~isempty(K2.q)
    ix = K2.mainblks;
    detxz = w.tdetx .* w.tdetz / 4;
    halfxz = (x(ix(1):ix(2)-1).*z(ix(1):ix(2)-1) + ddot(x(ix(2):ix(3)-1),z,K2.qblkstart)) / 2;
    tmp = halfxz.^2 - detxz;
    if all(tmp > 0)
        lab2q = halfxz + sqrt(tmp);
    else
        lab2q = halfxz;
    end
end
detxz = w.tdetx .* w.tdetz / 4;
w.lab = [x(1:K2.l).*z(1:K2.l); detxz./lab2q; lab2q; psdeig(w.s,K2)];

[d, vfrm] = updtransfo(x, z, w, dIN, K2);

save('-v7', fullfile(out_dir, 'updtransfo.mat'), ...
     'K2', 'dIN', 'x', 'z', 'w', 'd', 'vfrm');

fprintf('updtransfo oracle written to %s\n', out_dir);
