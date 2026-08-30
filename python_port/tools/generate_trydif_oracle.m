% generate_trydif_oracle.m
%
% Runs the real Octave trydif.m on the same mixed L+Lorentz+real-SDP
% problem as the recent maxstep/updtransfo fixtures, covering both
% branches: wr.delta <= pars.beta (trial point kept) and wr.delta >
% pars.beta (falls back to t=0, wIN, wrIN).
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_trydif_oracle"

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

function xv = make_point(K2, seed_add, spread)
    rand('seed', 88 + seed_add);
    xv = rand(K2.N,1) - 0.5;
    xv(1:K2.l) = abs(xv(1:K2.l))*spread + 0.5;
    lorN = length(K2.q);
    xi = K2.l + lorN;
    for k = 1:lorN
        kk = K2.q(k) - 1;
        vecpart = xv(xi+1:xi+kk) * spread;
        xv(xi+1:xi+kk) = vecpart;
        xv(K2.l+k) = norm(vecpart) + 1;
        xi = xi + kk;
    end
    off = K2.lq;
    A1 = (rand(2) - 0.5) * spread; X1 = A1*A1' + 2*eye(2);
    xv(off+1:off+4) = X1(:);
    off = off + 4;
    A2 = (rand(3) - 0.5) * spread; X2 = A2*A2' + 3*eye(3);
    xv(off+1:off+9) = X2(:);
end

% Case 1: x close to z (small spread difference) -> should stay well
% inside the central region, wr.delta small, trial point kept.
x1 = make_point(K2, 1, 1.0);
z1 = make_point(K2, 2, 1.0);
wrIN1.desc = 1;
wIN1 = struct();
t1 = 0.5;
[t1_out, wr1, w1] = trydif(t1, wrIN1, wIN1, x1, z1, pars, K2);

% Case 2: x, z with a large spectral mismatch -> should push wr.delta
% above pars.beta, falling back to (t=0, wIN, wrIN).
x2 = make_point(K2, 3, 1.0);
z2 = make_point(K2, 4, 8.0);   % much larger scale -> pushes delta up
wrIN2.desc = 1;
wrIN2.delta = 0.1234; wrIN2.h = 5.678; wrIN2.alpha = 0.0321;
wIN2.lab = rand(size(w1.lab));
wIN2.tdetx = rand(size(w1.tdetx)); wIN2.tdetz = rand(size(w1.tdetz));
wIN2.ux = rand(size(w1.ux)); wIN2.s = rand(size(w1.s));
t2 = 0.7;
[t2_out, wr2, w2] = trydif(t2, wrIN2, wIN2, x2, z2, pars, K2);

save('-v7', fullfile(out_dir, 'trydif.mat'), ...
     'K2', 'pars', ...
     'x1', 'z1', 'wrIN1', 'wIN1', 't1', 't1_out', 'wr1', 'w1', ...
     'x2', 'z2', 'wrIN2', 'wIN2', 't2', 't2_out', 'wr2', 'w2');

fprintf('trydif oracle written to %s (wr1.delta=%g, wr2.delta=%g, beta=%g)\n', ...
    out_dir, wr1.delta, wr2.delta, pars.beta);
