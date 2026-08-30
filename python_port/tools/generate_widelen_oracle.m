% generate_widelen_oracle.m
%
% Runs the real Octave widelen.m on the same mixed L+Lorentz+real-SDP
% problem as the recent trydif/maxstep fixtures, with a genuine interior
% point xc/zc and a descent direction dx/dz scaled small enough that a
% full step (t=maxt) stays near-central, exercising widelen's bisection
% search over several iterations.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_widelen_oracle"

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

function xv = make_point(K2, seed_add)
    rand('seed', 88 + seed_add);
    xv = rand(K2.N,1) - 0.5;
    xv(1:K2.l) = abs(xv(1:K2.l)) + 1.0;
    lorN = length(K2.q);
    xi = K2.l + lorN;
    for k = 1:lorN
        kk = K2.q(k) - 1;
        vecpart = xv(xi+1:xi+kk);
        xv(K2.l+k) = norm(vecpart) + 1.5;
        xi = xi + kk;
    end
    off = K2.lq;
    A1 = rand(2) - 0.5; X1 = A1*A1' + 3*eye(2);
    xv(off+1:off+4) = X1(:);
    off = off + 4;
    A2 = rand(3) - 0.5; X2 = A2*A2' + 4*eye(3);
    xv(off+1:off+9) = X2(:);
end

xc = make_point(K2, 1);
zc = make_point(K2, 2);

% Small, generic step directions (not required to be a genuine sddir
% output -- widelen.m's own line search only needs xc+t*dx, zc+t*dz to
% stay valid Lorentz/PSD-cone-interior points for t in [0,maxt]).
rand('seed', 99);
dx = 0.1 * (rand(K2.N,1) - 0.5);
dz = 0.1 * (rand(K2.N,1) - 0.5);

y0 = 2.5;
dy0 = -0.3;     % descent direction
d2y0 = -0.05;   % concave decreasing, exercises the sqrt() branch of fullt
maxt = 1.0;

[t, wr, w] = widelen(xc, zc, y0, dx, dz, dy0, d2y0, maxt, pars, K2);

save('-v7', fullfile(out_dir, 'widelen.mat'), ...
     'K2', 'pars', 'xc', 'zc', 'y0', 'dx', 'dz', 'dy0', 'd2y0', 'maxt', ...
     't', 'wr', 'w');

fprintf('widelen oracle written to %s (t=%g, wr.delta=%g)\n', out_dir, t, wr.delta);
