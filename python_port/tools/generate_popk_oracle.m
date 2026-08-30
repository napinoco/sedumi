% generate_popk_oracle.m
%
% Runs the real Octave PopK.m on the same mixed L+Lorentz+real-SDP
% problem and scaling point `d` as generate_sdinit_oracle.m.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_popk_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'sdinit');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 77);

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

[d, v, vfrm, y, y0, R] = sdinit(At2, b2, c2, dense, K2, pars);

x = v + 0.05 * (rand(size(v)) - 0.5);

[y_full, ddotx_full, Dx_full, xTy_full] = PopK(d, x, K2);
y_lpq = PopK(d, x, K2, 1);

save('-v7', fullfile(out_dir, 'popk.mat'), ...
     'K2', 'd', 'x', 'y_full', 'ddotx_full', 'Dx_full', 'xTy_full', 'y_lpq');

fprintf('PopK oracle written to %s\n', out_dir);
