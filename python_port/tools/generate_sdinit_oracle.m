% generate_sdinit_oracle.m
%
% Runs the real Octave sdinit.m (and Amul.m, both directly and via
% sdinit's own use of it) on a small mixed-cone problem after a real
% pretransfo.m call, and saves inputs+outputs.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_sdinit_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'sdinit');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 77);

%% Build a small mixed-cone external problem: L + Lorentz + real SDP.
%% Real (not complex) SDP blocks only: qrK's own existing ctypes binding
%% (_native.py, from cluster 3) only implements the real-symmetric-block
%% path, matching qrK.c's mexFunction only partially -- see its own
%% docstring. sdinit's own vfrm.s = qrK(d.u,K) would be silently wrong
%% for a complex block until that binding gap is filled, so this fixture
%% avoids exercising it for now.
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

[d, v, vfrm, y, y0, R] = sdinit(At2, b2, c2, dense, K2, pars);

save('-v7', fullfile(out_dir, 'sdinit_case1.mat'), ...
     'At2', 'b2', 'c2', 'K2', 'pars', 'dense', ...
     'd', 'v', 'vfrm', 'y', 'y0', 'R');

fprintf('sdinit oracle written to %s\n', out_dir);

%% Amul: direct test of both the transp=0/1 paths and the dense-column
%% correction branch (dense.cols nonempty), independent of getdense.m
%% (not yet ported) -- hand-built dense.A/dense.cols.
mA = 5; nA = 7;
Amat = rand(nA, mA) - 0.5;
Atsp = sparse(Amat);
xN = rand(nA,1) - 0.5;
xm = rand(mA,1) - 0.5;
dense2.cols = [2;5;7];
dense2.A = rand(mA, length(dense2.cols)) - 0.5;

y_transp0 = Amul(Atsp, dense2, xN, 0);
y_transp1 = Amul(Atsp, dense2, xm, 1);

dense_empty.cols = zeros(0,1);
dense_empty.A = sparse(mA,0);
y_transp0_nodense = Amul(Atsp, dense_empty, xN, 0);
y_transp1_nodense = Amul(Atsp, dense_empty, xm, 1);

save('-v7', fullfile(out_dir, 'amul.mat'), ...
     'Amat', 'xN', 'xm', 'dense2', ...
     'y_transp0', 'y_transp1', 'y_transp0_nodense', 'y_transp1_nodense');

fprintf('Amul oracle written to %s\n', out_dir);
