% generate_getdatm_oracle.m
%
% Runs the real Octave getDAtm.m (DAt.q computation only -- dense.q=[],
% so the DAt.denq/adendotd correction is never exercised) on the same
% mixed L+Lorentz+real-SDP problem as generate_sdinit_oracle.m.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_getdatm_oracle"

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

% Perturb d away from the trivial identity-multiple scaling (still a
% valid scaling-point shape) so DAt.q isn't trivially proportional to a
% single scalar.
d.q1 = d.q1 .* (1 + 0.1*rand(size(d.q1)));
d.q2 = d.q2 + 0.05*(rand(size(d.q2))-0.5);

% NOTE: sedumi.m itself always builds Ablkjc via partitA(A,K.mainblks)
% (a 3-boundary partition: L-end, Lorentz-trace-end, Lorentz-vec-end) --
% NOT K.blkstart (which has many more boundary columns, one per PSD
% block too). ddot.c's sparse path (spddotxj) hard-codes reading
% Ablkjc's columns 2 and 3 as its nonzero-index search range, so passing
% a differently-partitioned Ablkjc silently searches the wrong range and
% drops every Lorentz block but the first -- confirmed by direct
% experimentation (not an upstream bug: this is what the real call site
% actually passes).
Ablkjc = partitA(At2, K2.mainblks);
DAtdenq = sparse(length(b2), 0);

DAt = getDAtm(At2, Ablkjc, dense, DAtdenq, d, K2);

save('-v7', fullfile(out_dir, 'getdatm.mat'), ...
     'At2', 'K2', 'd', 'dense', 'DAt');

fprintf('getDAtm oracle written to %s\n', out_dir);
