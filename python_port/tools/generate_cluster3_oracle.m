% generate_cluster3_oracle.m
%
% Runs the real Octave/MEX qrK, psdframeit, psdinvjmul, sqrtinv,
% givensrot, urotorder, invcholfac on small real-symmetric-PSD-only test
% data (K.s with no K.rsdpN override, i.e. no complex Hermitian blocks --
% those aren't covered by the Python port for this cluster) and saves
% inputs+outputs.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_cluster3_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'cluster3');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 11);

K.s = [3; 2];   % two real PSD blocks: 3x3 and 2x2
lenud = 3^2 + 2^2;

%% qrK: QR-factorize a random block-diagonal-ish vector.
x = rand(lenud, 1);
[frms, r] = qrK(x, K);
save('-v7', fullfile(out_dir, 'qrK.mat'), 'x', 'frms', 'r');

%% psdframeit: x = FRM*lab, using frms from qrK above (a genuine
%% orthogonal-matrix-in-product-form) and arbitrary eigenvalues lab.
lab = rand(3 + 2, 1) + 1;
xf = psdframeit(lab, frms, K);
save('-v7', fullfile(out_dir, 'psdframeit.mat'), 'lab', 'frms', 'xf');

%% sqrtinv: needs vlab of length K.l + 2*length(K.q) + sum(K.s) = sum(K.s) here.
vlab = rand(3 + 2, 1) + 1;
ysq = sqrtinv(frms, vlab, K);
save('-v7', fullfile(out_dir, 'sqrtinv.mat'), 'frms', 'vlab', 'ysq');

%% psdinvjmul: solve Xz+zX=2Y given eigenvalues `evx` and basis `frms`.
evx = rand(3 + 2, 1) + 1;
yin = rand(lenud, 1);
zout = psdinvjmul(evx, frms, yin, K);
save('-v7', fullfile(out_dir, 'psdinvjmul.mat'), 'frms', 'evx', 'yin', 'zout');

%% invcholfac: y = U'*U (no perm) and y = invperm(U'*U) (with perm).
u = rand(lenud, 1);
y_noperm = invcholfac(u, K);
permv = [randperm(3)'; 3 + randperm(2)'];
y_perm = invcholfac(u, K, permv);
save('-v7', fullfile(out_dir, 'invcholfac.mat'), 'u', 'y_noperm', 'permv', 'y_perm');

%% givensrot: apply a hand-built sequence of rotations to a PSD block.
%% Use urotorder's own output (gjc,g) on a random u to get a *valid*
%% rotation sequence, then re-apply it via givensrot and check it
%% reproduces the same permuted/rotated result urotorder itself gives.
maxu = 100;   % high threshold -> forces zero rotations sometimes, so use
              % a small one to force some pivoting on a badly-scaled u.
maxu = 1.01;
u2 = rand(lenud, 1) .* [1e-6*ones(9,1); ones(4,1)];   % ill-scaled -> pivots
[u_out, perm, gjc, g] = urotorder(u2, K, maxu);
xg = rand(lenud, 1);
yg = givensrot(gjc, g, xg, K);
save('-v7', fullfile(out_dir, 'urotorder.mat'), 'u2', 'maxu', 'u_out', 'perm', 'gjc', 'g');
save('-v7', fullfile(out_dir, 'givensrot.mat'), 'gjc', 'g', 'xg', 'yg', 'K');

save('-v7', fullfile(out_dir, 'K.mat'), 'K');
fprintf('Cluster 3 oracle written to %s\n', out_dir);
