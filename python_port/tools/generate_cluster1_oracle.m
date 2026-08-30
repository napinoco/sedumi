% generate_cluster1_oracle.m
%
% Runs the real Octave/MEX ddot, qblkmul, blkmul, vecsym, quadadd on
% fixed random test data and saves inputs+outputs, so
% python_port/tests/test_cluster1.py can check the Python ports against
% them without needing Octave installed to run.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_cluster1_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'cluster1');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 42);

%% ddot: dense X, several column counts
d = rand(6,1);
X = rand(6,4);
blkstart = [1;3;7]; % two blocks: [1,2], [3,4,5,6]  (1-indexed)
ddotX = ddot(d, X, blkstart);
save('-v7', fullfile(out_dir, 'ddot_dense.mat'), 'd', 'X', 'blkstart', 'ddotX');

%% qblkmul
mu = rand(3,1);
dq = rand(9,1);
blkstartq = [1;3;6;10];
yq = qblkmul(mu, dq, blkstartq);
save('-v7', fullfile(out_dir, 'qblkmul.mat'), 'mu', 'dq', 'blkstartq', 'yq');

% NOTE: blkmul.c is NOT part of the actual build (not in
% install_sedumi.m's target list, no blkmul.m wrapper either, and no .m
% file calls it) -- it's dead code, like bwblkslv2.c. It's still bound in
% _native.py (harmless), but there is no real MEX build to check it
% against, so it's skipped here.

%% vecsym: K.l=2, K.s=[2,3] (one 2x2 + one 3x3 real PSD block)
K.l = 2;
K.s = [2;3];
xlen = K.l + 4 + 9;
xv = rand(xlen,1);
yv = vecsym(xv, K);
save('-v7', fullfile(out_dir, 'vecsym.mat'), 'xv', 'yv');
Kl = K.l; Ks = K.s;
save('-v7', '-append', fullfile(out_dir, 'vecsym.mat'), 'Kl', 'Ks');

%% quadadd
xhi = [1e10; 3.5; -2.25];
xlo = [1e-10; 0.0001; 1e-8];
yqa = [2.0; 1.5; 100.25];
[zhi, zlo] = quadadd(xhi, xlo, yqa);
save('-v7', fullfile(out_dir, 'quadadd.mat'), 'xhi', 'xlo', 'yqa', 'zhi', 'zlo');

%% ddot: X given as a FULL, absolute-indexed array (blkstart(1) > 1), not
%% pre-sliced to the block span -- exercises the mexFunction's own
%% `X.pr += blkstart[0]` row-offset case (ddotxj() itself asserts
%% blkstart[0]==0 and never applies this offset itself). Appended at the
%% end (rather than alongside the other ddot test above) so it doesn't
%% shift the shared 'rand' stream and change the other, pre-existing
%% fixtures' random data.
d2 = rand(6,1);
X2 = rand(10,4);           % 4 extra rows before the block-span starts
blkstart2 = [5;7;11];      % blocks span absolute rows 5-6, 7-10 (1-indexed)
ddotX2 = ddot(d2, X2, blkstart2);
save('-v7', fullfile(out_dir, 'ddot_offset.mat'), 'd2', 'X2', 'blkstart2', 'ddotX2');

fprintf('Cluster 1 oracle written to %s\n', out_dir);
