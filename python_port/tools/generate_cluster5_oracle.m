% generate_cluster5_oracle.m
%
% Runs the real Octave/MEX sortnnz, cholsplit, mJdetd against small test
% data. finsymbden is verified separately, reusing the existing
% cluster2 dpr1fact fixtures (see test_cluster5.py) rather than new
% fixtures here.
%
% NOTE: incorder, iswnbr, symbfwblk are NOT covered -- see the cluster 5
% task notes for the scope discussion raised with the user (same
% category as getada1/getada2/getada3/adendotd/adenscale from cluster 4:
% deeply tied to sedumi.m's internal "dense columns" bookkeeping,
% deferred to Phase 3).
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_cluster5_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'cluster5');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 33);

%% sortnnz
At = sprandn(6, 5, 0.5);
perm = sortnnz(At, [], []);
save('-v7', fullfile(out_dir, 'sortnnz.mat'), 'At', 'perm');

%% cholsplit: needs a real symbolic factor L (perm, L, xsuper).
global ADA_sedumi_
X = sprandn(8, 8, 0.5); X = X*X' + 10*speye(8);  % SPD
ADA_sedumi_ = X;
L = symbchol();
cachesizeKB = 1;  % tiny cache to force actual splitting
split = cholsplit(L, cachesizeKB);
save('-v7', fullfile(out_dir, 'cholsplit.mat'), 'X', 'L', 'cachesizeKB', 'split');

% NOTE: mJdetd.c is NOT part of the actual build (not in
% install_sedumi.m's target list, no mJdetd.m wrapper, no .m file calls
% it) -- it's dead code, like blkmul.c (cluster 1) and bwblkslv2.c
% (Phase 1). Bound anyway (harmless) but not oracle-verified since no
% real MEX build of it exists to check against; see test_cluster5.py for
% an independent mathematical check instead.

fprintf('Cluster 5 oracle written to %s\n', out_dir);
