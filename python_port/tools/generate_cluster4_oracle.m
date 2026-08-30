% generate_cluster4_oracle.m
%
% Runs the real Octave/MEX partitA, findblks, extractA (the sparse
% row-range utilities used to assemble SeDuMi's Schur-complement matrix)
% against a fixed random sparse test matrix.
%
% NOTE: getada1/getada2 (no separable C core -- their whole computation
% lives directly in mexFunction) and getada3/adendotd/adenscale (deeply
% tied to sedumi.m's internal "dense columns" bookkeeping: dense.{cols,
% l,q,A}, d.{q1,q2} structs) are NOT covered by this oracle or by
% sedumi_port._native -- see the cluster 4 task notes for why these were
% deferred to Phase 3 rather than bound now.
%
% Run from the repository root:
%   octave-cli --no-gui --eval "cd python_port/tools; generate_cluster4_oracle"

pkg_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(pkg_root);
out_dir = fullfile(pkg_root, 'tests', 'fixtures', 'cluster4');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(repo_root);

rand('seed', 21);
m = 6; ncols = 4;
At = sprandn(m, ncols, 0.5);

blkstart3 = [1; 3; 5; 7];   % 3 row-blocks: [1,2], [3,4], [5,6]
Ablkjc = partitA(At, blkstart3);
save('-v7', fullfile(out_dir, 'partitA.mat'), 'At', 'blkstart3', 'Ablkjc');

blkstart2 = [2; 4; 7];      % 2 row-blocks: [2,3], [4,6] (findblks.c's
% mexFunction asserts blkstart(i)>=2 for every entry -- decrements it
% twice while converting to C-style, asserting positivity after each)
Ablkjc2 = partitA(At, blkstart2);
Ablk = findblks(At, Ablkjc2, 1, 2, blkstart2);
save('-v7', fullfile(out_dir, 'findblks.mat'), 'At', 'blkstart2', 'Ablkjc2', 'Ablk');

Apart = extractA(At, Ablkjc2, 2, 3, [3;7]);
save('-v7', fullfile(out_dir, 'extractA.mat'), 'At', 'Ablkjc2', 'Apart');

fprintf('Cluster 4 oracle written to %s\n', out_dir);
